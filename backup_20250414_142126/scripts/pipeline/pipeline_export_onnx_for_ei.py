#!/usr/bin/env python3
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import argparse
import json
import torch
from datetime import datetime
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
import onnx
import onnxruntime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.inference.inference import compare_models
from src.tagging.faiss_tagger import ModelTagger
from src.utils.model_io import load_model_and_info, save_model_and_info
from src.utils.data_handling import load_saved_dataset, prepare_data_for_training
from src.utils.metrics import calculate_accuracy
from src.utils.config import load_config
from src.utils.numpy_encoder import NumpyEncoder
from src.pruning.structural_prune import (
    analyze_hidden_layer,
    identify_pruning_candidates,
    analyze_neuron_sensitivity,
    create_pruned_model
)
from src.training.train_model import train_model
from src.export.onnx_converter import export_models_to_onnx, convert_to_onnx, make_mcu_compatible, validate_onnx_model
from src.export.profile_models import profile_models
from src.tagging.tag_onnx_models import tag_onnx_models
from src.inference.inference import run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)

def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the training phase of the pipeline."""
    logger.info("Starting model training...")
    
    training_config = {
        'dataset': config['dataset'],
        'model': config['model'],
        'training': config['training']
    }
    
    results = train_model(training_config, config['dataset']['name'])
    
    # Save model path for later use
    model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
    model_path = os.path.join(model_dir, f"{config['dataset']['name']}_model.pth")
    
    return {
        'model_dir': model_dir,
        'model_path': model_path,
        'metrics': results[1]
    }

def structural_prune(config: Dict[str, Any]) -> None:
    """Run structural pruning if enabled."""
    if config['pruning']['enabled']:
        logger.info("Starting model pruning...")
        
        # Get model paths
        model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
        
        # Check if the directory exists
        if not os.path.exists(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            return
            
        model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not model_dirs:
            logger.error(f"No model directories found in {model_dir}")
            return
            
        latest_model = max(model_dirs, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
        latest_model_dir = os.path.join(model_dir, latest_model)
        
        # Get model and info paths
        model_path = os.path.join(latest_model_dir, f"{config['dataset']['name']}_model.pth")
        model_info_path = os.path.join(latest_model_dir, f"{config['dataset']['name']}_model_info.json")
        
        # Load model and info
        model, model_info = load_model_and_info(latest_model_dir, config['dataset']['name'])
        
        # Get pruning parameters
        correlation_threshold = config['pruning']['correlation_threshold']
        activity_threshold = config['pruning']['activity_threshold']
        sensitivity_threshold = config['pruning']['sensitivity_threshold']
        
        # Load data for correlation analysis
        train_data_path = os.path.join(latest_model_dir, f"{config['dataset']['name']}_train_data.npz")
        test_data_path = os.path.join(latest_model_dir, f"{config['dataset']['name']}_test_data.npz")
        
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            logger.error(f"Training data not found at {train_data_path} or {test_data_path}")
            return
        
        # Load and prepare data
        train_data = np.load(train_data_path)
        test_data = np.load(test_data_path)
        
        # Convert to tensors with correct shapes
        X_train_tensor = torch.FloatTensor(train_data['X_train'])
        X_test_tensor = torch.FloatTensor(test_data['X_test'])
        
        # Ensure input tensors have shape (batch_size, input_features)
        if X_train_tensor.dim() == 1:
            X_train_tensor = X_train_tensor.unsqueeze(0)
        if X_test_tensor.dim() == 1:
            X_test_tensor = X_test_tensor.unsqueeze(0)
        
        # Convert labels to tensors
        num_classes = config['dataset']['num_classes']
        if num_classes > 2:  # Multiclass classification
            y_train_tensor = torch.LongTensor(train_data['y_train'])
            y_test_tensor = torch.LongTensor(test_data['y_test'])
        else:  # Binary classification
            y_train_tensor = torch.FloatTensor(train_data['y_train']).view(-1, 1)
            y_test_tensor = torch.FloatTensor(test_data['y_test']).view(-1, 1)
        
        # Analyze hidden layer for correlations
        logger.info("Analyzing hidden layer for correlations...")
        analysis_results = analyze_hidden_layer(
            model, 
            model.hidden,  # Specifically analyze the hidden layer
            0,  # layer_idx
            X_train_tensor,  # validation_data
            enhanced_pruning=True,
            activation=getattr(model, 'activation_name', 'relu')
        )
        
        # Extract correlation results
        correlation_results = {
            'correlation_mask': analysis_results.get('correlation_mask', torch.zeros(model.hidden.out_features)),
            'sensitivity_mask': analysis_results.get('sensitivity_mask', torch.zeros(model.hidden.out_features)),
            'activity_mask': analysis_results.get('activity_mask', torch.zeros(model.hidden.out_features)),
            'importance_scores': analysis_results.get('importance_scores', torch.zeros(model.hidden.out_features)),
            'correlated_pairs': analysis_results.get('correlated_pairs', [])
        }
        
        # Log correlation results
        logger.info(f"Found {len(correlation_results['correlated_pairs'])} correlated pairs in hidden layer")
        logger.info(f"Correlation mask: {correlation_results['correlation_mask'].sum()} neurons")
        logger.info(f"Sensitivity mask: {correlation_results['sensitivity_mask'].sum()} neurons")
        logger.info(f"Activity mask: {correlation_results['activity_mask'].sum()} neurons")
        
        # Identify pruning candidates based on correlation analysis
        pruning_candidates = []
        
        # Add correlated neurons from pairs
        for neuron1, neuron2 in correlation_results['correlated_pairs']:
            # Keep the neuron with higher importance score
            if correlation_results['importance_scores'][neuron1] > correlation_results['importance_scores'][neuron2]:
                pruning_candidates.append(neuron2)
            else:
                pruning_candidates.append(neuron1)
        
        # Add neurons from correlation mask
        correlated_indices = torch.where(correlation_results['correlation_mask'])[0].tolist()
        pruning_candidates.extend(correlated_indices)
        
        # Add neurons from sensitivity mask
        insensitive_indices = torch.where(correlation_results['sensitivity_mask'])[0].tolist()
        pruning_candidates.extend(insensitive_indices)
        
        # Add neurons from activity mask
        inactive_indices = torch.where(correlation_results['activity_mask'])[0].tolist()
        pruning_candidates.extend(inactive_indices)
        
        # Remove duplicates and sort
        pruning_candidates = sorted(list(set(pruning_candidates)))
        
        # Limit number of neurons to prune based on config
        max_neurons = config['pruning']['validation'].get('max_neurons_pruned', model.hidden.out_features // 2)
        if len(pruning_candidates) > max_neurons:
            # Sort by importance score and take the least important ones
            pruning_candidates.sort(key=lambda x: correlation_results['importance_scores'][x])
            pruning_candidates = pruning_candidates[:max_neurons]
        
        # Create pruning info
        pruning_info = {
            'pruned_indices': pruning_candidates,
            'correlation_results': {
                'correlation_mask': correlation_results['correlation_mask'].cpu().numpy().tolist(),
                'sensitivity_mask': correlation_results['sensitivity_mask'].cpu().numpy().tolist(),
                'activity_mask': correlation_results['activity_mask'].cpu().numpy().tolist(),
                'importance_scores': correlation_results['importance_scores'].cpu().numpy().tolist(),
                'correlated_pairs': correlation_results['correlated_pairs']
            },
            'original_hidden_neurons': model.hidden.out_features,
            'pruned_hidden_neurons': model.hidden.out_features - len(pruning_candidates),
            'activation': getattr(model, 'activation_name', 'relu')
        }
        
        # Adjust weights using linear regression for correlated neurons
        logger.info("Adjusting weights using linear regression for correlated neurons...")
        with torch.no_grad():
            # Get original hidden layer output
            original_hidden_output = model.hidden(X_train_tensor)
            
            # Create temporary tensors for adjusted weights
            adjusted_hidden_weights = model.hidden.weight.data.clone()
            adjusted_output_weights = model.output.weight.data.clone()
            
            # For each pruned neuron, find the best linear combination of kept neurons
            for pruned_idx in pruning_candidates:
                # Get the output of the pruned neuron
                pruned_neuron_output = original_hidden_output[:, pruned_idx].cpu().numpy()
                
                # Get indices of neurons to keep
                kept_indices = [i for i in range(model.hidden.out_features) if i not in pruning_candidates]
                
                # Get the outputs of the kept neurons
                kept_neurons_output = original_hidden_output[:, kept_indices].cpu().numpy()
                
                # Fit linear regression
                reg = LinearRegression()
                reg.fit(kept_neurons_output, pruned_neuron_output)
                
                # Adjust weights of the output layer
                for i in range(model.output.out_features):
                    # Get the weight of the pruned neuron in the output layer
                    pruned_weight = model.output.weight.data[i, pruned_idx].item()
                    
                    # Distribute this weight among the kept neurons based on the regression coefficients
                    for j, kept_idx in enumerate(kept_indices):
                        adjusted_output_weights[i, kept_idx] += pruned_weight * reg.coef_[j]
                    
                    # Remove the weight of the pruned neuron
                    adjusted_output_weights[i, pruned_idx] = 0
            
            # Create pruned model
            if config['dataset']['num_classes'] == 2:
                from src.models.binary_model import BinaryModel
                pruned_model = BinaryModel(
                    input_size=model.hidden.in_features,
                    hidden_size=model.hidden.out_features,
                    activation=model.activation_name
                )
            else:
                from src.models.categorical_model import CategoricalModel
                pruned_model = CategoricalModel(
                    input_size=model.hidden.in_features,
                    hidden_size=model.hidden.out_features,
                    output_size=model.output.out_features,
                    activation=model.activation_name
                )
            
            # Apply adjusted weights
            pruned_model.hidden.weight.data = adjusted_hidden_weights
            pruned_model.output.weight.data = adjusted_output_weights
            pruned_model.hidden.bias.data = model.hidden.bias.data
            pruned_model.output.bias.data = model.output.bias.data
        
        # Evaluate original model
        model.eval()
        with torch.no_grad():
            # Ensure input tensor has correct shape (batch_size, input_features)
            if len(X_test_tensor.shape) == 1:
                X_test_tensor = X_test_tensor.unsqueeze(0)  # Add batch dimension
            elif len(X_test_tensor.shape) == 2 and X_test_tensor.shape[1] != model.hidden.in_features:
                # Transpose if needed to match expected input shape
                X_test_tensor = X_test_tensor.t()
            
            original_outputs, _ = model(X_test_tensor)
            # Ensure outputs have correct shape for binary classification
            if config['dataset']['num_classes'] <= 2:
                original_outputs = original_outputs.view(-1, 1)
            original_accuracy = calculate_accuracy(original_outputs, y_test_tensor, config['dataset']['num_classes'])
        
        # Evaluate pruned model
        pruned_model.eval()
        with torch.no_grad():
            # Use the same reshaped input tensor
            pruned_outputs, _ = pruned_model(X_test_tensor)
            # Log shapes for debugging
            logger.info(f"pruned_outputs shape: {pruned_outputs.shape}")
            logger.info(f"y_test_tensor shape: {y_test_tensor.shape}")
            # Ensure outputs have correct shape for binary classification
            if config['dataset']['num_classes'] <= 2:
                pruned_outputs = pruned_outputs.view(-1, 1)
            pruned_accuracy = calculate_accuracy(pruned_outputs, y_test_tensor, config['dataset']['num_classes'])
        
        logger.info(f"Original model accuracy: {original_accuracy:.2f}%")
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2f}%")
        
        # Save pruned model and info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pruned_model_dir = os.path.join(config['pruning']['save_dir'], config['dataset']['name'])
        os.makedirs(pruned_model_dir, exist_ok=True)
        
        pruned_model_path = os.path.join(pruned_model_dir, f"{config['dataset']['name']}_pruned_{pruned_model.hidden.out_features}_{timestamp}")
        os.makedirs(pruned_model_path, exist_ok=True)
        
        # Save model using save_model_and_info
        save_model_and_info(
            pruned_model,
            pruned_model_path,
            config['dataset']['name'],
            'pruned'
        )
        
        # Update pruning info with accuracy metrics
        pruning_info['original_accuracy'] = original_accuracy
        pruning_info['pruned_accuracy'] = pruned_accuracy
        pruning_info['accuracy_drop'] = original_accuracy - pruned_accuracy
        
        # Save pruning metrics
        metrics_path = os.path.join(config['output']['results_dir'], config['dataset']['name'], f"{config['dataset']['name']}_pruned_metrics_{timestamp}.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(pruning_info, f, indent=4, cls=NumpyEncoder)
        
        # Print clear information about pruning results
        num_neurons_pruned = len(pruning_info['pruned_indices'])
        num_original_neurons = pruning_info['original_hidden_neurons']
        pruning_percentage = (num_neurons_pruned / num_original_neurons) * 100
        
        logger.info(f"PRUNING SUMMARY: {num_neurons_pruned} out of {num_original_neurons} neurons were pruned ({pruning_percentage:.1f}%)")
        logger.info(f"Original accuracy: {original_accuracy:.2f}%")
        logger.info(f"Pruned accuracy: {pruned_accuracy:.2f}%")
        logger.info(f"Accuracy drop: {pruning_info['accuracy_drop']*100:.2f}%")
        
        logger.info(f"Pruned model saved to {pruned_model_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("Model pruning completed")
    else:
        logger.info("Pruning step disabled, skipping...")

def run_inference_steps(config: Dict[str, Any]) -> None:
    """Run inference steps if enabled."""
    if config['inference']['enabled']:
        logger.info("Starting inference steps...")
        
        # Get the latest model directories
        original_model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
        pruned_model_dir = os.path.join(config['pruning']['save_dir'], config['dataset']['name'])
        
        # Get the latest model folders
        original_dirs = [d for d in os.listdir(original_model_dir) if os.path.isdir(os.path.join(original_model_dir, d))]
        latest_original = max(original_dirs, key=lambda x: os.path.getctime(os.path.join(original_model_dir, x)))
        latest_original_dir = os.path.join(original_model_dir, latest_original)
        
        pruned_dirs = [d for d in os.listdir(pruned_model_dir) if os.path.isdir(os.path.join(pruned_model_dir, d))]
        latest_pruned = max(pruned_dirs, key=lambda x: os.path.getctime(os.path.join(pruned_model_dir, x)))
        latest_pruned_dir = os.path.join(pruned_model_dir, latest_pruned)
        
        # Load data
        train_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_train_data.npz")
        test_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_test_data.npz")
        data = load_saved_dataset(train_path, test_path)
        
        # Convert to tensors with correct shape
        X_train_tensor = torch.FloatTensor(data[0])
        X_test_tensor = torch.FloatTensor(data[1])
        if config['dataset']['num_classes'] > 2:
            y_train_tensor = torch.LongTensor(data[2])
            y_test_tensor = torch.LongTensor(data[3])
        else:
            y_train_tensor = torch.FloatTensor(data[2]).view(-1, 1)
            y_test_tensor = torch.FloatTensor(data[3]).view(-1, 1)
        
        data_tuple = (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
        
        # Load original model
        original_model_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_model.pth")
        original_model, original_info = load_model_and_info(latest_original_dir, config['dataset']['name'])
        
        # Load pruned model
        pruned_model_path = os.path.join(latest_pruned_dir, f"{config['dataset']['name']}_model.pth")
        pruned_model, pruned_info = load_model_and_info(latest_pruned_dir, config['dataset']['name'])
        
        # Run inference on original model
        logger.info("Running inference on original model...")
        original_results = run_inference(original_model, data_tuple)
        
        # Run inference on pruned model
        logger.info("Running inference on pruned model...")
        pruned_results = run_inference(pruned_model, data_tuple)
        
        # Save results
        results_dir = os.path.join(config['inference']['output_dir'], config['dataset']['name'])
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'original_model': {
                'accuracy': original_results['metrics']['accuracy'],
                'precision': original_results['metrics']['precision'],
                'recall': original_results['metrics']['recall'],
                'f1': original_results['metrics']['f1']
            },
            'pruned_model': {
                'accuracy': pruned_results['metrics']['accuracy'],
                'precision': pruned_results['metrics']['precision'],
                'recall': pruned_results['metrics']['recall'],
                'f1': pruned_results['metrics']['f1']
            },
            'timestamp': timestamp
        }
        
        results_path = os.path.join(results_dir, f"inference_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Original model accuracy: {original_results['metrics']['accuracy']:.2f}%")
        logger.info(f"Pruned model accuracy: {pruned_results['metrics']['accuracy']:.2f}%")
        logger.info(f"Results saved to {results_path}")
        logger.info("Inference steps completed")
    else:
        logger.info("Inference steps disabled, skipping...")

def clear_faiss_tags(results_dir: str) -> None:
    """Clear all FAISS tags from the results directory."""
    faiss_index_dir = os.path.join(results_dir, 'faiss_index')
    if os.path.exists(faiss_index_dir):
        logger.info(f"Clearing FAISS tags from {faiss_index_dir}")
        # Remove all files in the faiss_index directory
        for file in os.listdir(faiss_index_dir):
            file_path = os.path.join(faiss_index_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        logger.info("FAISS tags cleared successfully")
    else:
        logger.info("No FAISS tags found to clear")

def delete_all_results(results_dir: str) -> None:
    """Delete all results directories and files."""
    logger.info(f"Deleting all results from {results_dir}")
    
    # List of directories to delete
    dirs_to_delete = [
        'results',
        'models',
        'results/faiss_index',
        'results/inference',
        'models/original',
        'models/pruned'
    ]
    
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            try:
                import shutil
                shutil.rmtree(dir_path)
                logger.info(f"Deleted directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error deleting {dir_path}: {e}")
    
    logger.info("All results deleted successfully")

def run_onnx_export(config: Dict[str, Any]) -> None:
    """Run ONNX export if enabled."""
    if config['export'].get('enable_onnx', False):
        logger.info("Starting ONNX export...")
        
        # Get dataset name and activation function
        dataset_name = config['dataset']['name']
        activation = config['model']['activation']
        
        # Get the latest model directories
        original_model_dir = os.path.join(config['training']['save_dir'], dataset_name)
        pruned_model_dir = os.path.join(config['pruning']['save_dir'], dataset_name)
        
        # Get the latest model folders
        original_dirs = [d for d in os.listdir(original_model_dir) if os.path.isdir(os.path.join(original_model_dir, d))]
        latest_original = max(original_dirs, key=lambda x: os.path.getctime(os.path.join(original_model_dir, x)))
        latest_original_dir = os.path.join(original_model_dir, latest_original)
        
        pruned_dirs = [d for d in os.listdir(pruned_model_dir) if os.path.isdir(os.path.join(pruned_model_dir, d))]
        latest_pruned = max(pruned_dirs, key=lambda x: os.path.getctime(os.path.join(pruned_model_dir, x)))
        latest_pruned_dir = os.path.join(pruned_model_dir, latest_pruned)
        
        # Create ONNX directories
        original_onnx_dir = os.path.join("models", "original", activation, dataset_name)
        pruned_onnx_dir = os.path.join("models", "pruned", activation, dataset_name)
        os.makedirs(original_onnx_dir, exist_ok=True)
        os.makedirs(pruned_onnx_dir, exist_ok=True)
        
        # Export original model to ONNX
        logger.info("Exporting original model to ONNX...")
        original_model_path = os.path.join(latest_original_dir, f"{dataset_name}_model.pth")
        original_onnx_path = os.path.join(original_onnx_dir, f"{dataset_name}_model.onnx")
        
        success = convert_to_onnx(
            model_path=original_model_path,
            onnx_path=original_onnx_path,
            input_shape=(1, config['dataset']['input_features']),
            hidden_size=config['model']['hidden_neurons'],
            output_size=config['dataset']['num_classes'],
            activation=config['model']['activation'],
            opset_version=11
        )
        
        if not success:
            logger.error("Failed to export original model to ONNX")
            return
            
        # Create MCU compatible version of original model
        original_mcu_path = os.path.join(original_onnx_dir, f"{dataset_name}_model_mcu.onnx")
        if make_mcu_compatible(original_onnx_path, original_mcu_path):
            logger.info(f"Created MCU compatible version: {original_mcu_path}")
        
        # Export pruned model to ONNX
        logger.info("Exporting pruned model to ONNX...")
        pruned_model_path = os.path.join(latest_pruned_dir, f"{dataset_name}_model.pth")
        pruned_onnx_path = os.path.join(pruned_onnx_dir, f"{dataset_name}_model.onnx")
        
        success = convert_to_onnx(
            model_path=pruned_model_path,
            onnx_path=pruned_onnx_path,
            input_shape=(1, config['dataset']['input_features']),
            hidden_size=config['model']['hidden_neurons'],
            output_size=config['dataset']['num_classes'],
            activation=config['model']['activation'],
            opset_version=11
        )
        
        if not success:
            logger.error("Failed to export pruned model to ONNX")
            return
            
        # Create MCU compatible version of pruned model
        pruned_mcu_path = os.path.join(pruned_onnx_dir, f"{dataset_name}_model_mcu.onnx")
        if make_mcu_compatible(pruned_onnx_path, pruned_mcu_path):
            logger.info(f"Created MCU compatible version: {pruned_mcu_path}")
        
        # Validate the exported models
        test_data_path = os.path.join(latest_original_dir, f"{dataset_name}_test_data.npz")
        if os.path.exists(test_data_path):
            logger.info("Validating exported models...")
            original_acc = validate_onnx_model(original_onnx_path, test_data_path)
            pruned_acc = validate_onnx_model(pruned_onnx_path, test_data_path)
            logger.info(f"Original model accuracy: {original_acc:.2%}")
            logger.info(f"Pruned model accuracy: {pruned_acc:.2%}")
        
        logger.info("ONNX export completed successfully")
    else:
        logger.info("ONNX export disabled, skipping...")

def tag_onnx_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Tag ONNX models with profiling results."""
    logger.info("Starting ONNX model tagging...")
    
    # Get dataset name and activation function
    dataset_name = config['dataset']['name']
    activation = config['model']['activation']
    
    # Get the latest model directories
    original_model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
    pruned_model_dir = os.path.join(config['pruning']['save_dir'], config['dataset']['name'])
    
    # Get the latest model folders
    original_dirs = [d for d in os.listdir(original_model_dir) if os.path.isdir(os.path.join(original_model_dir, d))]
    latest_original = max(original_dirs, key=lambda x: os.path.getctime(os.path.join(original_model_dir, x)))
    latest_original_dir = os.path.join(original_model_dir, latest_original)
    
    pruned_dirs = [d for d in os.listdir(pruned_model_dir) if os.path.isdir(os.path.join(pruned_model_dir, d))]
    latest_pruned = max(pruned_dirs, key=lambda x: os.path.getctime(os.path.join(pruned_model_dir, x)))
    latest_pruned_dir = os.path.join(pruned_model_dir, latest_pruned)
    
    # Construct paths for original and pruned models
    original_model_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_model.pth")
    pruned_model_path = os.path.join(latest_pruned_dir, f"{config['dataset']['name']}_model.pth")
    original_model_info_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_model_info.json")
    pruned_model_info_path = os.path.join(latest_pruned_dir, f"{config['dataset']['name']}_model_info.json")
    
    # ONNX model paths - using the correct directory structure
    original_onnx_path = os.path.join("models", "original", activation, f"{dataset_name}_model", f"{dataset_name}_model.onnx")
    pruned_onnx_path = os.path.join("models", "pruned", activation, f"{dataset_name}_model", f"{dataset_name}_model.onnx")
    
    # Verify paths exist
    if not os.path.exists(original_onnx_path):
        raise FileNotFoundError(f"Original ONNX model not found at {original_onnx_path}")
    if not os.path.exists(pruned_onnx_path):
        raise FileNotFoundError(f"Pruned ONNX model not found at {pruned_onnx_path}")
    if not os.path.exists(original_model_info_path):
        raise FileNotFoundError(f"Original model info not found at {original_model_info_path}")
    if not os.path.exists(pruned_model_info_path):
        raise FileNotFoundError(f"Pruned model info not found at {pruned_model_info_path}")
    
    # Load model info
    with open(original_model_info_path, 'r') as f:
        original_model_info = json.load(f)
    with open(pruned_model_info_path, 'r') as f:
        pruned_model_info = json.load(f)
    
    # Create tagging index directory
    tagging_index_dir = os.path.join("results", "tagging_index", config['model']['activation'])
    os.makedirs(tagging_index_dir, exist_ok=True)
    
    # Initialize tagger
    tagger = ModelTagger(tagging_index_dir)
    
    # Tag original model
    logger.info(f"Tagging original model at {original_onnx_path}")
    original_tag = tagger.tag_model(original_onnx_path, original_model_info)
    
    # Tag pruned model
    logger.info(f"Tagging pruned model at {pruned_onnx_path}")
    pruned_tag = tagger.tag_model(pruned_onnx_path, pruned_model_info)
    
    # Save tagging results
    tagging_results = {
        'original': original_tag,
        'pruned': pruned_tag
    }
    
    # Save tagging results to file
    results_dir = os.path.join("results", "tagging", config['dataset']['name'])
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config['dataset']['name']}_tagging_results.json")
    
    with open(results_path, 'w') as f:
        json.dump(tagging_results, f, indent=4)
    
    logger.info(f"Tagging results saved to {results_path}")
    
    return tagging_results

def run_edge_impulse_export(config: dict) -> bool:
    """
    Run Edge Impulse export pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        # Get model paths from config
        model_path = config.get('model_path')
        if not model_path:
            logger.error("Model path not found in config")
            return False
            
        # Create Edge Impulse export directory
        ei_export_dir = os.path.join(os.path.dirname(model_path), 'edge_impulse')
        os.makedirs(ei_export_dir, exist_ok=True)
        
        # Get input shape from config
        input_shape = config.get('input_shape', (1, config.get('num_features', 10)))
        
        # Export for Edge Impulse
        ei_model_path = os.path.join(ei_export_dir, 'model_ei.onnx')
        success = export_for_edge_impulse(
            model_path=model_path,
            output_path=ei_model_path,
            input_shape=input_shape
        )
        
        if success:
            # Update config with Edge Impulse model path
            config['edge_impulse_model_path'] = ei_model_path
            logger.info(f"Edge Impulse export completed successfully: {ei_model_path}")
            return True
        else:
            logger.error("Edge Impulse export failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in Edge Impulse export pipeline: {str(e)}")
        return False

def export_for_edge_impulse(model_path: str, output_path: str, input_shape: tuple) -> bool:
    """
    Export a model specifically for Edge Impulse deployment.
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the Edge Impulse compatible model
        input_shape: Shape of the input tensor (batch_size, features)
        
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        # Load the model
        model = load_model(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, dtype=torch.float32)
        
        # Export with Edge Impulse specific settings
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # Edge Impulse requires opset 11
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare PyTorch and ONNX outputs
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()
        
        # Check if outputs match within tolerance
        np.testing.assert_allclose(ort_outputs[0], torch_output, rtol=1e-03, atol=1e-05)
        
        logger.info(f"Successfully exported model for Edge Impulse: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting model for Edge Impulse: {str(e)}")
        return False

def load_model(model_path: str):
    """
    Load a PyTorch model from the given path.
    
    Args:
        model_path: Path to the PyTorch model file
        
    Returns:
        The loaded PyTorch model
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model info if available
    model_info_path = model_path.replace('.pth', '_info.json')
    model_info = {}
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
    
    # Determine model type from info or path
    is_binary = False
    if model_info:
        is_binary = model_info.get('num_classes', 0) == 2
    else:
        # Try to infer from path
        is_binary = 'binary' in model_path.lower()
    
    # Get model parameters
    input_size = model_info.get('input_features', 10)
    hidden_size = model_info.get('hidden_neurons', 16)
    output_size = model_info.get('num_classes', 2)
    activation = model_info.get('activation', 'relu')
    
    # Create appropriate model
    if is_binary:
        from src.models.binary_model import BinaryModel
        model = BinaryModel(input_size, hidden_size, activation=activation)
    else:
        from src.models.categorical_model import CategoricalModel
        model = CategoricalModel(input_size, hidden_size, output_size, activation=activation)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model

def main():
    """Main function to run the training and pruning pipeline."""
    parser = argparse.ArgumentParser(description='Run the neural network optimization pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Run training if enabled
    run_training(config)
    
    # Run structural pruning if enabled
    structural_prune(config)
    
    # Export models to ONNX
    if config.get('export', {}).get('enabled', False):
        logger.info("Starting ONNX export...")
        export_models_to_onnx(config)
        
        # Profile the exported models only if Edge Impulse is enabled
        if config.get('export', {}).get('edge_impulse', {}).get('enabled', False):
            logger.info("Starting model profiling...")
            profiling_results = profile_models(config)
            if profiling_results:
                logger.info("Model profiling completed successfully")
                logger.info(f"Profiling results: {profiling_results}")
            else:
                logger.warning("Model profiling failed or was skipped")
                
            # Tag the models with profiling results
            logger.info("Tagging models with profiling results...")
            tagging_results = tag_onnx_models(config)
            if tagging_results:
                logger.info("Model tagging completed successfully")
                logger.info(f"Tagged models: {tagging_results}")
            else:
                logger.warning("Model tagging failed or was skipped")
        else:
            logger.info("Edge Impulse profiling disabled, skipping...")
            
        # Run inference steps if enabled
        logger.info("Starting inference steps...")
        run_inference_steps(config)
    else:
        logger.info("ONNX export disabled, skipping...")
        
    # Export for Edge Impulse
    if config.get('run_edge_impulse_export', True):
        run_edge_impulse_export(config)
        
    logger.info("Training and pruning pipeline execution finished successfully")
    logger.info(f"Results saved to: {config['output']['results_dir']}")

if __name__ == "__main__":
    main() 