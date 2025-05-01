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

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.inference.inference import compare_models
from src.inference.quantized_inference import run_quantized_inference, compare_quantized_models
from src.quantization.quantize import quantize_model
from src.tagging.faiss_tagger import ModelTagger
from src.utils.model_io import load_model_and_info
from src.utils.data_handling import load_saved_dataset, prepare_data_for_training
from src.utils.metrics import calculate_accuracy
from src.pruning.structural_prune import (
    analyze_hidden_layer,
    identify_pruning_candidates,
    analyze_neuron_sensitivity,
    create_pruned_model
)
from src.utils.config import load_config
from src.export.onnx_converter import export_models_to_onnx
from src.export.profile_models import profile_models
from src.tagging.tag_onnx_models import tag_onnx_models

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

def run_training(config: Dict[str, Any]) -> None:
    """Run model training if enabled."""
    if config['training']['enabled']:
        logger.info("Starting model training...")
        # Import and run training module
        from src.training.train_model import train_model
        
        # Initialize model tagger
        tagger = ModelTagger(os.path.join(config['output']['results_dir'], 'faiss_index'))
        
        # Adapt config structure for training module
        training_config = {
            'dataset_path': config['dataset']['path'],
            'hidden_neurons': config['model']['hidden_neurons'],
            'activation': config['model'].get('activation', 'relu'),
            'epochs': config['model'].get('epochs', 1000),
            'learning_rate': config['model'].get('learning_rate', 0.001),
            'test_size': config['dataset'].get('test_size', 0.2),
            'random_state': config['dataset'].get('random_state', 42),
            'save_dir': config['training']['save_dir']
        }
        
        # Train model and get results
        results = train_model(training_config)
        
        # Extract metrics for FAISS tagging
        model_metrics = {
            'accuracy': results['metrics']['test_accuracy'],
            'inference_time': 0.0,  # Will be measured during inference
            'model_size': results['metrics']['model_size'],
            'num_layers': 2,  # Fixed for our architecture
            'hidden_neurons': results['metrics']['hidden_neurons'],
            'total_parameters': results['metrics']['model_size'],
            'epochs': results['metrics']['epochs'],
            'learning_rate': results['metrics']['learning_rate'],
            'dataset_name': config['dataset']['name']
        }
        
        # Tag the trained model
        model_path = results['paths']['model_path']
        model_id = tagger.tag_model(model_path, model_metrics)
        logger.info(f"Tagged trained model with ID: {model_id}")
        
        logger.info("Model training completed")
    else:
        logger.info("Training step disabled, skipping...")

def structural_prune(config: Dict[str, Any]) -> None:
    """Run structural pruning if enabled."""
    if config['pruning']['enabled']:
        logger.info("Starting model pruning...")
        
        # Get model paths
        model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
        model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
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
        train_data_path = os.path.join(model_dir, f"{config['dataset']['name']}_train_data.npz")
        test_data_path = os.path.join(model_dir, f"{config['dataset']['name']}_test_data.npz")
        
        # Load and prepare data
        train_data = np.load(train_data_path)
        test_data = np.load(test_data_path)
        
        X_train = train_data['X_train']
        y_train = train_data['y_train']
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Convert data to PyTorch tensors
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = prepare_data_for_training(
            X_train, X_test, y_train, y_test, config['dataset']['num_classes']
        )
        
        # Get hidden layer activations
        model.eval()
        with torch.no_grad():
            if config['dataset']['num_classes'] > 2:
                _, hidden_layer_output = model(X_train_tensor)
            else:
                _, hidden_layer_output = model(X_train_tensor)
        
        # Analyze hidden layer for correlations
        correlation_results = analyze_hidden_layer(
            hidden_layer_output, 
            correlation_threshold, 
            sensitivity_threshold, 
            activity_threshold,
            config['model']['activation']
        )
        
        # Identify pruning candidates
        pruning_candidates = identify_pruning_candidates(
            model, 
            hidden_layer_output, 
            config['model']['activation'], 
            activity_threshold
        )
        
        # Analyze neuron sensitivity
        insensitive_neurons, sensitivity_scores = analyze_neuron_sensitivity(
            model, 
            X_train_tensor, 
            hidden_layer_output, 
            sensitivity_threshold
        )
        
        # Combine pruning candidates
        pruning_candidates = set(pruning_candidates)
        pruning_candidates.update(insensitive_neurons)
        
        # Add correlated neurons
        for pair in correlation_results['correlated_neurons']:
            i, j = pair
            if j in pruning_candidates:  # Only add if the dependent neuron is already a candidate
                pruning_candidates.add(i)
        
        # Initialize pruning info
        pruning_info = {
            'dataset_name': config['dataset']['name'],
            'original_hidden_neurons': model.hidden.out_features,
            'pruned_hidden_neurons': model.hidden.out_features - len(pruning_candidates),
            'neurons_pruned': len(pruning_candidates),
            'correlation_threshold': correlation_threshold,
            'activity_threshold': activity_threshold,
            'sensitivity_threshold': sensitivity_threshold,
            'pruned_indices': list(pruning_candidates),
            'correlation_results': correlation_results,
            'low_activity_indices': list(pruning_candidates),
            'insensitive_indices': insensitive_neurons,
            'sensitivity_scores': sensitivity_scores.tolist(),
            'activation': config['model']['activation'],
            'num_features': config['dataset']['input_features']
        }
        
        # Create pruned model with new weight adjustment approach
        pruned_model = create_pruned_model(
            model, 
            pruning_info, 
            config['dataset']['input_features'],
            model.hidden.out_features,
            config['dataset']['num_classes'],
            X_train_tensor
        )
        
        # Evaluate pruned model
        pruned_model.eval()
        with torch.no_grad():
            if config['dataset']['num_classes'] > 2:
                pruned_outputs, _ = pruned_model(X_test_tensor)
            else:
                pruned_outputs, _ = pruned_model(X_test_tensor)
            
            pruned_accuracy = calculate_accuracy(pruned_outputs, y_test_tensor, config['dataset']['num_classes'])
        
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2f}%")
        
        # Save pruned model
        pruned_model_dir = os.path.join(config['pruning']['save_dir'], config['dataset']['name'])
        os.makedirs(pruned_model_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create pruned model directory
        pruned_model_path = os.path.join(pruned_model_dir, f"{config['dataset']['name']}_pruned_{pruned_model.hidden.out_features}_{timestamp}")
        os.makedirs(pruned_model_path, exist_ok=True)
        
        # Save pruned model weights
        torch.save(pruned_model.state_dict(), os.path.join(pruned_model_path, f"{config['dataset']['name']}_model.pth"))
        
        # Save model info
        model_info.update({
            'pruned_indices': pruning_info['pruned_indices'],
            'correlation_results': pruning_info['correlation_results'],
            'low_activity_neurons': pruning_info['low_activity_indices'],
            'correlated_neurons': pruning_info['correlation_results']['correlated_neurons'],
            'original_hidden_neurons': model_info['hidden_neurons'],
            'pruned_hidden_neurons': pruned_model.hidden.out_features
        })
        
        with open(os.path.join(pruned_model_path, f"{config['dataset']['name']}_model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4)
        
        # Save pruning metrics
        metrics_path = os.path.join(config['output']['results_dir'], config['dataset']['name'], f"{config['dataset']['name']}_pruned_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(pruning_info, f, indent=4)
        
        logger.info(f"Pruned model saved to {pruned_model_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Update config with pruning information
        config['pruning'].update({
            'pruned_indices': pruning_info['pruned_indices'],
            'correlation_results': pruning_info['correlation_results'],
            'low_activity_neurons': pruning_info['low_activity_indices'],
            'correlated_neurons': pruning_info['correlation_results']['correlated_neurons'],
            'original_hidden_neurons': model_info['hidden_neurons'],
            'pruned_hidden_neurons': pruned_model.hidden.out_features,
            'final_accuracy': pruned_accuracy
        })
        
        # Save updated config to results directory
        config_path = os.path.join(config['output']['results_dir'], config['dataset']['name'], f"{config['dataset']['name']}_pruned_config_{timestamp}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Also update the original config file
        original_config_path = os.path.join('config', f'config_{config["dataset"]["name"]}.yaml')
        if os.path.exists(original_config_path):
            with open(original_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Updated original config file at {original_config_path}")
        else:
            logger.warning(f"Original config file not found at {original_config_path}")
        
        logger.info(f"Updated config saved to {config_path}")
        logger.info("Model pruning completed")
    else:
        logger.info("Pruning step disabled, skipping...")

def run_quantization(config: Dict[str, Any]) -> None:
    """Run model quantization if enabled."""
    if config['quantization']['enabled']:
        logger.info("Starting model quantization...")
        
        # Initialize model tagger
        tagger = ModelTagger(os.path.join(config['output']['results_dir'], 'faiss_index'))
        
        # Get the latest model directories
        original_model_dir = os.path.join(config['training']['save_dir'], config['dataset']['name'])
        pruned_model_dir = os.path.join('models/pruned', config['dataset']['name'])
        
        # Get the latest model folders
        original_dirs = [d for d in os.listdir(original_model_dir) if os.path.isdir(os.path.join(original_model_dir, d))]
        latest_original = max(original_dirs, key=lambda x: os.path.getctime(os.path.join(original_model_dir, x)))
        latest_original_dir = os.path.join(original_model_dir, latest_original)
        
        pruned_dirs = [d for d in os.listdir(pruned_model_dir) if os.path.isdir(os.path.join(pruned_model_dir, d))]
        latest_pruned = max(pruned_dirs, key=lambda x: os.path.getctime(os.path.join(pruned_model_dir, x)))
        latest_pruned_dir = os.path.join(pruned_model_dir, latest_pruned)
        
        # Create quantization output directories using config paths
        original_quant_dir = os.path.join(config['quantization']['save_dir'], 'original', config['dataset']['name'])
        pruned_quant_dir = os.path.join(config['quantization']['save_dir'], 'pruned', config['dataset']['name'])
        os.makedirs(original_quant_dir, exist_ok=True)
        os.makedirs(pruned_quant_dir, exist_ok=True)
        
        # Quantize original model
        original_quant_info = quantize_model(
            model_path=latest_original_dir,  # Pass the specific model directory
            output_dir=original_quant_dir,
            dataset_name=config['dataset']['name']
        )
        
        # Extract metrics for FAISS tagging of original quantized model
        original_quant_metrics = {
            'accuracy': original_quant_info.get('accuracy', 0.0),
            'inference_time': original_quant_info.get('inference_time', 0.0),
            'model_size': original_quant_info.get('model_size', 0),
            'num_layers': 2,  # Fixed for our architecture
            'hidden_neurons': original_quant_info.get('hidden_neurons', 0),
            'total_parameters': original_quant_info.get('total_parameters', 0),
            'quantization_bits': config['quantization']['bits'],
            'dataset_name': config['dataset']['name']
        }
        
        # Tag quantized original model
        original_quant_path = os.path.join(original_quant_dir, f"quantized_{os.path.basename(latest_original_dir)}.pth")
        original_quant_id = tagger.tag_model(original_quant_path, original_quant_metrics)
        logger.info(f"Tagged quantized original model with ID: {original_quant_id}")
        
        # Quantize pruned model
        pruned_quant_info = quantize_model(
            model_path=latest_pruned_dir,  # Pass the specific model directory
            output_dir=pruned_quant_dir,
            dataset_name=config['dataset']['name']
        )
        
        # Extract metrics for FAISS tagging of pruned quantized model
        pruned_quant_metrics = {
            'accuracy': pruned_quant_info.get('accuracy', 0.0),
            'inference_time': pruned_quant_info.get('inference_time', 0.0),
            'model_size': pruned_quant_info.get('model_size', 0),
            'num_layers': 2,  # Fixed for our architecture
            'hidden_neurons': pruned_quant_info.get('hidden_neurons', 0),
            'total_parameters': pruned_quant_info.get('total_parameters', 0),
            'quantization_bits': config['quantization']['bits'],
            'dataset_name': config['dataset']['name']
        }
        
        # Tag quantized pruned model
        pruned_quant_path = os.path.join(pruned_quant_dir, f"quantized_{os.path.basename(latest_pruned_dir)}.pth")
        pruned_quant_id = tagger.tag_model(pruned_quant_path, pruned_quant_metrics)
        logger.info(f"Tagged quantized pruned model with ID: {pruned_quant_id}")
        
        # Save quantization results
        results = {
            'original': original_quant_info,
            'pruned': pruned_quant_info
        }
        
        results_path = os.path.join(config['output']['results_dir'], 'quantization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Quantization results saved to {results_path}")
        logger.info("Model quantization completed")
    else:
        logger.info("Quantization step disabled, skipping...")

def run_inference_steps(config: Dict[str, Any]) -> None:
    """Run both standard and quantized inference if enabled."""
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
        
        # Construct paths for original and pruned models
        original_model_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_model.pth")
        pruned_model_path = os.path.join(latest_pruned_dir, f"{config['dataset']['name']}_model.pth")
        original_model_info_path = os.path.join(latest_original_dir, f"{config['dataset']['name']}_model_info.json")
        pruned_model_info_path = os.path.join(latest_pruned_dir, f"{config['dataset']['name']}_model_info.json")
        
        # Verify paths exist
        if not os.path.exists(original_model_path):
            raise FileNotFoundError(f"Original model not found at {original_model_path}")
        if not os.path.exists(pruned_model_path):
            raise FileNotFoundError(f"Pruned model not found at {pruned_model_path}")
        if not os.path.exists(original_model_info_path):
            raise FileNotFoundError(f"Original model info not found at {original_model_info_path}")
        if not os.path.exists(pruned_model_info_path):
            raise FileNotFoundError(f"Pruned model info not found at {pruned_model_info_path}")
        
        # Run standard inference
        logger.info("Running standard inference...")
        compare_models(
            original_model_path=original_model_path,
            pruned_model_path=pruned_model_path,
            dataset_path=config['dataset']['path'],
            output_dir=config['inference']['output_dir'],
            dataset_name=config['dataset']['name'],
            config=config,
            original_model_info_path=original_model_info_path,
            pruned_model_info_path=pruned_model_info_path
        )
        
        # Run quantized inference if quantization is enabled
        if config['quantization']['enabled']:
            logger.info("Running quantized inference...")
            # Get paths to quantized models
            original_quant_dir = os.path.join(config['quantization']['save_dir'], 'original', config['dataset']['name'])
            pruned_quant_dir = os.path.join(config['quantization']['save_dir'], 'pruned', config['dataset']['name'])
            
            # Get the latest quantized model files
            original_quant_files = [f for f in os.listdir(original_quant_dir) if f.endswith('.pth')]
            pruned_quant_files = [f for f in os.listdir(pruned_quant_dir) if f.endswith('.pth')]
            
            if not original_quant_files or not pruned_quant_files:
                logger.warning("Quantized model files not found, skipping quantized inference")
                return
                
            latest_original_quant = max(original_quant_files, key=lambda x: os.path.getctime(os.path.join(original_quant_dir, x)))
            latest_pruned_quant = max(pruned_quant_files, key=lambda x: os.path.getctime(os.path.join(pruned_quant_dir, x)))
            
            original_quant_path = os.path.join(original_quant_dir, latest_original_quant)
            pruned_quant_path = os.path.join(pruned_quant_dir, latest_pruned_quant)
            
            # Get corresponding info files
            original_info_path = original_quant_path.replace('.pth', '_info.json')
            pruned_info_path = pruned_quant_path.replace('.pth', '_info.json')
            
            # Verify quantized model paths exist
            if not os.path.exists(original_quant_path):
                raise FileNotFoundError(f"Quantized original model not found at {original_quant_path}")
            if not os.path.exists(pruned_quant_path):
                raise FileNotFoundError(f"Quantized pruned model not found at {pruned_quant_path}")
            if not os.path.exists(original_info_path):
                raise FileNotFoundError(f"Quantized original model info not found at {original_info_path}")
            if not os.path.exists(pruned_info_path):
                raise FileNotFoundError(f"Quantized pruned model info not found at {pruned_info_path}")
            
            compare_quantized_models(
                data_path=config['dataset']['path'],
                original_quantized_path=original_quant_path,
                original_info_path=original_info_path,
                pruned_quantized_path=pruned_quant_path,
                pruned_info_path=pruned_info_path,
                output_dir=config['inference']['output_dir'],
                dataset_name=config['dataset']['name']
            )
        
        logger.info("Inference steps completed")
    else:
        logger.info("Inference step disabled, skipping...")

def run_deployment(config: Dict[str, Any]) -> None:
    """Run model deployment steps."""
    if not config.get('deployment', {}).get('enabled', False):
        logger.info("Deployment disabled, skipping...")
        return
    
    # Import deployment modules
    from src.deployment.export_arduino_model import export_model_to_c, export_test_data
    from src.deployment.generate_arduino_sketch import generate_arduino_sketch
    
    dataset_name = config['dataset']['name']
    logger.info(f"Starting model deployment export...")
    
    # Get deployment configuration
    deployment_config = config['deployment']
    export_dir = deployment_config.get('export_dir', 'models/arduino')
    test_samples = deployment_config.get('test_samples', 5)
    
    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)
    dataset_export_dir = os.path.join(export_dir, dataset_name)
    os.makedirs(dataset_export_dir, exist_ok=True)
    
    # Export test data
    logger.info(f"Exporting test data for {dataset_name}...")
    test_data_path = export_test_data(
        data_path=config['dataset']['path'],
        output_dir=export_dir,
        dataset_name=dataset_name,
        num_samples=test_samples
    )
    
    # Get the latest quantized model files
    original_quant_dir = os.path.join(config['quantization']['save_dir'], 'original', dataset_name)
    pruned_quant_dir = os.path.join(config['quantization']['save_dir'], 'pruned', dataset_name)
    
    # Get the latest quantized model files
    original_quant_files = [f for f in os.listdir(original_quant_dir) if f.endswith('.pth')]
    pruned_quant_files = [f for f in os.listdir(pruned_quant_dir) if f.endswith('.pth')]
    
    if not original_quant_files or not pruned_quant_files:
        logger.error("Quantized model files not found")
        return
        
    latest_original_quant = max(original_quant_files, key=lambda x: os.path.getctime(os.path.join(original_quant_dir, x)))
    latest_pruned_quant = max(pruned_quant_files, key=lambda x: os.path.getctime(os.path.join(pruned_quant_dir, x)))
    
    original_quant_path = os.path.join(original_quant_dir, latest_original_quant)
    pruned_quant_path = os.path.join(pruned_quant_dir, latest_pruned_quant)
    
    # Export original model
    logger.info(f"Exporting original model for {dataset_name}...")
    export_result = export_model_to_c(
        model_path=original_quant_path,
        output_dir=export_dir,
        model_type='original',
        dataset_name=dataset_name,
        bit_width=config['quantization']['bits']
    )
    if not export_result:
        logger.error("Failed to export original model")
        return
    
    # Export pruned model if enabled
    if config.get('pruning', {}).get('enabled', False):
        logger.info(f"Exporting pruned model for {dataset_name}...")
        export_result = export_model_to_c(
            model_path=pruned_quant_path,
            output_dir=export_dir,
            model_type='pruned',
            dataset_name=dataset_name,
            bit_width=config['quantization']['bits']
        )
        if not export_result:
            logger.error("Failed to export pruned model")
            return
    
    logger.info("Model deployment completed")

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
        'results/quantized',
        'models/original',
        'models/pruned',
        'models/arduino'
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run neural network optimization pipeline')
    parser.add_argument('--config', type=str, default='config/config_breast_cancer.yaml',
                      help='Path to the configuration file')
    parser.add_argument('--clear-tags', action='store_true',
                      help='Clear all FAISS tags before running the pipeline')
    parser.add_argument('--delete-results', action='store_true',
                      help='Delete all results before running the pipeline')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Delete all results if requested
        if args.delete_results:
            delete_all_results(config['output']['results_dir'])
            return
        
        # Clear FAISS tags if requested
        if args.clear_tags:
            clear_faiss_tags(config['output']['results_dir'])
        
        # Create output directory
        os.makedirs(config['output']['results_dir'], exist_ok=True)
        
        # Run pipeline steps
        run_training(config)
        structural_prune(config)
        run_quantization(config)
        run_inference_steps(config)
        run_deployment(config)
        
        logger.info("Complete pipeline execution finished successfully")
        logger.info(f"Results saved to: {config['output']['results_dir']}")
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 