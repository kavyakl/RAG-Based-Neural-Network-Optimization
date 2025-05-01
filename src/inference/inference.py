import os
import json
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import logging
import glob
import time
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from datetime import datetime

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel
from ..utils.data_handling import load_and_preprocess_data
from ..utils.model_io import load_model_weights

# Configure logging
logger = logging.getLogger(__name__)

def load_model_and_info(model_dir: str) -> Tuple[torch.nn.Module, Dict]:
    """
    Load model and its info from directory.
    
    Args:
        model_dir (str): Directory containing model files
        
    Returns:
        Tuple[torch.nn.Module, Dict]: Model and model info
    """
    # Load model info
    model_info_path = os.path.join(model_dir, 'model_info.json')
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info not found at {model_info_path}")
    
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Get number of classes from model info (handle both output_classes and num_classes)
    num_classes = model_info.get('output_classes') or model_info.get('num_classes', 2)
    
    # Get number of features (handle both num_features and input_features)
    num_features = model_info.get('input_features') or model_info.get('num_features')
    hidden_neurons = model_info.get('hidden_neurons')
    activation = model_info.get('activation', 'relu')  # Default to ReLU if not specified
    
    if num_features is None or hidden_neurons is None:
        raise ValueError(f"Model info missing required fields. Found: {model_info.keys()}")
    
    # Create model instance with correct output shape and activation
    if num_classes > 2:
        model = CategoricalModel(num_features, hidden_neurons, num_classes, activation=activation)
        logger.info(f"Created CategoricalModel with {num_classes} output classes and {activation} activation")
    else:
        model = BinaryModel(num_features, hidden_neurons, activation=activation)
        logger.info(f"Created BinaryModel with {activation} activation")
    
    # Load weights
    load_model_weights(model, model_dir)
    
    return model, model_info

def run_inference(model, data, num_runs=100):
    """
    Run inference on the given model and data.
    
    Args:
        model: PyTorch model
        data: Tuple of (X_train, X_test, y_train, y_test) numpy arrays or tensors
        num_runs: Number of inference runs to measure average time
        
    Returns:
        dict: Inference results including predictions, probabilities, metrics, and timing
    """
    # Unpack the data tuple
    X_train, X_test, y_train, y_test = data
    
    # Use test data for inference
    # Check if X_test is already a tensor
    if isinstance(X_test, torch.Tensor):
        X = X_test.float()
    else:
        X = torch.from_numpy(X_test).float()
    
    # Ensure X has the correct shape (batch_size, input_features)
    if X.dim() == 1:
        X = X.unsqueeze(0)  # Add batch dimension
    elif X.dim() == 2 and X.shape[1] != model.hidden.in_features:
        X = X.t()  # Transpose if needed to match expected input shape
    
    # Check if y_test is already a tensor
    if isinstance(y_test, torch.Tensor):
        y_true = y_test
    else:
        y_true = torch.from_numpy(y_test)
    
    # Ensure y_true has the correct shape for binary classification
    if model.output.out_features == 1:  # Binary classification
        y_true = y_true.view(-1, 1).float()
    else:  # Multi-class classification
        y_true = y_true.long()
    
    # Calculate model size
    model_size = get_model_size(model)
    logger.info(f"Model size: {model_size / 1024:.2f} KB")
    
    # Measure inference time
    model.eval()
    inference_times = []
    
    # Warm-up run
    with torch.no_grad():
        _ = model(X)
    
    # Measure inference time over multiple runs
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs, _ = model(X)
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    logger.info(f"Average inference time: {avg_inference_time:.4f} ms")
    
    # Run inference for metrics
    with torch.no_grad():
        # Get model outputs (outputs, hidden_layer_output)
        outputs, _ = model(X)
        
        # Log shapes for debugging
        logger.info(f"Model output shape: {outputs.shape}")
        logger.info(f"y_true shape: {y_true.shape}")
        
        # Calculate metrics
        if model.output.out_features == 1:  # Binary classification
            predictions = (outputs > 0.5).float()
            probabilities = torch.sigmoid(outputs)
        else:  # Multi-class classification
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Calculate accuracy
        if model.output.out_features == 1:  # Binary classification
            accuracy = (predictions == y_true).float().mean().item() * 100
        else:  # Multi-class classification
            accuracy = (predictions == y_true).float().mean().item() * 100
        
        # Calculate other metrics
        if model.output.out_features == 1:  # Binary classification
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true.cpu().numpy(), 
                predictions.cpu().numpy(), 
                average='binary'
            )
        else:  # Multi-class classification
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true.cpu().numpy(), 
                predictions.cpu().numpy(), 
                average='weighted'
            )
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100
            },
            'size_bytes': model_size,
            'inference_time_ms': avg_inference_time
        }

def compare_models(original_model, pruned_model, data, config):
    """
    Compare original and pruned models.
    
    Args:
        original_model: Original PyTorch model
        pruned_model: Pruned PyTorch model
        data: Tuple of (X_train, X_test, y_train, y_test) numpy arrays
        config: Configuration dictionary
        
    Returns:
        dict: Comparison results
    """
    # Run inference on both models
    original_results = run_inference(original_model, data)
    pruned_results = run_inference(pruned_model, data)
    
    # Calculate model sizes
    original_size = original_results['size_bytes']
    pruned_size = pruned_results['size_bytes']
    
    # Calculate compression ratio
    compression_ratio = original_size / pruned_size if pruned_size > 0 else float('inf')
    
    # Calculate speedup ratio
    speedup_ratio = original_results['inference_time_ms'] / pruned_results['inference_time_ms'] if pruned_results['inference_time_ms'] > 0 else float('inf')
    
    # Print comparison summary
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*50)
    logger.info(f"Original model size: {original_size / 1024:.2f} KB")
    logger.info(f"Pruned model size: {pruned_size / 1024:.2f} KB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Original inference time: {original_results['inference_time_ms']:.4f} ms")
    logger.info(f"Pruned inference time: {pruned_results['inference_time_ms']:.4f} ms")
    logger.info(f"Speedup ratio: {speedup_ratio:.2f}x")
    logger.info(f"Original accuracy: {original_results['metrics']['accuracy']:.2f}%")
    logger.info(f"Pruned accuracy: {pruned_results['metrics']['accuracy']:.2f}%")
    logger.info(f"Accuracy drop: {original_results['metrics']['accuracy'] - pruned_results['metrics']['accuracy']:.2f}%")
    logger.info("="*50)
    
    # Prepare comparison results
    comparison = {
        'original_model': {
            'size_bytes': original_size,
            'size_kb': original_size / 1024,
            'inference_time_ms': original_results['inference_time_ms'],
            'metrics': original_results['metrics']
        },
        'pruned_model': {
            'size_bytes': pruned_size,
            'size_kb': pruned_size / 1024,
            'inference_time_ms': pruned_results['inference_time_ms'],
            'metrics': pruned_results['metrics']
        },
        'compression_ratio': compression_ratio,
        'speedup_ratio': speedup_ratio,
        'dataset': config['dataset']['name'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    return comparison

def get_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Size of model in bytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_bytes = param_size + buffer_size
    return size_all_bytes 