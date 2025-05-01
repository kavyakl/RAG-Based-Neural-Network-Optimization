import os
import json
import yaml
import pickle
import torch
import numpy as np
import logging
import glob
import re
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel
from ..utils.model_io import load_model_weights, load_model_and_info

# Configure logging
logger = logging.getLogger(__name__)

def quantize_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Quantize model weights to 8-bit precision.
    
    Args:
        state_dict (Dict[str, torch.Tensor]): Model state dictionary
        
    Returns:
        Dict[str, torch.Tensor]: Quantized state dictionary
    """
    quantized_state_dict = {}
    
    for name, param in state_dict.items():
        # Skip non-weight parameters
        if 'weight' not in name:
            quantized_state_dict[name] = param
            continue
            
        # Get weight statistics
        min_val = param.min().item()
        max_val = param.max().item()
        mean_val = param.mean().item()
        std_val = param.std().item()
        
        logger.info(f"Layer {name} statistics before quantization:")
        logger.info(f"  Min: {min_val:.4f}")
        logger.info(f"  Max: {max_val:.4f}")
        logger.info(f"  Mean: {mean_val:.4f}")
        logger.info(f"  Std: {std_val:.4f}")
        
        # Normalize weights to [0, 1] range
        normalized = (param - min_val) / (max_val - min_val + 1e-8)
        
        # Quantize to 8 bits
        quantized = torch.round(normalized * 255)
        
        # Dequantize back to float32
        dequantized = quantized / 255.0
        
        # Scale back to original range
        dequantized = dequantized * (max_val - min_val + 1e-8) + min_val
        
        # Calculate statistics after quantization
        quant_min = dequantized.min().item()
        quant_max = dequantized.max().item()
        quant_mean = dequantized.mean().item()
        quant_std = dequantized.std().item()
        
        logger.info(f"Layer {name} statistics after quantization:")
        logger.info(f"  Min: {quant_min:.4f}")
        logger.info(f"  Max: {quant_max:.4f}")
        logger.info(f"  Mean: {quant_mean:.4f}")
        logger.info(f"  Std: {quant_std:.4f}")
        
        # Calculate average change
        avg_change = torch.mean(torch.abs(param - dequantized)).item()
        logger.info(f"  Average change: {avg_change:.4f}")
        
        quantized_state_dict[name] = dequantized
    
    return quantized_state_dict

def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types.
    Also handles tuple keys in dictionaries by converting them to strings.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert tuple keys to strings and handle values
        return {str(k) if isinstance(k, tuple) else k: convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def save_quantized_model(
    model: torch.nn.Module,
    model_info: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    model_type: str
) -> Dict[str, str]:
    """
    Save quantized model and its metadata.
    
    Args:
        model (torch.nn.Module): Quantized model
        model_info (dict): Model metadata
        output_dir (str): Directory to save the model
        dataset_name (str): Name of the dataset
        model_type (str): Type of model ('original' or 'pruned')
        
    Returns:
        dict: Paths to saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_quantized.pth")
    torch.save(model.state_dict(), model_path)
    
    # Convert NumPy types to Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # Convert tuple keys to strings and handle values
            return {str(k) if isinstance(k, tuple) else k: convert_numpy_types(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        return obj
    
    # Create a new model info dictionary with required fields
    quantized_info = {
        'num_features': model.hidden.in_features,
        'num_classes': 2 if isinstance(model, BinaryModel) else model.output.out_features,
        'hidden_neurons': model.hidden.out_features,
        'activation': model_info.get('activation', 'relu'),
        'dataset_name': dataset_name
    }
    
    # Add additional info from the original model_info
    if model_type == 'pruned':
        quantized_info.update({
            'pruned_indices': model_info.get('pruned_indices', []),
            'correlation_results': model_info.get('correlation_results', {}),
            'low_activity_neurons': model_info.get('low_activity_neurons', []),
            'correlated_neurons': model_info.get('correlated_neurons', []),
            'original_hidden_neurons': model_info.get('original_hidden_neurons', 0),
            'pruned_hidden_neurons': model_info.get('pruned_hidden_neurons', 0)
        })
    
    # Convert model info
    quantized_info = convert_numpy_types(quantized_info)
    
    # Save model info
    info_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_quantized_info.json")
    with open(info_path, 'w') as f:
        json.dump(quantized_info, f, indent=4)
    
    return {
        'model_path': model_path,
        'info_path': info_path
    }

def quantize_model(model_path: str, output_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Quantize a model's weights to 8-bit precision."""
    # Load model info to get architecture details
    model_info_path = os.path.join(model_path, f"{dataset_name}_model_info.json")
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Determine if this is a pruned model and get correct number of hidden neurons
    is_pruned = 'pruned_hidden_neurons' in model_info
    hidden_neurons = model_info['pruned_hidden_neurons'] if is_pruned else model_info['hidden_neurons']
    
    # Create model with correct architecture
    if model_info['output_classes'] > 2:
        model = CategoricalModel(
            input_size=model_info['input_features'],
            hidden_size=hidden_neurons,
            output_size=model_info['output_classes'],
            activation=model_info['activation']
        )
    else:
        model = BinaryModel(
            input_size=model_info['input_features'],
            hidden_size=hidden_neurons,
            activation=model_info['activation']
        )
    
    # Load weights
    weights_path = os.path.join(model_path, f"{dataset_name}_model.pth")
    model.load_state_dict(torch.load(weights_path))
    
    # Quantize weights
    quantized_state_dict = quantize_weights(model.state_dict())
    
    # Create new model with same architecture
    if model_info['output_classes'] > 2:
        quantized_model = CategoricalModel(
            input_size=model_info['input_features'],
            hidden_size=hidden_neurons,
            output_size=model_info['output_classes'],
            activation=model_info['activation']
        )
    else:
        quantized_model = BinaryModel(
            input_size=model_info['input_features'],
            hidden_size=hidden_neurons,
            activation=model_info['activation']
        )
    
    # Load quantized weights
    quantized_model.load_state_dict(quantized_state_dict)
    
    # Save quantized model
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    quantized_path = os.path.join(output_dir, f"quantized_{os.path.basename(model_path)}.pth")
    torch.save(quantized_state_dict, quantized_path)
    
    # Save quantized model info
    quantized_info_path = quantized_path.replace('.pth', '_info.json')
    with open(quantized_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Calculate model size
    model_size = sum(p.numel() for p in quantized_model.parameters())
    
    return {
        'accuracy': 0.0,  # Will be updated during inference
        'inference_time': 0.0,  # Will be updated during inference
        'model_size': model_size,
        'hidden_neurons': hidden_neurons,
        'total_parameters': model_size,
        'quantization_bits': 8
    } 