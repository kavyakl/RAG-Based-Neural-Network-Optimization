import torch
import numpy as np
import os
from datetime import datetime
import logging
import glob
import json
from typing import Tuple, Dict, Any

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel

logger = logging.getLogger(__name__)

def save_model_weights(model, save_dir, dataset_name, model_type, hidden_neurons):
    """
    Save model weights and biases to npz files.
    
    Args:
        model (nn.Module): PyTorch model
        save_dir (str): Directory to save weights
        dataset_name (str): Name of the dataset
        model_type (str): Type of model (original, pruned, etc.)
        hidden_neurons (int): Number of hidden neurons
        
    Returns:
        str: Path to the saved model directory
    """
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path
    model_dir = os.path.join(save_dir, f"{dataset_name}_{model_type}_{hidden_neurons}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save weights and biases to npz files
    layer_count = 1
    for name, layer in model.named_children():
        if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            weights = layer.weight.data.cpu().numpy()
            biases = layer.bias.data.cpu().numpy()
            
            weights_file = os.path.join(model_dir, f"fc{layer_count}_weight.npz")
            biases_file = os.path.join(model_dir, f"fc{layer_count}_bias.npz")
            
            np.savez(weights_file, weights=weights)
            np.savez(biases_file, biases=biases)
            
            layer_count += 1
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    
    # Save model architecture info as JSON
    model_info = {
        "model_type": model_type,
        "dataset": dataset_name,
        "hidden_neurons": hidden_neurons,
        "activation": getattr(model, 'activation_name', 'relu'),
        "timestamp": timestamp,
        "total_parameters": total_params,
        "input_features": model.hidden.in_features,
        "output_classes": model.output.out_features
    }
    
    with open(os.path.join(model_dir, f"{dataset_name}_model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    
    return model_dir

def load_model_weights(model: torch.nn.Module, model_path: str) -> None:
    """
    Load model weights from either a .pth file or a directory containing model weights.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        model_path (str): Path to either a .pth file or a directory containing model weights
    """
    # If model_path is a directory, try loading from .pth files first
    if os.path.isdir(model_path):
        # Get the model name from the directory path
        model_name = os.path.basename(model_path).split('_')[0]  # e.g., 'iris' from 'iris_pruned_5_20250402_175646'
        
        # Try loading from .pth file first
        pth_files = glob.glob(os.path.join(model_path, f"{model_name}*model.pth"))
        if pth_files:
            try:
                state_dict = torch.load(pth_files[0])
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded weights from {pth_files[0]}")
                return
            except Exception as e:
                logger.warning(f"Failed to load weights from {pth_files[0]}: {str(e)}")
        
        # If no .pth file found or loading failed, try loading from .npz files
        layer_count = 1
        while True:
            weight_file = os.path.join(model_path, f"fc{layer_count}_weight.npz")
            bias_file = os.path.join(model_path, f"fc{layer_count}_bias.npz")
            
            if not os.path.exists(weight_file) or not os.path.exists(bias_file):
                raise FileNotFoundError(f"Weight or bias file not found for layer {layer_count}")
            
            weights = np.load(weight_file)
            bias = np.load(bias_file)
            
            # Convert to torch tensors and load into model
            if layer_count == 1:
                model.hidden.weight.data = torch.from_numpy(weights['weights']).float()
                model.hidden.bias.data = torch.from_numpy(bias['biases']).float()
            elif layer_count == 2:
                model.output.weight.data = torch.from_numpy(weights['weights']).float()
                model.output.bias.data = torch.from_numpy(bias['biases']).float()
                break
            
            layer_count += 1
    else:
        # If model_path is a file, try loading it directly
        try:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {model_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load weights from {model_path}: {str(e)}")

def save_pytorch_model(model, save_path):
    """
    Save the complete PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model
        save_path (str): Path to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    
    return save_path

def load_pytorch_model(model_class, model_path, **kwargs):
    """
    Load a PyTorch model.
    
    Args:
        model_class (class): Class of the model to load
        model_path (str): Path to the saved model
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Loaded PyTorch model
    """
    # Initialize model
    model = model_class(**kwargs)
    
    # Load model state
    model.load_state_dict(torch.load(model_path))
    
    return model

def load_model_and_info(model_dir: str, dataset_name: str) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a model and its information from a directory.
    
    Args:
        model_dir: Directory containing model and info
        dataset_name: Name of the dataset
        
    Returns:
        Tuple[torch.nn.Module, Dict]: Loaded model and model info
    """
    # Try loading model info from different possible filenames
    info_paths = [
        os.path.join(model_dir, "model_info.json"),  # Simple model info
        os.path.join(model_dir, f"{dataset_name}_model_info.json"),  # Dataset-specific model info
        os.path.join(model_dir, f"{dataset_name}_pruned_model_info.json")  # Pruned model info
    ]
    
    model_info = None
    for info_path in info_paths:
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            break
    
    if model_info is None:
        raise FileNotFoundError(f"Model info file not found in {model_dir}")
    
    # If input_features not in model_info, try to get it from config
    if 'input_features' not in model_info:
        config_path = os.path.join('config', f'config_{dataset_name}.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                model_info['input_features'] = config['dataset']['input_features']
        else:
            raise KeyError(f"input_features not found in model info and config file not found at {config_path}")
    
    # If output_classes not in model_info, try to get it from config
    if 'output_classes' not in model_info:
        config_path = os.path.join('config', f'config_{dataset_name}.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                model_info['output_classes'] = config['dataset']['num_classes']
        else:
            # Default values for known datasets
            defaults = {
                'iris': 3,
                'breast_cancer': 2,
                'wine': 3,
                'digits_binary': 2,
                'mnist_binary': 2,
                'mnist_digit0': 2,
                'yeast': 10,
                'heart': 2,
                'diabetes': 2,
                'fetalhealth': 3,
                'liverdiagnostic': 2,
                'customers': 2,
                'balance': 3,
                'raisin': 2
            }
            model_info['output_classes'] = defaults.get(dataset_name, 2)
    
    # Find the model weights file
    weights_paths = [
        os.path.join(model_dir, f"{dataset_name}_model.pth"),  # Original model weights
        os.path.join(model_dir, f"{dataset_name}_pruned_model.pth")  # Pruned model weights
    ]
    
    weights_path = None
    for path in weights_paths:
        if os.path.exists(path):
            weights_path = path
            break
    
    if weights_path is None:
        raise FileNotFoundError(f"Model weights file not found in {model_dir}")
    
    # Load the state dict to get the actual input features
    state_dict = torch.load(weights_path)
    actual_input_features = state_dict['hidden.weight'].shape[1]
    
    # Update model_info with the actual input features
    model_info['input_features'] = actual_input_features
    
    # Create model based on updated info
    if model_info['output_classes'] > 2:
        model = CategoricalModel(
            model_info['input_features'],
            model_info['hidden_neurons'],
            model_info['output_classes'],
            activation=model_info.get('activation', 'relu')
        )
    else:
        model = BinaryModel(
            model_info['input_features'],
            model_info['hidden_neurons'],
            activation=model_info.get('activation', 'relu')
        )
    
    # Load the weights
    model.load_state_dict(state_dict)
    
    return model, model_info

def save_model_and_info(model: torch.nn.Module, model_dir: str, dataset_name: str, model_type: str = 'original') -> None:
    """
    Save a model and its information to a directory.
    
    Args:
        model: Model to save
        model_dir: Directory to save model and info
        dataset_name: Name of the dataset
        model_type: Type of model (original, pruned, quantized)
    """
    # Extract model info
    input_features = model.hidden.in_features
    hidden_neurons = model.hidden.out_features
    output_classes = model.output.out_features
    
    # Create model info dictionary
    model_info = {
        'model_type': model_type,
        'dataset': dataset_name,
        'input_features': input_features,
        'hidden_neurons': hidden_neurons,
        'output_classes': output_classes,
        'activation': getattr(model, 'activation_name', 'relu')
    }
    
    # If this is a pruned model, check if it has pruning attributes
    if model_type == 'pruned':
        # Check if model has pruning attributes
        if hasattr(model, 'pruning_info'):
            # Use the pruning info directly from the model
            pruning_info = model.pruning_info
            # Merge pruning info with basic model info
            model_info.update(pruning_info)
        else:
            # Try to load from file if available
            info_path = os.path.join(model_dir, f'{dataset_name}_model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    pruning_info = json.load(f)
                    # Merge pruning info with basic model info
                    model_info.update(pruning_info)
    
    # Save model info
    info_path = os.path.join(model_dir, f'{dataset_name}_model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Save model weights
    weights_path = os.path.join(model_dir, f'{dataset_name}_model.pth')
    torch.save(model.state_dict(), weights_path)
    
    logger.info(f"Saved model info to {info_path}")
    logger.info(f"Saved model weights to {weights_path}")
    logger.info(f"Model info: {model_info}") 