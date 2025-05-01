import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import json
import pickle
import yaml
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy import stats
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel
from src.utils.data_handling import load_saved_dataset, prepare_data_for_training
from src.utils.metrics import calculate_accuracy, save_model_metrics
from src.utils.model_io import load_model_weights, save_model_weights, save_pytorch_model, load_model_and_info
from src.utils.numpy_encoder import NumpyEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_neuron_sensitivity(model, X_train_tensor, hidden_layer_output, threshold=0.1, activation='relu'):
    """
    Analyze neuron sensitivity by measuring how changes in input affect neuron activation.
    
    Args:
        model: PyTorch model (can be None)
        X_train_tensor: Training data (can be None)
        hidden_layer_output: Hidden layer activations
        threshold (float): Threshold for sensitivity score
        activation (str): Activation function ('relu', 'sigmoid', or 'tanh')
        
    Returns:
        tuple: (list of insensitive neuron indices, array of sensitivity scores)
    """
    # Check if model is a tensor instead of a model object
    if isinstance(model, torch.Tensor):
        logger.warning(f"model is a tensor, not a model object. Using alternative approach.")
        # Create dummy sensitivity scores
        if isinstance(hidden_layer_output, torch.Tensor):
            num_neurons = hidden_layer_output.shape[1]
        else:
            # If hidden_layer_output is not a tensor, use a default size
            num_neurons = 10  # Default size
            logger.warning(f"hidden_layer_output is not a tensor: {type(hidden_layer_output)}")
        
        # Create dummy sensitivity scores
        sensitivity_scores = np.ones(num_neurons) / num_neurons  # Equal sensitivity
        insensitive_indices = []  # No insensitive neurons
        
        logger.info(f"Using dummy sensitivity scores for {num_neurons} neurons")
        return insensitive_indices, sensitivity_scores
    
    # Ensure output is a numpy array
    if isinstance(hidden_layer_output, torch.Tensor):
        hidden_layer_output = hidden_layer_output.detach().cpu().numpy()
    
    # Calculate sensitivity scores for each neuron
    num_neurons = hidden_layer_output.shape[1]
    sensitivity_scores = np.zeros(num_neurons)
    
    # Set activation-specific parameters
    if activation.lower() == 'sigmoid':
        neutral_point = 0.5
        extreme_threshold = 0.1
        neutral_range = (0.4, 0.6)
    elif activation.lower() == 'tanh':
        neutral_point = 0.0
        extreme_threshold = 0.1
        neutral_range = (-0.2, 0.2)
    else:  # relu
        neutral_point = 0.0
        extreme_threshold = 0.1
        neutral_range = (-0.1, 0.1)
    
    # For each neuron, calculate its sensitivity based on activation patterns
    for i in range(num_neurons):
        # Get activations for this neuron
        activations = hidden_layer_output[:, i]
        
        # Calculate sensitivity based on activation statistics
        mean_activation = np.mean(activations)
        std_activation = np.std(activations)
        max_activation = np.max(activations)
        min_activation = np.min(activations)
        
        # Calculate range of activations
        activation_range = max_activation - min_activation
        
        # Check for dead neuron conditions
        is_dead = False
        
        # Condition 1: Low standard deviation
        if std_activation < threshold:
            is_dead = True
            logger.info(f"Neuron {i} marked as dead: Low standard deviation ({std_activation:.3f} < {threshold})")
        
        # Condition 2: Consistently neutral
        elif (neutral_range[0] <= mean_activation <= neutral_range[1] and 
              std_activation < 0.1):
            is_dead = True
            logger.info(f"Neuron {i} marked as dead: Consistently neutral (mean={mean_activation:.3f}, std={std_activation:.3f})")
        
        # Condition 3: Small activation range
        elif activation_range < threshold:
            is_dead = True
            logger.info(f"Neuron {i} marked as dead: Small activation range ({activation_range:.3f} < {threshold})")
        
        # Condition 4: Consistently at extreme values
        if activation.lower() == 'tanh':
            # For tanh, check if consistently at -1 or 1
            if ((mean_activation < -0.9 or mean_activation > 0.9) and 
                std_activation < 0.05):
                is_dead = True
                logger.info(f"Neuron {i} marked as dead: Consistently at extreme values (mean={mean_activation:.3f}, std={std_activation:.3f})")
        else:
            # For sigmoid and ReLU
            if ((mean_activation < extreme_threshold or mean_activation > (1 - extreme_threshold)) and 
                std_activation < 0.05):
                is_dead = True
                logger.info(f"Neuron {i} marked as dead: Consistently at extreme values (mean={mean_activation:.3f}, std={std_activation:.3f})")
        
        if is_dead:
            sensitivity_scores[i] = 0
        else:
            # For non-dead neurons, calculate sensitivity score based on multiple factors
            if activation.lower() == 'tanh':
                # For tanh, distance from neutral point is absolute value
                distance_from_neutral = abs(mean_activation - neutral_point)
                sensitivity_scores[i] = (
                    0.4 * std_activation +  # Standard deviation contribution
                    0.3 * activation_range +  # Range contribution
                    0.3 * (1 - distance_from_neutral)  # Distance from neutral point
                )
            else:
                # For sigmoid and ReLU
                sensitivity_scores[i] = (
                    0.4 * std_activation +  # Standard deviation contribution
                    0.3 * activation_range +  # Range contribution
                    0.3 * (1 - abs(mean_activation - neutral_point))  # Distance from neutral point
                )
    
    # Normalize sensitivity scores
    sensitivity_scores = (sensitivity_scores - np.min(sensitivity_scores)) / (np.max(sensitivity_scores) - np.min(sensitivity_scores) + 1e-10)
    
    # Find insensitive neurons
    insensitive_indices = [i for i, score in enumerate(sensitivity_scores) if score < threshold]
    
    # Log sensitivity information
    logger.info(f"Found {len(insensitive_indices)} insensitive neurons (sensitivity < {threshold})")
    for idx in insensitive_indices:
        logger.info(f"Neuron {idx}: sensitivity score = {sensitivity_scores[idx]:.3f}")
    
    return insensitive_indices, sensitivity_scores

def calculate_neuron_importance(
    model: nn.Module,
    hidden_layer: nn.Linear,
    layer_idx: int,
    inputs: torch.Tensor,
    activation: str = 'relu'
) -> Dict[str, torch.Tensor]:
    """
    Calculate importance scores for neurons in a hidden layer.
    
    Args:
        model: The neural network model
        hidden_layer: The hidden layer to analyze
        layer_idx: Index of the hidden layer
        inputs: Input tensor for validation
        activation: Activation function used in the model ('relu', 'sigmoid', or 'tanh')
        
    Returns:
        Dictionary containing importance scores and masks
    """
    # Check if model is a tensor instead of a model object
    if isinstance(model, torch.Tensor):
        logger.warning(f"model is a tensor, not a model object. Using alternative approach.")
        # Create dummy importance scores
        if isinstance(hidden_layer, nn.Linear):
            num_neurons = hidden_layer.out_features
        else:
            # If hidden_layer is also not a proper layer, use a default size
            num_neurons = 10  # Default size
            logger.warning(f"hidden_layer is not a proper PyTorch Linear layer: {type(hidden_layer)}")
        
        importance_scores = torch.ones(num_neurons) / num_neurons  # Equal importance
        correlation_mask = torch.zeros(num_neurons, dtype=torch.bool)
        sensitivity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        activity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        
        return {
            'importance_scores': importance_scores,
            'correlation_mask': correlation_mask,
            'sensitivity_mask': sensitivity_mask,
            'activity_mask': activity_mask
        }
    
    # Check if hidden_layer is a proper PyTorch layer
    if not isinstance(hidden_layer, nn.Linear):
        logger.warning(f"hidden_layer is not a proper PyTorch Linear layer: {type(hidden_layer)}")
        logger.warning("Using alternative approach for importance calculation")
        
        # Create dummy importance scores
        num_neurons = model.hidden.out_features
        importance_scores = torch.ones(num_neurons) / num_neurons  # Equal importance
        correlation_mask = torch.zeros(num_neurons, dtype=torch.bool)
        sensitivity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        activity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        
        return {
            'importance_scores': importance_scores,
            'correlation_mask': correlation_mask,
            'sensitivity_mask': sensitivity_mask,
            'activity_mask': activity_mask
        }
    
    # Calculate correlation scores
    with torch.no_grad():
        activations = hidden_layer(inputs)
        # Reshape activations if needed
        if len(activations.shape) > 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        # Apply activation function if not already applied
        if activation.lower() == 'relu':
            activations = F.relu(activations)
        elif activation.lower() == 'sigmoid':
            activations = torch.sigmoid(activations)
        elif activation.lower() == 'tanh':
            activations = torch.tanh(activations)
        
        # Handle case where correlation_matrix is empty
        if activations.shape[0] < 2:
            logger.warning("Not enough samples to calculate correlation matrix")
            correlation_scores = torch.zeros(hidden_layer.out_features)
        else:
            correlation_matrix = torch.corrcoef(activations.T)
            # Handle case where correlation_matrix is empty
            if correlation_matrix.numel() == 0:
                correlation_scores = torch.zeros(hidden_layer.out_features)
            else:
                correlation_scores = torch.max(correlation_matrix - torch.eye(correlation_matrix.shape[0]), dim=1)[0]
    
    # Calculate sensitivity scores
    sensitivity_scores = torch.zeros(hidden_layer.out_features)
    for i in range(hidden_layer.out_features):
        # Create a copy of the model based on its type
        if isinstance(model, CategoricalModel):
            model_copy = CategoricalModel(
                input_size=model.hidden.in_features,
                hidden_size=model.hidden.out_features,
                output_size=model.output.out_features,
                activation=model.activation_name
            )
        else:  # BinaryModel
            model_copy = BinaryModel(
                input_size=model.hidden.in_features,
                hidden_size=model.hidden.out_features,
                activation=model.activation_name
            )
        
        model_copy.load_state_dict(model.state_dict())
        
        # Zero out the neuron
        with torch.no_grad():
            model_copy.hidden.weight[i] = 0
            model_copy.hidden.bias[i] = 0
        
        # Calculate output difference
        with torch.no_grad():
            original_output = model(inputs)[0]  # Get just the outputs, not hidden states
            pruned_output = model_copy(inputs)[0]  # Get just the outputs, not hidden states
            sensitivity_scores[i] = torch.mean(torch.abs(original_output - pruned_output))
    
    # Calculate activity scores based on activation function
    with torch.no_grad():
        if activation.lower() == 'sigmoid':
            # For sigmoid, look for neurons that are consistently close to 0 or 1
            activity_scores = torch.mean(torch.abs(activations - 0.5), dim=0)
        elif activation.lower() == 'tanh':
            # For tanh, look for neurons that are consistently close to -1 or 1
            activity_scores = torch.mean(torch.abs(activations), dim=0)
        else:  # ReLU
            # For ReLU, look for neurons with low activation
            activity_scores = torch.mean(activations, dim=0)
    
    # Normalize scores
    correlation_scores = (correlation_scores - correlation_scores.min()) / (correlation_scores.max() - correlation_scores.min() + 1e-8)
    sensitivity_scores = (sensitivity_scores - sensitivity_scores.min()) / (sensitivity_scores.max() - sensitivity_scores.min() + 1e-8)
    activity_scores = (activity_scores - activity_scores.min()) / (activity_scores.max() - activity_scores.min() + 1e-8)
    
    # Adjust thresholds based on activation function
    if activation.lower() == 'sigmoid':
        # Sigmoid neurons tend to have more extreme values, so we need more conservative thresholds
        correlation_threshold = 0.85  # Higher correlation threshold for sigmoid
        sensitivity_threshold = 0.15  # Higher sensitivity threshold for sigmoid
        activity_threshold = 0.4     # Higher activity threshold for sigmoid
    elif activation.lower() == 'tanh':
        # Tanh neurons have values between -1 and 1, so we need different thresholds
        correlation_threshold = 0.8   # Medium correlation threshold for tanh
        sensitivity_threshold = 0.12  # Medium sensitivity threshold for tanh
        activity_threshold = 0.7     # Medium activity threshold for tanh
    else:  # ReLU
        # ReLU uses the default thresholds
        correlation_threshold = 0.7   # Lower correlation threshold for ReLU
        sensitivity_threshold = 0.1   # Lower sensitivity threshold for ReLU
        activity_threshold = 0.1     # Lower activity threshold for ReLU
    
    # Calculate final importance scores with activation-specific weighting
    if activation.lower() == 'sigmoid':
        # For sigmoid, weight correlation more heavily
        importance_scores = (0.5 * correlation_scores + 0.3 * sensitivity_scores + 0.2 * activity_scores)
    elif activation.lower() == 'tanh':
        # For tanh, weight sensitivity more heavily
        importance_scores = (0.3 * correlation_scores + 0.5 * sensitivity_scores + 0.2 * activity_scores)
    else:  # ReLU
        # For ReLU, weight activity more heavily
        importance_scores = (0.3 * correlation_scores + 0.3 * sensitivity_scores + 0.4 * activity_scores)
    
    # Create masks for different pruning criteria
    correlation_mask = correlation_scores > correlation_threshold
    sensitivity_mask = sensitivity_scores < sensitivity_threshold
    activity_mask = activity_scores < activity_threshold
    
    # Log information about the analysis
    logger.info(f"Neuron importance analysis for {activation} activation:")
    logger.info(f"  - Correlation threshold: {correlation_threshold}")
    logger.info(f"  - Sensitivity threshold: {sensitivity_threshold}")
    logger.info(f"  - Activity threshold: {activity_threshold}")
    logger.info(f"  - Found {correlation_mask.sum().item()} highly correlated neurons")
    logger.info(f"  - Found {sensitivity_mask.sum().item()} insensitive neurons")
    logger.info(f"  - Found {activity_mask.sum().item()} low activity neurons")
    
    return {
        'importance_scores': importance_scores,
        'correlation_mask': correlation_mask,
        'sensitivity_mask': sensitivity_mask,
        'activity_mask': activity_mask,
        'correlation_scores': correlation_scores,
        'sensitivity_scores': sensitivity_scores,
        'activity_scores': activity_scores,
        'activation': activation
    }

def analyze_hidden_layer(
    model: nn.Module,
    hidden_layer: nn.Linear,
    layer_idx: int,
    validation_data: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, float],
    enhanced_pruning: bool = False,
    activation: str = 'relu'
) -> Dict[str, torch.Tensor]:
    """
    Analyze a hidden layer to identify neurons that can be pruned.
    
    Args:
        model: The neural network model
        hidden_layer: The hidden layer to analyze
        layer_idx: Index of the hidden layer
        validation_data: Tuple of (inputs, targets) for validation, or just inputs tensor, or a float
        enhanced_pruning: Whether to use enhanced pruning
        activation: Activation function used in the model ('relu', 'sigmoid', or 'tanh')
        
    Returns:
        Dictionary containing analysis results
    """
    # Check if model is a tensor instead of a model object
    if isinstance(model, torch.Tensor):
        logger.warning(f"model is a tensor, not a model object. Using alternative approach.")
        # Create dummy analysis results
        if isinstance(hidden_layer, nn.Linear):
            num_neurons = hidden_layer.out_features
        else:
            # If hidden_layer is also not a proper layer, use a default size
            num_neurons = 10  # Default size
            logger.warning(f"hidden_layer is not a proper PyTorch Linear layer: {type(hidden_layer)}")
        
        # Create dummy correlation mask
        correlation_mask = torch.zeros(num_neurons, dtype=torch.bool)
        
        # Create dummy importance scores
        importance_scores = torch.ones(num_neurons) / num_neurons  # Equal importance
        sensitivity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        activity_mask = torch.zeros(num_neurons, dtype=torch.bool)
        
        return {
            'correlation_mask': correlation_mask,
            'importance_scores': importance_scores,
            'sensitivity_mask': sensitivity_mask,
            'activity_mask': activity_mask
        }
    
    # Handle different types of validation_data
    if isinstance(validation_data, tuple):
        inputs = validation_data[0]
    elif isinstance(validation_data, torch.Tensor):
        inputs = validation_data
    else:
        # If validation_data is a float or any other type, use a dummy tensor
        logger.warning(f"validation_data is of type {type(validation_data)}, using dummy tensor")
        # Use a reasonable default batch size or get from config
        default_batch_size = 32  # Standard mini-batch size
        
        # Check if hidden_layer is a PyTorch layer
        if isinstance(hidden_layer, nn.Linear):
            inputs = torch.randn(default_batch_size, hidden_layer.in_features)
            logger.info(f"Created dummy input tensor with shape: {inputs.shape}")
        else:
            # If hidden_layer is not a PyTorch layer, use a default size
            logger.warning(f"hidden_layer is of type {type(hidden_layer)}, using default input size")
            inputs = torch.randn(default_batch_size, hidden_layer.in_features if hasattr(hidden_layer, 'in_features') else 10)
            logger.info(f"Created dummy input tensor with default shape: {inputs.shape}")
    
    if enhanced_pruning:
        return calculate_neuron_importance(model, hidden_layer, layer_idx, inputs, activation=activation)
    else:
        # Original pruning logic with more aggressive correlation detection
        with torch.no_grad():
            activations = hidden_layer(inputs)
            # Reshape activations if needed
            if len(activations.shape) > 2:
                activations = activations.reshape(activations.shape[0], -1)
            
            # Apply activation function if not already applied
            if activation.lower() == 'relu':
                activations = F.relu(activations)
            elif activation.lower() == 'sigmoid':
                activations = torch.sigmoid(activations)
            elif activation.lower() == 'tanh':
                activations = torch.tanh(activations)
            
            # Calculate correlation matrix
            correlation_matrix = torch.corrcoef(activations.T)
            
            # Set correlation threshold based on activation function
            if activation.lower() == 'sigmoid':
                correlation_threshold = 0.85  # Higher threshold for sigmoid
            elif activation.lower() == 'tanh':
                correlation_threshold = 0.8   # Medium threshold for tanh
            else:  # ReLU
                correlation_threshold = 0.7   # Lower threshold for ReLU
            
            # Find highly correlated neurons
            num_neurons = activations.shape[1]
            correlation_mask = torch.zeros(num_neurons, dtype=torch.bool)
            correlated_pairs = []
            
            # Process top 30% of correlated pairs
            num_pairs_to_process = int(0.3 * (num_neurons * (num_neurons - 1)) / 2)
            correlation_values = []
            neuron_pairs = []
            
            # Collect all correlation values and pairs
            for i in range(num_neurons):
                for j in range(i + 1, num_neurons):
                    correlation_values.append(correlation_matrix[i, j].item())
                    neuron_pairs.append((i, j))
            
            # Sort by correlation strength
            sorted_indices = np.argsort(correlation_values)[::-1]
            top_pairs = [neuron_pairs[i] for i in sorted_indices[:num_pairs_to_process]]
            
            # Mark neurons in top correlated pairs
            for i, j in top_pairs:
                if correlation_matrix[i, j] > correlation_threshold:
                    correlation_mask[i] = True
                    correlation_mask[j] = True
                    correlated_pairs.append((i, j))
            
            logger.info(f"Found {len(correlated_pairs)} highly correlated neuron pairs")
            for i, j in correlated_pairs:
                logger.info(f"Neurons {i} and {j} are correlated with coefficient {correlation_matrix[i, j]:.3f}")
            
            # Calculate sensitivity scores
            sensitivity_scores = torch.zeros(num_neurons)
            for i in range(num_neurons):
                # Create a copy of the model
                if isinstance(model, CategoricalModel):
                    model_copy = CategoricalModel(
                        input_size=model.hidden.in_features,
                        hidden_size=model.hidden.out_features,
                        output_size=model.output.out_features,
                        activation=model.activation_name
                    )
                else:  # BinaryModel
                    model_copy = BinaryModel(
                        input_size=model.hidden.in_features,
                        hidden_size=model.hidden.out_features,
                        activation=model.activation_name
                    )
                
                model_copy.load_state_dict(model.state_dict())
                
                # Zero out the neuron
                with torch.no_grad():
                    model_copy.hidden.weight[i] = 0
                    model_copy.hidden.bias[i] = 0
                
                # Calculate output difference
                with torch.no_grad():
                    original_output = model(inputs)[0]
                    pruned_output = model_copy(inputs)[0]
                    sensitivity_scores[i] = torch.mean(torch.abs(original_output - pruned_output))
            
            # Calculate activity scores
            activity_scores = torch.mean(activations, dim=0)
            
            # Normalize scores
            sensitivity_scores = (sensitivity_scores - sensitivity_scores.min()) / (sensitivity_scores.max() - sensitivity_scores.min() + 1e-8)
            activity_scores = (activity_scores - activity_scores.min()) / (activity_scores.max() - activity_scores.min() + 1e-8)
            
            # Create masks for different pruning criteria
            sensitivity_threshold = 0.1 if activation.lower() == 'relu' else 0.15
            activity_threshold = 0.1 if activation.lower() == 'relu' else 0.4
            
            sensitivity_mask = sensitivity_scores < sensitivity_threshold
            activity_mask = activity_scores < activity_threshold
            
            # Calculate importance scores
            importance_scores = (0.4 * (1 - sensitivity_scores) + 0.3 * (1 - activity_scores))
            if len(correlated_pairs) > 0:
                importance_scores += 0.3 * correlation_mask.float()
            
            return {
                'correlation_mask': correlation_mask,
                'importance_scores': importance_scores,
                'sensitivity_mask': sensitivity_mask,
                'activity_mask': activity_mask,
                'correlation_matrix': correlation_matrix,
                'sensitivity_scores': sensitivity_scores,
                'activity_scores': activity_scores,
                'correlated_pairs': correlated_pairs
            }

def create_pruned_model(model, pruning_info, input_features, hidden_neurons, num_classes, X_train_tensor):
    """
    Create a completely new, smaller model with only the neurons we want to keep.
    First adjusts weights using linear regression, then copies to a new model.
    
    Args:
        model: Original model
        pruning_info: Dictionary containing pruning information
        input_features: Number of input features
        hidden_neurons: Number of hidden neurons
        num_classes: Number of output classes
        X_train_tensor: Training data tensor for weight adjustment
        
    Returns:
        New, smaller pruned model with only the kept neurons
    """
    # Check if model is a tensor instead of a model object
    if isinstance(model, torch.Tensor):
        logger.warning(f"Model is a tensor, not a model object. Cannot create pruned model.")
        # Return a dummy model
        if num_classes == 2:
            pruned_model = BinaryModel(input_features, hidden_neurons, activation='relu')
        else:
            pruned_model = CategoricalModel(input_features, hidden_neurons, num_classes, activation='relu')
        return pruned_model
    
    # Get pruned indices
    pruned_indices = pruning_info['pruned_indices']
    
    # Calculate number of neurons to keep
    num_neurons_to_keep = hidden_neurons - len(pruned_indices)
    
    # Get indices of neurons to keep
    kept_indices = [i for i in range(hidden_neurons) if i not in pruned_indices]
    
    # Get activation function from model or pruning_info
    activation = getattr(model, 'activation_name', None)
    if activation is None:
        activation = pruning_info.get('activation', 'relu')
        logger.warning(f"Model activation not found, using activation from pruning_info: {activation}")
    
    # First, adjust weights using linear regression
    with torch.no_grad():
        # Get original model's hidden layer output
        original_hidden_output = model.hidden(X_train_tensor)
        
        # Create temporary tensors for adjusted weights
        adjusted_hidden_weights = model.hidden.weight.data.clone()
        adjusted_output_weights = model.output.weight.data.clone()
        
        # For each pruned neuron, find the best linear combination of kept neurons
        for pruned_idx in pruned_indices:
            # Get the output of the pruned neuron
            pruned_neuron_output = original_hidden_output[:, pruned_idx].cpu().numpy()
            
            # Get the outputs of the kept neurons
            kept_neurons_output = original_hidden_output[:, kept_indices].cpu().numpy()
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(kept_neurons_output, pruned_neuron_output)
            
            # Adjust weights of the output layer
            # Use the actual number of output neurons instead of num_classes
            num_output_neurons = model.output.weight.data.shape[0]
            for i in range(num_output_neurons):
                # Get the weight of the pruned neuron in the output layer
                pruned_weight = model.output.weight.data[i, pruned_idx].item()
                
                # Distribute this weight among the kept neurons based on the regression coefficients
                for j, kept_idx in enumerate(kept_indices):
                    # Add the contribution of the pruned neuron to each kept neuron
                    adjusted_output_weights[i, kept_idx] += pruned_weight * reg.coef_[j]
    
    # Now create a new model with the reduced number of hidden neurons
    if num_classes == 2:
        pruned_model = BinaryModel(input_features, num_neurons_to_keep, activation=activation)
        # Ensure output layer has correct size for binary classification
        pruned_model.output = nn.Linear(num_neurons_to_keep, 1)
    else:
        pruned_model = CategoricalModel(input_features, num_neurons_to_keep, num_classes, activation=activation)
    
    # Copy adjusted weights and biases for kept neurons
    with torch.no_grad():
        # Copy hidden layer weights
        for i, kept_idx in enumerate(kept_indices):
            pruned_model.hidden.weight.data[i] = adjusted_hidden_weights[kept_idx]
            pruned_model.hidden.bias.data[i] = model.hidden.bias.data[kept_idx]
        
        # Copy output layer weights
        if num_classes == 2:
            # For binary classification, only copy the first output neuron's weights
            pruned_model.output.weight.data = adjusted_output_weights[0:1, kept_indices]
            pruned_model.output.bias.data = model.output.bias.data[0:1]
        else:
            pruned_model.output.weight.data = adjusted_output_weights[:, kept_indices]
            pruned_model.output.bias.data = model.output.bias.data
    
    return pruned_model

def get_validation_config(dataset_name: str) -> Dict[str, Any]:
    """Get validation configuration for a specific dataset from config file."""
    config_path = os.path.join('config', f'config_{dataset_name}.yaml')
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found for dataset {dataset_name}. Using default validation config.")
        # Return a default configuration
        return {
            'max_accuracy_drop': 0.10,
            'min_accuracy': 0.85,
            'require_min_neurons': 5,
            'max_neurons_pruned': None,  # Will be set dynamically based on model size
            'validation_strategy': 'single',
            'batch_size': 1,
            'accuracy_weight': 1.0,
            'size_weight': 0.0,
            'target_pruning_rate': 0.40  # Target 40% pruning rate
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validation_config = config.get('pruning', {}).get('validation', {})
    
    if not validation_config:
        logger.warning(f"Validation config not found in {config_path}. Using default validation config.")
        validation_config = {
            'max_accuracy_drop': 0.10,
            'min_accuracy': 0.85,
            'require_min_neurons': 5,
            'max_neurons_pruned': None,  # Will be set dynamically based on model size
            'validation_strategy': 'single',
            'batch_size': 1,
            'accuracy_weight': 1.0,
            'size_weight': 0.0,
            'target_pruning_rate': 0.40  # Target 40% pruning rate
        }
    
    # Add target pruning rate if not present
    if 'target_pruning_rate' not in validation_config:
        validation_config['target_pruning_rate'] = 0.40  # Target 40% pruning rate
    
    return validation_config

def validate_pruned_model(model, X_test_tensor, y_test_tensor):
    """Validate pruned model accuracy."""
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X_test_tensor)
        accuracy = calculate_accuracy(outputs, y_test_tensor, model.output.out_features)
    return accuracy

def validate_pruning_candidate(model, pruning_info, X_test_tensor, y_test_tensor, 
                             original_accuracy, validation_config, num_classes, X_train_tensor):
    """Validate a pruning candidate based on dataset-specific criteria."""
    # Check if there are any pruned indices
    if not pruning_info['pruned_indices']:
        logger.warning("No pruned indices provided for validation")
        return False, 0.0, original_accuracy
    
    # Create temporary pruned model
    temp_pruning_info = {
        'pruned_indices': pruning_info['pruned_indices'],
        'correlation_results': pruning_info.get('correlation_results', {}),
        'low_activity_indices': pruning_info.get('low_activity_indices', []),
        'insensitive_indices': pruning_info.get('insensitive_indices', []),
        'sensitivity_scores': pruning_info.get('sensitivity_scores', []),
        'activation': getattr(model, 'activation_name', 'relu')
    }
    
    # Create pruned model
    temp_model = create_pruned_model(
        model, temp_pruning_info, X_test_tensor.shape[1], 
        model.hidden.out_features, num_classes, X_train_tensor
    )
    
    # Evaluate temporary model
    temp_model.eval()
    with torch.no_grad():
        if num_classes > 2:
            temp_test_outputs, _ = temp_model(X_test_tensor)
        else:
            temp_test_outputs, _ = temp_model(X_test_tensor)
        
        temp_test_accuracy = calculate_accuracy(temp_test_outputs, y_test_tensor, num_classes)
    
    # Calculate accuracy drop
    accuracy_drop = (original_accuracy - temp_test_accuracy) / 100.0
    
    # Get neuron type (insensitive, low activity, or correlated)
    neuron_type = "unknown"
    pruned_idx = pruning_info['pruned_indices'][0]  # We're validating one at a time
    if pruned_idx in pruning_info.get('insensitive_indices', []):
        neuron_type = "insensitive"
    elif pruned_idx in pruning_info.get('low_activity_indices', []):
        neuron_type = "low activity"
    elif any(pruned_idx in pair for pair in pruning_info.get('correlation_results', {}).get('correlated_neurons', [])):
        neuron_type = "correlated"
    
    # Get validation criteria from config
    max_drop = validation_config['max_accuracy_drop']
    min_acc = validation_config['min_accuracy']
    require_min_neurons = validation_config['require_min_neurons']
    
    # Adjust validation criteria based on neuron type
    if neuron_type == "insensitive":
        # More lenient for insensitive neurons
        max_drop *= 1.2  # Allow 20% more accuracy drop
        min_acc *= 0.9  # Allow 10% lower minimum accuracy
    elif neuron_type == "low activity":
        # Slightly more lenient for low activity neurons
        max_drop *= 1.1  # Allow 10% more accuracy drop
        min_acc *= 0.95  # Allow 5% lower minimum accuracy
    elif neuron_type == "correlated":
        # Stricter for correlated neurons
        max_drop *= 0.8  # Allow 20% less accuracy drop
        min_acc *= 1.05  # Require 5% higher minimum accuracy
    
    # Check validation criteria
    is_valid = True
    validation_reasons = []
    
    # Check accuracy drop
    if accuracy_drop > max_drop:
        is_valid = False
        validation_reasons.append(f"Accuracy drop {accuracy_drop*100:.2f}% exceeds maximum allowed {max_drop*100:.2f}%")
    
    # Check minimum accuracy
    if temp_test_accuracy < min_acc * 100:
        is_valid = False
        validation_reasons.append(f"New accuracy {temp_test_accuracy:.2f}% below minimum required {min_acc*100:.2f}%")
    
    # Check minimum neurons
    if temp_model.hidden.out_features < require_min_neurons:
        is_valid = False
        validation_reasons.append(f"Remaining neurons {temp_model.hidden.out_features} below minimum required {require_min_neurons}")
    
    # Log detailed validation information
    logger.info(f"Validating neuron {pruned_idx} ({neuron_type}):")
    logger.info(f"  - Original accuracy: {original_accuracy:.2f}%")
    logger.info(f"  - New accuracy: {temp_test_accuracy:.2f}%")
    logger.info(f"  - Accuracy drop: {accuracy_drop*100:.2f}%")
    logger.info(f"  - Max allowed drop: {max_drop*100:.2f}%")
    logger.info(f"  - Min required accuracy: {min_acc*100:.2f}%")
    logger.info(f"  - Remaining neurons: {temp_model.hidden.out_features}")
    logger.info(f"  - Required minimum neurons: {require_min_neurons}")
    
    if not is_valid:
        logger.info("  - Rejected for the following reasons:")
        for reason in validation_reasons:
            logger.info(f"    * {reason}")
    
    return is_valid, accuracy_drop, temp_test_accuracy

def identify_pruning_candidates(model, X_train_tensor, y_train_tensor, pruning_info, 
                              pruning_config, num_classes, X_test_tensor, y_test_tensor):
    """Identify neurons that can be pruned based on activation patterns and weight correlations."""
    # Get correlation results
    correlation_results = pruning_info.get('correlation_results', {})
    if not correlation_results:
        logger.warning("No correlation results found, skipping correlation-based pruning")
        correlation_results = {'correlated_neurons': []}
    
    # Get sensitivity results
    sensitivity_results = pruning_info.get('sensitivity_results', {})
    if not sensitivity_results:
        logger.warning("No sensitivity results found, skipping sensitivity-based pruning")
        sensitivity_results = {'insensitive_indices': []}
    
    # Get activity results
    activity_results = pruning_info.get('activity_results', {})
    if not activity_results:
        logger.warning("No activity results found, skipping activity-based pruning")
        activity_results = {'low_activity_indices': []}
    
    # Initialize pruning candidates
    pruning_candidates = []
    
    # Get original accuracy
    model.eval()
    with torch.no_grad():
        if num_classes > 2:
            test_outputs, _ = model(X_test_tensor)
        else:
            test_outputs, _ = model(X_test_tensor)
        original_accuracy = calculate_accuracy(test_outputs, y_test_tensor, num_classes)
    
    # Get validation config
    validation_config = pruning_config.get('validation', {
        'max_accuracy_drop': 0.05,  # 5% maximum accuracy drop
        'min_accuracy': 0.8,  # 80% minimum accuracy
        'require_min_neurons': 5  # Minimum number of neurons to keep
    })
    
    # Process correlation-based candidates - more aggressive approach
    # First, check if we have correlated pairs
    if 'correlated_pairs' in correlation_results:
        logger.info(f"Processing {len(correlation_results['correlated_pairs'])} correlated pairs")
        
        # Sort pairs by correlation strength (if available)
        sorted_pairs = correlation_results['correlated_pairs']
        if 'correlation_matrix' in correlation_results:
            # Sort by correlation strength (descending)
            sorted_pairs = sorted(
                sorted_pairs, 
                key=lambda pair: correlation_results['correlation_matrix'][pair[0], pair[1]].item(),
                reverse=True
            )
        
        # Process top 30% of correlated pairs
        num_pairs_to_process = max(1, int(len(sorted_pairs) * 0.3))
        logger.info(f"Processing top {num_pairs_to_process} correlated pairs")
        
        for i, (neuron1, neuron2) in enumerate(sorted_pairs[:num_pairs_to_process]):
            # Skip if either neuron is already in pruning candidates
            if any(neuron1 in candidate['pruned_indices'] or 
                   neuron2 in candidate['pruned_indices'] 
                   for candidate in pruning_candidates):
                continue
            
            # Create pruning info for this candidate
            candidate_info = {
                'pruned_indices': [neuron1],  # Prune first neuron of the pair
                'correlation_results': correlation_results,
                'sensitivity_results': sensitivity_results,
                'activity_results': activity_results,
                'activation': getattr(model, 'activation_name', 'relu')
            }
            
            # Validate the candidate
            is_valid, accuracy_drop, new_accuracy = validate_pruning_candidate(
                model, candidate_info, X_test_tensor, y_test_tensor,
                original_accuracy, validation_config, num_classes, X_train_tensor
            )
            
            if is_valid:
                pruning_candidates.append({
                    'pruned_indices': [neuron1],
                    'accuracy_drop': accuracy_drop,
                    'new_accuracy': new_accuracy,
                    'reason': f"Correlated with neuron {neuron2}"
                })
                logger.info(f"Added neuron {neuron1} to pruning candidates (correlated with {neuron2})")
    
    # Process correlation mask if available
    if 'correlation_mask' in correlation_results:
        correlation_mask = correlation_results['correlation_mask']
        if isinstance(correlation_mask, torch.Tensor):
            # Get indices of highly correlated neurons
            correlated_indices = torch.where(correlation_mask)[0].tolist()
            logger.info(f"Found {len(correlated_indices)} highly correlated neurons from mask")
            
            # Process top 30% of correlated neurons
            num_neurons_to_process = max(1, int(len(correlated_indices) * 0.3))
            logger.info(f"Processing top {num_neurons_to_process} correlated neurons")
            
            for i, neuron_idx in enumerate(correlated_indices[:num_neurons_to_process]):
                # Skip if neuron is already in pruning candidates
                if any(neuron_idx in candidate['pruned_indices'] 
                       for candidate in pruning_candidates):
                    continue
                
                # Create pruning info for this candidate
                candidate_info = {
                    'pruned_indices': [neuron_idx],
                    'correlation_results': correlation_results,
                    'sensitivity_results': sensitivity_results,
                    'activity_results': activity_results,
                    'activation': getattr(model, 'activation_name', 'relu')
                }
                
                # Validate the candidate
                is_valid, accuracy_drop, new_accuracy = validate_pruning_candidate(
                    model, candidate_info, X_test_tensor, y_test_tensor,
                    original_accuracy, validation_config, num_classes, X_train_tensor
                )
                
                if is_valid:
                    pruning_candidates.append({
                        'pruned_indices': [neuron_idx],
                        'accuracy_drop': accuracy_drop,
                        'new_accuracy': new_accuracy,
                        'reason': "Highly correlated with other neurons"
                    })
                    logger.info(f"Added neuron {neuron_idx} to pruning candidates (highly correlated)")
    
    # Process sensitivity-based candidates
    for neuron_idx in sensitivity_results.get('insensitive_indices', []):
        # Skip if neuron is already in pruning candidates
        if any(neuron_idx in candidate['pruned_indices'] 
               for candidate in pruning_candidates):
            continue
            
        # Create pruning info for this candidate
        candidate_info = {
            'pruned_indices': [neuron_idx],
            'correlation_results': correlation_results,
            'sensitivity_results': sensitivity_results,
            'activity_results': activity_results,
            'activation': getattr(model, 'activation_name', 'relu')
        }
        
        # Validate the candidate
        is_valid, accuracy_drop, new_accuracy = validate_pruning_candidate(
            model, candidate_info, X_test_tensor, y_test_tensor,
            original_accuracy, validation_config, num_classes, X_train_tensor
        )
        
        if is_valid:
            pruning_candidates.append({
                'pruned_indices': [neuron_idx],
                'accuracy_drop': accuracy_drop,
                'new_accuracy': new_accuracy,
                'reason': "Low sensitivity to input changes"
            })
            logger.info(f"Added neuron {neuron_idx} to pruning candidates (low sensitivity)")
    
    # Process activity-based candidates
    for neuron_idx in activity_results.get('low_activity_indices', []):
        # Skip if neuron is already in pruning candidates
        if any(neuron_idx in candidate['pruned_indices'] 
               for candidate in pruning_candidates):
            continue
    
        # Create pruning info for this candidate
        candidate_info = {
            'pruned_indices': [neuron_idx],
            'correlation_results': correlation_results,
            'sensitivity_results': sensitivity_results,
            'activity_results': activity_results,
            'activation': getattr(model, 'activation_name', 'relu')
        }
        
        # Validate the candidate
        is_valid, accuracy_drop, new_accuracy = validate_pruning_candidate(
            model, candidate_info, X_test_tensor, y_test_tensor,
            original_accuracy, validation_config, num_classes, X_train_tensor
        )
        
        if is_valid:
            pruning_candidates.append({
                'pruned_indices': [neuron_idx],
                'accuracy_drop': accuracy_drop,
                'new_accuracy': new_accuracy,
                'reason': "Low activation activity"
            })
            logger.info(f"Added neuron {neuron_idx} to pruning candidates (low activity)")
    
    # Sort candidates by accuracy drop (ascending)
    pruning_candidates.sort(key=lambda x: x['accuracy_drop'])
    
    # Log pruning candidates
    logger.info(f"Found {len(pruning_candidates)} valid pruning candidates:")
    for i, candidate in enumerate(pruning_candidates):
        logger.info(f"Candidate {i+1}:")
        logger.info(f"  - Neurons to prune: {candidate['pruned_indices']}")
        logger.info(f"  - Accuracy drop: {candidate['accuracy_drop']*100:.2f}%")
        logger.info(f"  - New accuracy: {candidate['new_accuracy']:.2f}%")
        logger.info(f"  - Reason: {candidate['reason']}")
    
    return pruning_candidates

def prune_model(model, X_train_tensor, y_train_tensor, pruning_info, pruning_config, num_classes, X_test_tensor=None, y_test_tensor=None):
    """
    Prune the model based on the pruning configuration and validation data.
    """
    # Get model architecture info
    input_features = model.hidden.in_features
    hidden_neurons = model.hidden.out_features
    activation = getattr(model, 'activation_name', 'relu')
    
    # Analyze hidden layer
    analysis_results = analyze_hidden_layer(
        model, 
        model.hidden, 
        0,  # layer_idx
        X_train_tensor,  # validation_data
        enhanced_pruning=True,
        activation=activation
    )
    
    # Extract results
    correlation_mask = analysis_results['correlation_mask']
    sensitivity_mask = analysis_results['sensitivity_mask']
    activity_mask = analysis_results['activity_mask']
    importance_scores = analysis_results['importance_scores']
    correlated_pairs = analysis_results.get('correlated_pairs', [])
    
    # Log correlation results
    if len(correlated_pairs) > 0:
        logger.info(f"Found {len(correlated_pairs)} correlated neuron pairs:")
        for pair in correlated_pairs:
            logger.info(f"Neurons {pair[0]} and {pair[1]} are highly correlated")
    
    # Log sensitivity results
    insensitive_neurons = torch.where(sensitivity_mask)[0].tolist()
    if insensitive_neurons:
        logger.info(f"Found {len(insensitive_neurons)} insensitive neurons:")
        for neuron in insensitive_neurons:
            logger.info(f"Neuron {neuron}: sensitivity score = {importance_scores[neuron]:.3f}")
    
    # Log activity results
    inactive_neurons = torch.where(activity_mask)[0].tolist()
    if inactive_neurons:
        logger.info(f"Found {len(inactive_neurons)} inactive neurons:")
        for neuron in inactive_neurons:
            logger.info(f"Neuron {neuron}: activity score = {importance_scores[neuron]:.3f}")
    
    # Combine masks to identify neurons to prune
    neurons_to_prune = []
    
    # Add correlated neurons
    if len(correlated_pairs) > 0:
        for pair in correlated_pairs:
            # Keep the neuron with higher importance score
            if importance_scores[pair[0]] > importance_scores[pair[1]]:
                neurons_to_prune.append(pair[1])
            else:
                neurons_to_prune.append(pair[0])
    
    # Add insensitive neurons
    neurons_to_prune.extend(insensitive_neurons)
    
    # Add inactive neurons
    neurons_to_prune.extend(inactive_neurons)
    
    # Remove duplicates and sort
    neurons_to_prune = sorted(list(set(neurons_to_prune)))
    
    # Limit number of neurons to prune based on config
    max_neurons = pruning_config.get('max_neurons_pruned', hidden_neurons // 2)
    if len(neurons_to_prune) > max_neurons:
        # Sort by importance score and take the least important ones
        neurons_to_prune.sort(key=lambda x: importance_scores[x])
        neurons_to_prune = neurons_to_prune[:max_neurons]
    
    # Create pruned model
    pruned_model = create_pruned_model(model, {'pruned_indices': neurons_to_prune}, 
                                     input_features, hidden_neurons, num_classes, X_train_tensor)
    
    # Validate pruned model
    if X_test_tensor is not None and y_test_tensor is not None:
        pruned_accuracy = validate_pruned_model(pruned_model, X_test_tensor, y_test_tensor)
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2%}")
    
    return pruned_model, neurons_to_prune

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
        
        # Check if model is a tensor instead of a model object
        if isinstance(model, torch.Tensor):
            logger.warning(f"Model is a tensor, not a model object. Skipping pruning.")
            return
        
        # Get pruning parameters
        correlation_threshold = config['pruning']['correlation_threshold']
        activity_threshold = config['pruning']['activity_threshold']
        sensitivity_threshold = config['pruning']['sensitivity_threshold']
        enhanced_pruning = config['pruning'].get('enhanced_pruning', False)
        activation = config['model'].get('activation', 'relu')
        
        # Get validation config
        validation_config = get_validation_config(config['dataset']['name'])
        target_pruning_rate = validation_config.get('target_pruning_rate', 0.40)
        
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
        
        # Check if model.hidden is a proper PyTorch layer
        if not isinstance(model.hidden, nn.Linear):
            logger.warning(f"model.hidden is not a proper PyTorch Linear layer: {type(model.hidden)}")
            logger.warning("Skipping pruning.")
            return
        
        # Analyze hidden layer for correlations
        correlation_results = analyze_hidden_layer(
            model, 
            model.hidden, 
            0, 
            X_train_tensor,  # Pass only the input tensor
            enhanced_pruning=True,
            activation=activation
        )
        
        # Identify pruning candidates
        pruning_candidates = identify_pruning_candidates(
            model, 
            X_train_tensor, 
            y_train_tensor, 
            correlation_results, 
            validation_config, 
            config['dataset']['num_classes'], 
            X_test_tensor, 
            y_test_tensor
        )
        
        # Analyze neuron sensitivity
        insensitive_neurons, sensitivity_scores = analyze_neuron_sensitivity(
            model, 
            X_train_tensor, 
            hidden_layer_output, 
            sensitivity_threshold,
            activation
        )
        
        # Combine pruning candidates
        pruning_candidates = set(pruning_candidates)
        pruning_candidates.update(insensitive_neurons)
        
        # Add correlated neurons
        for pair in correlation_results.get('correlated', []):
            i, j = pair
            if j in pruning_candidates:  # Only add if the dependent neuron is already a candidate
                pruning_candidates.add(i)
        
        # Calculate target number of neurons to prune based on target pruning rate
        total_neurons = model.hidden.out_features
        target_neurons_to_prune = int(total_neurons * target_pruning_rate)
        
        logger.info(f"Total neurons: {total_neurons}")
        logger.info(f"Target pruning rate: {target_pruning_rate*100:.1f}%")
        logger.info(f"Target neurons to prune: {target_neurons_to_prune}")
        logger.info(f"Initial pruning candidates: {len(pruning_candidates)}")
        
        # If we don't have enough candidates, we need to be more aggressive
        if len(pruning_candidates) < target_neurons_to_prune:
            logger.info(f"Not enough pruning candidates ({len(pruning_candidates)}) to meet target ({target_neurons_to_prune})")
            logger.info("Using importance scores to identify additional candidates")
            
            # Get importance scores
            importance_results = calculate_neuron_importance(
                model, 
                model.hidden, 
                0, 
                X_train_tensor,
                activation=activation
            )
            
            # Sort neurons by importance (ascending order - less important first)
            importance_scores = importance_results['importance_scores']
            sorted_indices = torch.argsort(importance_scores)
            
            # Add additional candidates based on importance scores
            additional_candidates = []
            for idx in sorted_indices:
                if idx.item() not in pruning_candidates and len(additional_candidates) < (target_neurons_to_prune - len(pruning_candidates)):
                    additional_candidates.append(idx.item())
            
            pruning_candidates.update(additional_candidates)
            logger.info(f"Added {len(additional_candidates)} additional candidates based on importance scores")
            logger.info(f"Total pruning candidates: {len(pruning_candidates)}")
        
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
            'num_features': config['dataset']['input_features'],
            'output_classes': config['dataset']['num_classes'],
            'input_features': config['dataset']['input_features']
        }
        
        # Create pruned model with new weight adjustment approach
        pruned_model, neurons_to_prune = prune_model(
            model, 
            X_train_tensor, 
            y_train_tensor, 
            pruning_info, 
            validation_config, 
            config['dataset']['num_classes'], 
            X_test_tensor, 
            y_test_tensor
        )
        
        # Evaluate pruned model
        pruned_model.eval()
        with torch.no_grad():
            if config['dataset']['num_classes'] > 2:
                pruned_outputs, _ = pruned_model(X_test_tensor)
            else:
                pruned_outputs, _ = pruned_model(X_test_tensor)
            
            pruned_accuracy = calculate_accuracy(pruned_outputs, y_test_tensor, config['dataset']['num_classes'])
        
        # Evaluate original model for comparison
        model.eval()
        with torch.no_grad():
            if config['dataset']['num_classes'] > 2:
                original_outputs, _ = model(X_test_tensor)
            else:
                original_outputs, _ = model(X_test_tensor)
            
            original_accuracy = calculate_accuracy(original_outputs, y_test_tensor, config['dataset']['num_classes'])
        
        # Calculate accuracy drop
        accuracy_drop = (original_accuracy - pruned_accuracy) / 100.0
        
        logger.info(f"Original model accuracy: {original_accuracy:.2f}%")
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2f}%")
        logger.info(f"Accuracy drop: {accuracy_drop*100:.2f}%")
        logger.info(f"Pruning rate: {len(pruning_candidates)/model.hidden.out_features*100:.2f}%")
        
        # Check if accuracy drop is acceptable
        max_accuracy_drop = validation_config.get('max_accuracy_drop', 0.10)
        if accuracy_drop > max_accuracy_drop:
            logger.warning(f"Accuracy drop ({accuracy_drop*100:.2f}%) exceeds maximum allowed ({max_accuracy_drop*100:.2f}%)")
            logger.warning("Consider adjusting pruning parameters or using a more conservative approach")
        
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
            'correlated_neurons': pruning_info['correlation_results'].get('correlated_neurons', []),
            'original_hidden_neurons': model_info['hidden_neurons'],
            'pruned_hidden_neurons': pruned_model.hidden.out_features,
            'pruning_rate': len(pruning_candidates)/model.hidden.out_features,
            'accuracy_drop': accuracy_drop,
            'original_accuracy': original_accuracy,
            'pruned_accuracy': pruned_accuracy
        })
        
        with open(os.path.join(pruned_model_path, f"{config['dataset']['name']}_model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4, cls=NumpyEncoder)
        
        # Save pruning metrics
        metrics_path = os.path.join(config['output']['results_dir'], config['dataset']['name'], f"{config['dataset']['name']}_pruned_metrics_{timestamp}.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(pruning_info, f, indent=4, cls=NumpyEncoder)
        
        logger.info(f"Pruned model saved to {pruned_model_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Update config with pruning information
        config['pruning'].update({
            'pruned_indices': pruning_info['pruned_indices'],
            'correlation_results': pruning_info['correlation_results'],
            'low_activity_neurons': pruning_info['low_activity_indices'],
            'correlated_neurons': pruning_info['correlation_results'].get('correlated_neurons', []),
            'original_hidden_neurons': model_info['hidden_neurons'],
            'pruned_hidden_neurons': pruned_model.hidden.out_features,
            'final_accuracy': pruned_accuracy,
            'accuracy_drop': accuracy_drop,
            'pruning_rate': len(pruning_candidates)/model.hidden.out_features
        })
        
        # Save updated config
        config_path = os.path.join(config['output']['results_dir'], config['dataset']['name'], f"{config['dataset']['name']}_pruned_config_{timestamp}.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
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

def main():
    """
    Main function to run the pruning process independently.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prune a neural network model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Prune model
    results = prune_model(config)
    
    logger.info("Pruning completed successfully")
    logger.info(f"Pruned model saved to {results['pruned_model_path']}")

if __name__ == '__main__':
    main() 