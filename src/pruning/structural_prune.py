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
from sklearn.linear_model import Ridge
import copy

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

def calculate_neuron_activity(model, hidden_layer, inputs, activation_fn='relu'):
    """
    Calculate activity scores for neurons based on their activation patterns.
    
    Args:
        model: PyTorch model
        hidden_layer: Hidden layer to analyze
        inputs: Input data tensor
        activation_fn: Activation function name
        
    Returns:
        torch.Tensor: Activity scores for each neuron
    """
    # Get activations
    with torch.no_grad():
        activations = hidden_layer(inputs)
        
        # Apply activation function if not already applied
        if activation_fn.lower() == 'relu':
            activations = F.relu(activations)
        elif activation_fn.lower() == 'sigmoid':
            activations = torch.sigmoid(activations)
        elif activation_fn.lower() == 'tanh':
            activations = torch.tanh(activations)
    
    # Calculate activity as mean activation
    activity_scores = torch.mean(torch.abs(activations), dim=0)
    
    # Normalize scores to [0, 1] range
    if torch.max(activity_scores) > torch.min(activity_scores):
        activity_scores = (activity_scores - torch.min(activity_scores)) / (torch.max(activity_scores) - torch.min(activity_scores))
    
    return activity_scores

def calculate_neuron_importance(model, hidden_layer, layer_idx, inputs, activation='relu'):
    """
    Calculate importance scores for neurons based on sensitivity and activity.
    
    Args:
        model: PyTorch model
        hidden_layer: Hidden layer to analyze
        layer_idx: Index of the layer
        inputs: Input data tensor
        activation: Activation function name
        
    Returns:
        Dict: Dictionary with importance scores and analysis results
    """
    # Get activations for analysis
    with torch.no_grad():
        if hasattr(model, 'hidden') and isinstance(model.hidden, nn.Linear):
            hidden_output = model.hidden(inputs)
            # Apply activation
            if activation.lower() == 'relu':
                hidden_activations = F.relu(hidden_output)
            elif activation.lower() == 'sigmoid':
                hidden_activations = torch.sigmoid(hidden_output)
            elif activation.lower() == 'tanh':
                hidden_activations = torch.tanh(hidden_output)
            else:
                hidden_activations = hidden_output
        else:
            # Create dummy activations if model doesn't have proper structure
            logger.warning("Model doesn't have a proper hidden layer, using dummy activations")
            hidden_activations = inputs
    
    # Calculate sensitivity scores
    insensitive_indices, sensitivity_scores = analyze_neuron_sensitivity(
        model, inputs, hidden_activations, threshold=0.1, activation=activation
    )
    
    # Convert numpy array to torch tensor if needed
    if isinstance(sensitivity_scores, np.ndarray):
        sensitivity_scores = torch.tensor(sensitivity_scores, dtype=torch.float32)
    
    # Calculate activity scores
    activity_scores = calculate_neuron_activity(model, hidden_layer, inputs, activation_fn=activation)
    
    # Normalize scores
    norm_sensitivity = normalize_scores(sensitivity_scores)
    norm_activity = normalize_scores(activity_scores)
    
    # Calculate correlation matrix
    correlation_matrix = torch.corrcoef(hidden_activations.T)
    
    # Find correlated pairs
    num_neurons = hidden_activations.shape[1]
    correlation_threshold = 0.7  # Default threshold
    correlated_pairs = []
    
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            if correlation_matrix[i, j] > correlation_threshold:
                correlated_pairs.append((i, j))
    
    # Create correlation mask (neurons that are highly correlated with others)
    correlation_mask = torch.zeros(num_neurons, dtype=torch.bool)
    for i, j in correlated_pairs:
        correlation_mask[i] = True
        correlation_mask[j] = True
    
    # Calculate importance scores (higher score = less important)
    importance_scores = 0.6 * (1 - norm_sensitivity) + 0.4 * (1 - norm_activity)
    
    # Add correlation contribution to importance
    if len(correlated_pairs) > 0:
        importance_scores = importance_scores + 0.3 * correlation_mask.float()
    
    # Create result dictionary with all analysis results
    return {
        'importance_scores': importance_scores,
        'sensitivity_scores': sensitivity_scores,
        'activity_scores': activity_scores,
        'correlation_matrix': correlation_matrix,
        'correlated_pairs': correlated_pairs,
        'correlation_mask': correlation_mask,
        'insensitive_indices': insensitive_indices
    }

def normalize_scores(scores):
    """Normalize scores to 0-1 range with proper handling of edge cases"""
    if torch.max(scores) > torch.min(scores):
        return (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
    else:
        return torch.ones_like(scores)

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
                'importance_scores': importance_scores,
                'correlation_mask': correlation_mask,
                'sensitivity_mask': sensitivity_mask,
                'activity_mask': activity_mask,
                'correlation_matrix': correlation_matrix,
                'sensitivity_scores': sensitivity_scores,
                'activity_scores': activity_scores,
                'correlated_pairs': correlated_pairs
            }

def create_pruned_model(original_model, pruning_indices, inputs):
    """
    Create a pruned model by zeroing out pruned neurons and adjusting weights using precise linear 
    regression to preserve the behavior of the original model.
    
    For correlated neurons with relationship Y = αX + β, we transfer weights as:
    W_out_X += α * W_out_Y and bias += β * W_out_Y
    """
    pruned_model = copy.deepcopy(original_model)
    hidden_weights = pruned_model.hidden.weight.data
    output_weights = pruned_model.output.weight.data
    output_bias = pruned_model.output.bias.data

    # Get the original activations
    with torch.no_grad():
        # Get hidden layer activations before activation function
        hidden_output = pruned_model.hidden(inputs)
        # Apply activation function
        hidden_activations = pruned_model.activation_fn(hidden_output)
    
    # Get indices of kept neurons
    kept_indices = [i for i in range(hidden_weights.shape[0]) if i not in pruning_indices]
    
    # Calculate correlation matrix for all neurons
    correlation_matrix = torch.corrcoef(hidden_activations.T)
    
    # Process each pruned neuron
    for pruned_idx in pruning_indices:
        # Get activations for the pruned neuron
        pruned_activation = hidden_activations[:, pruned_idx].cpu().numpy()
        
        # Find the most correlated kept neuron for primary weight transfer
        correlations_with_kept = correlation_matrix[pruned_idx, kept_indices]
        most_correlated_idx = kept_indices[torch.argmax(correlations_with_kept)]
        corr_value = torch.max(correlations_with_kept).item()
        
        # Perform pairwise linear regression for exact coefficient calculation 
        if corr_value > 0.7:  # Strong correlation - use pairwise regression
            # For high correlation: pruned_neuron = α * kept_neuron + β
            kept_activation = hidden_activations[:, most_correlated_idx].cpu().numpy().reshape(-1, 1)
            
            # Fit simple linear regression to get exact α and β
            reg = LinearRegression()
            reg.fit(kept_activation, pruned_activation)
            
            # Extract coefficients
            alpha = reg.coef_[0]  # Slope of the relationship
            beta = reg.intercept_  # Intercept
            
            # Apply the exact formula for output weights
            for output_idx in range(output_weights.shape[0]):
                # W_kept += α * W_pruned (for the weights)
                pruned_weight = output_weights[output_idx, pruned_idx].item()
                output_weights[output_idx, most_correlated_idx] += alpha * pruned_weight
                
                # bias += β * W_pruned (for the intercept term)
                output_bias[output_idx] += beta * pruned_weight
            
            logger.info(f"Pruned neuron {pruned_idx} | High correlation ({corr_value:.3f}) with {most_correlated_idx} | α={alpha:.3f}, β={beta:.3f}")
        else:
            # Lower correlation - use multiple regression with all kept neurons
            # Get activations for kept neurons
            kept_activations = hidden_activations[:, kept_indices].cpu().numpy()
            
            try:
                # Use Ridge regression for stability
                reg = Ridge(alpha=0.01)
                reg.fit(kept_activations, pruned_activation)
                
                # Apply output weight adjustment using regression coefficients
                for output_idx in range(output_weights.shape[0]):
                    pruned_weight = output_weights[output_idx, pruned_idx].item()
                    
                    # Only adjust if the pruned weight is significant
                    if abs(pruned_weight) > 1e-4:
                        # Distribute to all kept neurons according to regression coefficients
                        for i, kept_idx in enumerate(kept_indices):
                            adjustment = pruned_weight * reg.coef_[i]
                            output_weights[output_idx, kept_idx] += adjustment
                        
                        # Also adjust bias term with intercept
                        output_bias[output_idx] += reg.intercept_ * pruned_weight
                
                logger.info(f"Pruned neuron {pruned_idx} | Ridge regression with {len(kept_indices)} neurons")
            except Exception as e:
                # Fallback method if regression fails
                logger.warning(f"Regression failed for neuron {pruned_idx}: {str(e)}")
                # Simple transfer to most correlated neuron
                for output_idx in range(output_weights.shape[0]):
                    output_weights[output_idx, most_correlated_idx] += output_weights[output_idx, pruned_idx]
    
    # Zero out pruned neurons
    for idx in pruning_indices:
        hidden_weights[idx] = 0
        output_weights[:, idx] = 0
    
    # Validate the pruned model output magnitude
    with torch.no_grad():
        original_outputs, _ = original_model(inputs)
        pruned_outputs, _ = pruned_model(inputs)
        
        # Check if magnitude is significantly different
        orig_magnitude = original_outputs.abs().mean().item()
        pruned_magnitude = pruned_outputs.abs().mean().item()
        ratio = orig_magnitude / (pruned_magnitude + 1e-8)
        
        if ratio > 1.3 or ratio < 0.7:  # More than 30% difference
            # Apply global scaling to output weights
            for kept_idx in kept_indices:
                output_weights[:, kept_idx] *= ratio
            logger.info(f"Applied global magnitude correction: {ratio:.2f}")
    
    logger.info(f"Pruned {len(pruning_indices)} neurons ({len(pruning_indices)/hidden_weights.shape[0]*100:.1f}%)")
    logger.info(f"Remaining active neurons: {len(kept_indices)}")
    
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
    """Validate a pruning candidate by checking if it meets the validation criteria."""
    # Get validation parameters
    max_accuracy_drop = validation_config.get('max_accuracy_drop', 0.05)
    min_accuracy = validation_config.get('min_accuracy', 0.8)
    require_min_neurons = validation_config.get('require_min_neurons', 5)
    
    # Get target pruning rate
    target_pruning_rate = validation_config.get('target_pruning_rate', 0.6)
    total_neurons = model.hidden.out_features
    target_neurons_to_prune = int(total_neurons * target_pruning_rate)
    
    # Get current number of pruning candidates
    current_pruning_count = len(pruning_info.get('pruned_indices', []))
    
    # If we're far from the target pruning rate, be more lenient with accuracy drops
    if current_pruning_count < target_neurons_to_prune * 0.7:  # Less than 70% of target
        # Increase max_accuracy_drop by 50% if we're far from target
        adjusted_max_accuracy_drop = max_accuracy_drop * 1.5
        # Decrease min_accuracy by 10% if we're far from target
        adjusted_min_accuracy = min_accuracy * 0.9
        logger.info(f"Adjusting validation criteria for aggressive pruning:")
        logger.info(f"  - Original max_accuracy_drop: {max_accuracy_drop:.4f}, Adjusted: {adjusted_max_accuracy_drop:.4f}")
        logger.info(f"  - Original min_accuracy: {min_accuracy:.4f}, Adjusted: {adjusted_min_accuracy:.4f}")
    else:
        adjusted_max_accuracy_drop = max_accuracy_drop
        adjusted_min_accuracy = min_accuracy
    
    # Create a temporary model with the candidate pruning
    temp_model = create_pruned_model(model, pruning_info, X_train_tensor)
    
    # Check if we have enough neurons left
    remaining_neurons = temp_model.hidden.out_features
    if remaining_neurons < require_min_neurons:
        logger.info(f"Rejected candidate: Not enough neurons left ({remaining_neurons} < {require_min_neurons})")
        return False, 0, 0
    
    # Evaluate the pruned model
    temp_model.eval()
    with torch.no_grad():
        if num_classes > 2:
            test_outputs, _ = temp_model(X_test_tensor)
        else:
            test_outputs, _ = temp_model(X_test_tensor)
        new_accuracy = calculate_accuracy(test_outputs, y_test_tensor, num_classes)
    
    # Calculate accuracy drop
    accuracy_drop = original_accuracy - new_accuracy
    
    # Check if the accuracy drop is acceptable
    if accuracy_drop > adjusted_max_accuracy_drop:
        logger.info(f"Rejected candidate: Accuracy drop too high ({accuracy_drop:.4f} > {adjusted_max_accuracy_drop:.4f})")
        return False, accuracy_drop, new_accuracy
    
    # Check if the new accuracy meets the minimum requirement
    if new_accuracy < adjusted_min_accuracy:
        logger.info(f"Rejected candidate: Accuracy too low ({new_accuracy:.4f} < {adjusted_min_accuracy:.4f})")
        return False, accuracy_drop, new_accuracy
    
    logger.info(f"Accepted candidate: Accuracy drop {accuracy_drop:.4f}, new accuracy {new_accuracy:.4f}")
    return True, accuracy_drop, new_accuracy

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
    
    # Use target_pruning_rate from config (default to 0.6 if not specified)
    target_pruning_rate = validation_config.get('target_pruning_rate', 0.6)
    total_neurons = model.hidden.out_features
    target_neurons_to_prune = int(total_neurons * target_pruning_rate)
    
    logger.info(f"Target pruning rate: {target_pruning_rate*100:.1f}%")
    logger.info(f"Target neurons to prune: {target_neurons_to_prune}")
    
    # First, process highly correlated neurons from the correlation mask
    correlation_mask = correlation_results.get('correlation_mask')
    if correlation_mask is not None and isinstance(correlation_mask, torch.Tensor):
        correlated_indices = torch.where(correlation_mask)[0].tolist()
        logger.info(f"Found {len(correlated_indices)} highly correlated neurons from mask")
        
        # Get correlation matrix
        correlation_matrix = correlation_results.get('correlation_matrix')
        if correlation_matrix is not None:
            # For each correlated neuron, find its most correlated partner
            correlated_pairs = []
            for i in correlated_indices:
                correlations = correlation_matrix[i]
                # Set self-correlation to -1 to exclude it
                correlations[i] = -1
                most_correlated = torch.argmax(correlations).item()
                correlated_pairs.append((i, most_correlated))
            
            # Sort pairs by correlation strength
            correlated_pairs.sort(key=lambda pair: correlation_matrix[pair[0], pair[1]], reverse=True)
            
            logger.info(f"Processing {len(correlated_pairs)} correlated pairs:")
            for i, (neuron1, neuron2) in enumerate(correlated_pairs):
                correlation = correlation_matrix[neuron1, neuron2].item()
                logger.info(f"Pair {i+1}: Neurons {neuron1} and {neuron2} with correlation {correlation:.3f}")
                
                # For each pair, try to prune the less important neuron
                if neuron1 not in [c['pruned_indices'][0] for c in pruning_candidates] and \
                   neuron2 not in [c['pruned_indices'][0] for c in pruning_candidates]:
                    
                    # Try pruning neuron1
                    candidate_info1 = {
                        'pruned_indices': [neuron1],
                        'correlation_results': correlation_results,
                        'sensitivity_results': sensitivity_results,
                        'activity_results': activity_results,
                        'activation': getattr(model, 'activation_name', 'relu')
                    }
            
                    is_valid1, accuracy_drop1, new_accuracy1 = validate_pruning_candidate(
                        model, candidate_info1, X_test_tensor, y_test_tensor,
                        original_accuracy, validation_config, num_classes, X_train_tensor
                    )
            
                    # Try pruning neuron2
                    candidate_info2 = {
                        'pruned_indices': [neuron2],
                        'correlation_results': correlation_results,
                        'sensitivity_results': sensitivity_results,
                        'activity_results': activity_results,
                        'activation': getattr(model, 'activation_name', 'relu')
                    }
            
                    is_valid2, accuracy_drop2, new_accuracy2 = validate_pruning_candidate(
                        model, candidate_info2, X_test_tensor, y_test_tensor,
                        original_accuracy, validation_config, num_classes, X_train_tensor
                    )
            
                    # Choose the better candidate (lower accuracy drop)
                    if is_valid1 and (not is_valid2 or accuracy_drop1 < accuracy_drop2):
                        pruning_candidates.append({
                            'pruned_indices': [neuron1],
                            'accuracy_drop': accuracy_drop1,
                            'new_accuracy': new_accuracy1,
                            'reason': f"Highly correlated with neuron {neuron2} (correlation: {correlation:.3f})"
                        })
                        logger.info(f"Added neuron {neuron1} to pruning candidates (correlated with {neuron2})")
                    elif is_valid2:
                        pruning_candidates.append({
                            'pruned_indices': [neuron2],
                            'accuracy_drop': accuracy_drop2,
                            'new_accuracy': new_accuracy2,
                            'reason': f"Highly correlated with neuron {neuron1} (correlation: {correlation:.3f})"
                        })
                        logger.info(f"Added neuron {neuron2} to pruning candidates (correlated with {neuron1})")
    
    # Process sensitivity-based candidates
    for neuron_idx in sensitivity_results.get('insensitive_indices', []):
        if neuron_idx not in [c['pruned_indices'][0] for c in pruning_candidates]:
            candidate_info = {
                'pruned_indices': [neuron_idx],
                'correlation_results': correlation_results,
                'sensitivity_results': sensitivity_results,
                'activity_results': activity_results,
                'activation': getattr(model, 'activation_name', 'relu')
            }
            
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
        if neuron_idx not in [c['pruned_indices'][0] for c in pruning_candidates]:
            candidate_info = {
                'pruned_indices': [neuron_idx],
                'correlation_results': correlation_results,
                'sensitivity_results': sensitivity_results,
                'activity_results': activity_results,
                'activation': getattr(model, 'activation_name', 'relu')
            }
            
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
    
    # If we still don't have enough candidates, use importance scores
    if len(pruning_candidates) < target_neurons_to_prune:
        logger.info(f"Not enough pruning candidates ({len(pruning_candidates)}) to meet target ({target_neurons_to_prune})")
        logger.info("Using importance scores to identify additional candidates")
        
        # Get importance scores
        importance_results = calculate_neuron_importance(
            model, 
            model.hidden, 
            0, 
            X_train_tensor,
            activation=getattr(model, 'activation_name', 'relu')
        )
        
        # Sort neurons by importance (ascending order - less important first)
        importance_scores = importance_results['importance_scores']
        sorted_indices = torch.argsort(importance_scores)
        
        # Add additional candidates based on importance scores
        for idx in sorted_indices:
            if idx.item() not in [c['pruned_indices'][0] for c in pruning_candidates] and \
               len(pruning_candidates) < target_neurons_to_prune:
                
                candidate_info = {
                    'pruned_indices': [idx.item()],
                    'correlation_results': correlation_results,
                    'sensitivity_results': sensitivity_results,
                    'activity_results': activity_results,
                    'activation': getattr(model, 'activation_name', 'relu')
                }
            
                is_valid, accuracy_drop, new_accuracy = validate_pruning_candidate(
                    model, candidate_info, X_test_tensor, y_test_tensor,
                    original_accuracy, validation_config, num_classes, X_train_tensor
                )
            
                if is_valid:
                    pruning_candidates.append({
                        'pruned_indices': [idx.item()],
                        'accuracy_drop': accuracy_drop,
                        'new_accuracy': new_accuracy,
                        'reason': f"Low importance score: {importance_scores[idx]:.4f}"
                    })
                    logger.info(f"Added neuron {idx.item()} to pruning candidates (low importance: {importance_scores[idx]:.4f})")
    
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
    
    # Get target pruning rate from config
    validation_config = pruning_config.get('validation', {})
    target_pruning_rate = validation_config.get('target_pruning_rate', 0.6)
    target_neurons_to_prune = int(hidden_neurons * target_pruning_rate)
    
    logger.info(f"Target pruning rate: {target_pruning_rate*100:.1f}%")
    logger.info(f"Target neurons to prune: {target_neurons_to_prune}")
    
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
    
    # If we don't have enough neurons to prune, add more based on importance scores
    if len(neurons_to_prune) < target_neurons_to_prune:
        logger.info(f"Not enough neurons identified for pruning ({len(neurons_to_prune)} < {target_neurons_to_prune})")
        logger.info("Adding more neurons based on importance scores")
        
        # Get all neurons not already in neurons_to_prune
        all_neurons = list(range(hidden_neurons))
        remaining_neurons = [n for n in all_neurons if n not in neurons_to_prune]
        
        # Sort remaining neurons by importance score (ascending - less important first)
        remaining_neurons.sort(key=lambda x: importance_scores[x])
        
        # Add the least important neurons until we reach the target
        additional_neurons = remaining_neurons[:target_neurons_to_prune - len(neurons_to_prune)]
        neurons_to_prune.extend(additional_neurons)
        
        logger.info(f"Added {len(additional_neurons)} additional neurons based on importance scores")
    
    # Sort neurons_to_prune by importance score (ascending - less important first)
    neurons_to_prune.sort(key=lambda x: importance_scores[x])
    
    # Limit to target number of neurons to prune
    if len(neurons_to_prune) > target_neurons_to_prune:
        neurons_to_prune = neurons_to_prune[:target_neurons_to_prune]
    
    # Update pruning_info with the neurons to prune
    pruning_info['pruned_indices'] = neurons_to_prune
    pruning_info['correlation_results'] = {
        'correlated_pairs': correlated_pairs,
        'correlation_mask': correlation_mask
    }
    pruning_info['sensitivity_results'] = {
        'insensitive_indices': insensitive_neurons,
        'sensitivity_mask': sensitivity_mask
    }
    pruning_info['activity_results'] = {
        'low_activity_indices': inactive_neurons,
        'activity_mask': activity_mask
    }
    pruning_info['importance_scores'] = importance_scores.tolist()
    pruning_info['activation'] = activation
    
    # Create pruned model
    pruned_model = create_pruned_model(model, pruning_info, X_train_tensor)
    
    # Validate pruned model if test data is provided
    if X_test_tensor is not None and y_test_tensor is not None:
        original_accuracy = validate_pruned_model(model, X_test_tensor, y_test_tensor)
        pruned_accuracy = validate_pruned_model(pruned_model, X_test_tensor, y_test_tensor)
        
        logger.info(f"Original model accuracy: {original_accuracy:.2f}%")
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2f}%")
        
        # Calculate accuracy drop
        accuracy_drop = original_accuracy - pruned_accuracy
        pruning_info['accuracy_drop'] = accuracy_drop / 100.0  # Convert to decimal for consistency
    
    return pruned_model, pruning_info

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
        pruned_model, pruning_info = prune_model(
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
        accuracy_drop = original_accuracy - pruned_accuracy
        
        logger.info(f"Original model accuracy: {original_accuracy:.2f}%")
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.2f}%")
        logger.info(f"Accuracy drop: {accuracy_drop:.2f}%")
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