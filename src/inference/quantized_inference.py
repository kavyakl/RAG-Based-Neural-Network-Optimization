import os
import json
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import f1_score
import torch.nn.functional as F

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel
from ..utils.data_handling import load_and_preprocess_data

# Configure logging
logger = logging.getLogger(__name__)

def load_quantized_model(model_path: str, info_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a quantized model and its metadata.
    
    Args:
        model_path (str): Path to the quantized model weights
        info_path (str): Path to the model info file
        
    Returns:
        tuple: (loaded_model, model_info)
    """
    # Load model info
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    # Get model dimensions (handle both num_features and input_features)
    num_features = model_info.get('num_features') or model_info.get('input_features')
    
    # Get the correct number of hidden neurons (handle both original and pruned models)
    hidden_neurons = model_info.get('pruned_hidden_neurons') or model_info.get('hidden_neurons')
    activation = model_info.get('activation', 'relu').lower()  # Convert to lowercase
    
    if num_features is None or hidden_neurons is None:
        raise ValueError(f"Model info missing required fields. Found: {model_info.keys()}")
    
    # Load quantized weights first to check output dimensions
    saved_weights = torch.load(model_path)
    
    # Determine number of output neurons from the output weights
    if 'output.weight' in saved_weights:
        output_neurons = saved_weights['output.weight'].shape[0]
        logger.info(f"Determined output neurons ({output_neurons}) from saved weights")
    else:
        # Fall back to info if weights don't provide the information
        num_classes = model_info.get('num_classes', 2)
        output_neurons = num_classes if num_classes > 2 else 1
        logger.info(f"Using default output neurons ({output_neurons}) based on num_classes")
    
    # Create model instance with correct output shape
    if output_neurons > 1:
        model = CategoricalModel(
            input_size=num_features,
            hidden_size=hidden_neurons,
            output_size=output_neurons,
            activation=activation
        )
    else:
        model = BinaryModel(
            input_size=num_features,
            hidden_size=hidden_neurons,
            activation=activation
        )
    
    # Load quantized weights
    model.load_state_dict(saved_weights)
    model.eval()
    
    return model, model_info

def run_quantized_inference(
    data_path: str,
    model_path: str,
    info_path: str,
    output_dir: str,
    dataset_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Run inference using a quantized model.
    
    Args:
        data_path (str): Path to the input data
        model_path (str): Path to the quantized model weights
        info_path (str): Path to the model info file
        output_dir (str): Directory to save inference results
        dataset_name (str, optional): Name of the dataset
        
    Returns:
        dict: Inference metrics
    """
    # Load model and info
    model, model_info = load_quantized_model(model_path, info_path)
    
    # Load and preprocess data
    X_train, _, y_train, _, num_features, num_classes = load_and_preprocess_data(
        data_path,
        test_size=0.00001,  # Use a tiny test size instead of 0
        random_state=42
    )
    
    # Use the training data for inference
    X = X_train
    y = y_train
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    
    # Run inference
    with torch.no_grad():
        # Check if the model is multi-class (has more than 1 output neuron)
        is_multiclass = model.output.out_features > 1
        
        outputs, _ = model(X_tensor)
        
        if is_multiclass:
            predictions = torch.argmax(outputs, dim=1).numpy()
            probabilities = outputs.numpy()  # Outputs are already softmax probabilities
        else:
            predictions = (outputs.squeeze() > 0.5).numpy().astype(int)
            probabilities = outputs.squeeze().numpy()
    
    # Calculate accuracy
    accuracy = (predictions == y).mean() * 100
    
    # Calculate F1 score
    if is_multiclass:
        f1 = f1_score(y, predictions, average='weighted') * 100
    else:
        f1 = f1_score(y, predictions) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': predictions
    })
    
    # Add probability columns
    if is_multiclass:
        for i in range(model.output.out_features):
            results_df[f'Probability_Class_{i}'] = probabilities[:, i]
    else:
        results_df['Probability'] = probabilities
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_type = 'quantized'
    dataset_name = dataset_name or model_info.get('dataset_name', 'unknown')
    results_path = os.path.join(
        output_dir,
        f"{dataset_name}_{model_type}_inference_results.csv"
    )
    results_df.to_csv(results_path, index=False)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'num_samples': len(X),
        'model_type': model_type,
        'hidden_neurons': model.hidden.out_features,  # Get directly from the model
        'dataset_name': dataset_name,
        'results_path': results_path
    }
    
    metrics_path = os.path.join(
        output_dir,
        f"{dataset_name}_{model_type}_inference_metrics.json"
    )
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model Performance:")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"F1 Score: {f1:.2f}%")
    
    return metrics

def compare_quantized_models(
    data_path: str,
    original_quantized_path: str,
    original_info_path: str,
    pruned_quantized_path: str,
    pruned_info_path: str,
    output_dir: str,
    dataset_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare inference results between quantized original and pruned models.
    
    Args:
        data_path (str): Path to the input data
        original_quantized_path (str): Path to the quantized original model
        original_info_path (str): Path to the original model info
        pruned_quantized_path (str): Path to the quantized pruned model
        pruned_info_path (str): Path to the pruned model info
        output_dir (str): Directory to save comparison results
        dataset_name (str, optional): Name of the dataset
        
    Returns:
        dict: Comparison metrics
    """
    # Run inference on both models
    original_metrics = run_quantized_inference(
        data_path, original_quantized_path, original_info_path, output_dir, dataset_name
    )
    pruned_metrics = run_quantized_inference(
        data_path, pruned_quantized_path, pruned_info_path, output_dir, dataset_name
    )
    
    # Print accuracies and hidden neurons
    logger.info(f"Quantized original model - Hidden neurons: {original_metrics['hidden_neurons']}, Test accuracy: {original_metrics['accuracy']:.2f}%, F1: {original_metrics['f1_score']:.2f}%")
    logger.info(f"Quantized pruned model - Hidden neurons: {pruned_metrics['hidden_neurons']}, Test accuracy: {pruned_metrics['accuracy']:.2f}%, F1: {pruned_metrics['f1_score']:.2f}%")
    
    # Calculate differences
    accuracy_diff = pruned_metrics['accuracy'] - original_metrics['accuracy']
    f1_diff = pruned_metrics['f1_score'] - original_metrics['f1_score']
    neuron_diff = original_metrics['hidden_neurons'] - pruned_metrics['hidden_neurons']
    
    logger.info(f"Differences:")
    logger.info(f"Accuracy: {accuracy_diff:+.2f}%")
    logger.info(f"F1 Score: {f1_diff:+.2f}%")
    logger.info(f"Neurons removed: {neuron_diff}")
    
    return {
        'original': original_metrics,
        'pruned': pruned_metrics,
        'differences': {
            'accuracy': accuracy_diff,
            'f1_score': f1_diff,
            'neurons': neuron_diff
        }
    } 