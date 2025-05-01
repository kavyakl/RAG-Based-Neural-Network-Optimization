import torch
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import json

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple) and all(isinstance(i, np.integer) for i in obj):
            return tuple(int(i) for i in obj)
        return super(NumpyEncoder, self).default(obj)

def calculate_accuracy(outputs, targets, num_classes):
    """
    Calculate the accuracy of model predictions.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        float: Accuracy as a percentage
    """
    with torch.no_grad():
        if num_classes > 2:  # Multiclass classification
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets.flatten()).sum().item()
            total = targets.size(0)
        else:  # Binary classification
            # Ensure outputs and targets have the same shape
            outputs = outputs.view(-1, 1) if len(outputs.shape) == 1 else outputs
            targets = targets.view(-1, 1) if len(targets.shape) == 1 else targets
            predicted = (outputs > 0.5).float()  # Convert to float for comparison
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
        
        accuracy = (correct / total) * 100
    
    return accuracy

def save_model_metrics(metrics_dir, dataset_name, model_type, metrics_dict):
    """
    Save model evaluation metrics to a file.
    
    Args:
        metrics_dir (str): Directory to save metrics
        dataset_name (str): Name of the dataset
        model_type (str): Type of model (original, pruned, etc.)
        metrics_dict (dict): Dictionary of metrics to save
    """
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    filename = f"{dataset_name}_{model_type}_metrics_{timestamp}.json"
    filepath = os.path.join(metrics_dir, filename)
    
    # Add timestamp to metrics
    metrics_dict['timestamp'] = timestamp
    metrics_dict['dataset'] = dataset_name
    metrics_dict['model_type'] = model_type
    
    # Save metrics to JSON file
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4, cls=NumpyEncoder)
    
    return filepath

def add_result_to_csv(results_file, result_row, headers=None):
    """
    Add a result row to a CSV file.
    
    Args:
        results_file (str): Path to the CSV file
        result_row (list): Row of results to add
        headers (list, optional): Column headers if file is new
    """
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists and headers:
            writer.writerow(headers)
        
        # Write result row
        writer.writerow(result_row)

def compare_models(original_metrics, pruned_metrics):
    """
    Compare original and pruned model performance.
    
    Args:
        original_metrics (dict): Metrics of the original model
        pruned_metrics (dict): Metrics of the pruned model
        
    Returns:
        dict: Comparison metrics
    """
    comparison = {
        'accuracy_diff': pruned_metrics['test_accuracy'] - original_metrics['test_accuracy'],
        'size_reduction_percent': (1 - pruned_metrics['model_size'] / original_metrics['model_size']) * 100 if 'model_size' in pruned_metrics and 'model_size' in original_metrics else None,
        'neurons_pruned': pruned_metrics.get('neurons_pruned', 0),
        'neurons_approximated': pruned_metrics.get('neurons_approximated', 0),
    }
    
    return comparison 