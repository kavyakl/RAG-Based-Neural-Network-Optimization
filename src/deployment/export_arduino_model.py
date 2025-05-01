import os
import torch
import numpy as np
import json
import logging
import yaml
from typing import Dict, List, Any, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel

# Configure logging
logger = logging.getLogger(__name__)

def get_dataset_num_classes(dataset_name: str) -> int:
    """
    Get the number of classes for a dataset from its config file.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        int: Number of classes for the dataset
    """
    config_path = os.path.join('config', f'config_{dataset_name}.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['dataset']['num_classes']
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
        return defaults.get(dataset_name, 2)  # Default to binary classification if unknown

def export_model_to_c(
    model_path: str,
    output_dir: str,
    model_type: str,
    dataset_name: str,
    bit_width: int = 8
) -> bool:
    """
    Export a quantized model to C code for Arduino deployment.
    
    Args:
        model_path (str): Path to the quantized model weights
        output_dir (str): Directory to save the C code
        model_type (str): Type of model ('original' or 'pruned')
        dataset_name (str): Name of the dataset
        bit_width (int): Bit width for quantization
        
    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        # Load model info from the corresponding info file
        info_path = model_path.replace('.pth', '_info.json')
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create model instance with correct architecture
        num_features = model_info.get('num_features', model_info.get('input_features', 4))
        hidden_neurons = model_info.get('pruned_hidden_neurons') if model_type == 'pruned' else model_info.get('hidden_neurons', 10)
        # Get number of classes from config file
        num_classes = get_dataset_num_classes(dataset_name)
        activation = model_info.get('activation', 'relu')
        
        # Determine model type based on number of classes
        if num_classes == 2:
            model = BinaryModel(num_features, hidden_neurons, activation=activation)
        else:
            model = CategoricalModel(num_features, hidden_neurons, num_classes, activation=activation)
        
        # Load the quantized weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Get model architecture info
        model_arch = {
            'input_size': model.hidden.in_features,
            'hidden_size': model.hidden.out_features,
            'output_size': model.output.out_features,
            'activation': activation,
            'bit_width': bit_width
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Export model weights and architecture
        export_path = os.path.join(dataset_dir, f"{dataset_name}_{model_type}_model.h")
        with open(export_path, 'w') as f:
            f.write(f"// {dataset_name} {model_type} model weights\n")
            f.write(f"// Quantized to {bit_width} bits\n\n")
            
            # Write model architecture
            f.write("// Model Architecture\n")
            f.write(f"#define INPUT_SIZE {model_arch['input_size']}\n")
            f.write(f"#define HIDDEN_SIZE {model_arch['hidden_size']}\n")
            f.write(f"#define OUTPUT_SIZE {model_arch['output_size']}\n")
            f.write(f"#define BIT_WIDTH {model_arch['bit_width']}\n\n")
            
            # Write weights
            f.write("// Hidden layer weights\n")
            hidden_weights = model.hidden.weight.data.numpy()
            f.write("const float hidden_weights[] = {\n")
            for i in range(hidden_weights.shape[0]):
                f.write("    ")
                for j in range(hidden_weights.shape[1]):
                    f.write(f"{hidden_weights[i,j]:.6f}f, ")
                f.write("\n")
            f.write("};\n\n")
            
            # Write hidden layer bias
            f.write("// Hidden layer bias\n")
            hidden_bias = model.hidden.bias.data.numpy()
            f.write("const float hidden_bias[] = {\n")
            for i in range(hidden_bias.shape[0]):
                f.write(f"    {hidden_bias[i]:.6f}f,\n")
            f.write("};\n\n")
            
            # Write output layer weights
            f.write("// Output layer weights\n")
            output_weights = model.output.weight.data.numpy()
            f.write("const float output_weights[] = {\n")
            for i in range(output_weights.shape[0]):
                f.write("    ")
                for j in range(output_weights.shape[1]):
                    f.write(f"{output_weights[i,j]:.6f}f, ")
                f.write("\n")
            f.write("};\n\n")
            
            # Write output layer bias
            f.write("// Output layer bias\n")
            output_bias = model.output.bias.data.numpy()
            f.write("const float output_bias[] = {\n")
            for i in range(output_bias.shape[0]):
                f.write(f"    {output_bias[i]:.6f}f,\n")
            f.write("};\n")
        
        logger.info(f"Exported {model_type} model to {export_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting model to C: {str(e)}")
        return False

def export_test_data(
    data_path: str,
    output_dir: str,
    dataset_name: str,
    num_samples: int = 5
) -> str:
    """
    Export test samples for validation on the Arduino.
    
    Args:
        data_path (str): Path to the dataset
        output_dir (str): Directory to save the output files
        dataset_name (str): Name of the dataset
        num_samples (int): Number of test samples to export
        
    Returns:
        str: Path to the exported test data file
    """
    from ..utils.data_handling import load_and_preprocess_data
    
    # Create output directory if it doesn't exist
    model_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load a small subset of the data
    X_train, X_test, y_train, y_test, num_features, num_classes = load_and_preprocess_data(
        data_path,
        test_size=0.2,
        random_state=42
    )
    
    # Select a few samples for testing
    test_samples = X_test[:num_samples]
    test_labels = y_test[:num_samples]
    
    # Generate C header file for test data
    header_path = os.path.join(model_output_dir, f"{dataset_name}_test_data.h")
    
    with open(header_path, 'w') as f:
        # Write header
        f.write(f"/**\n")
        f.write(f" * Test data samples for {dataset_name} dataset\n")
        f.write(f" * Generated by Neural Network Optimization Pipeline\n")
        f.write(f" */\n\n")
        
        f.write(f"#ifndef {dataset_name.upper()}_TEST_DATA_H\n")
        f.write(f"#define {dataset_name.upper()}_TEST_DATA_H\n\n")
        
        # Write test samples
        f.write(f"#define TEST_NUM_SAMPLES {num_samples}\n")
        f.write(f"#define TEST_INPUT_SIZE {num_features}\n\n")
        
        # Write test input data
        f.write(f"const float test_samples[TEST_NUM_SAMPLES][TEST_INPUT_SIZE] = {{\n")
        for i, sample in enumerate(test_samples):
            f.write("    {")
            for j, val in enumerate(sample):
                f.write(f"{val}")
                if j < len(sample) - 1:
                    f.write(", ")
            f.write("}")
            if i < len(test_samples) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # Write expected outputs
        f.write(f"const int test_labels[TEST_NUM_SAMPLES] = {{\n    ")
        for i, label in enumerate(test_labels):
            f.write(f"{int(label)}")
            if i < len(test_labels) - 1:
                f.write(", ")
        f.write("\n};\n\n")
        
        # Close header guard
        f.write(f"#endif // {dataset_name.upper()}_TEST_DATA_H\n")
    
    logger.info(f"Exported test data to {header_path}")
    
    return header_path 