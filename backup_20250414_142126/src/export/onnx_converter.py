import os
import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil
from src.utils.data_handling import load_saved_dataset, prepare_data_for_training
from src.utils.metrics import calculate_accuracy
from typing import Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

class NeuralNetworkModel(nn.Module):
    """Generic neural network model for ONNX conversion"""
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super(NeuralNetworkModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        
        # Set activation function based on config
        activation = activation.lower()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        logger.info(f"Initialized model with {activation} activation function")
        
    def forward(self, x):
        # Hidden layer with activation
        x = self.activation(self.hidden(x))
        
        # Output layer
        x = self.output(x)
        
        # Apply appropriate activation based on output size
        if self.output_size == 1:  # Binary classification
            x = torch.sigmoid(x)
        else:  # Multi-class classification
            x = torch.softmax(x, dim=1)
        return x

def convert_to_onnx(model_path, onnx_path, input_shape=(1, 4), hidden_size=32, output_size=3, activation='relu', opset_version=11):
    """
    Convert PyTorch model to ONNX format with improved precision
    
    Args:
        model_path: Path to the PyTorch model
        onnx_path: Path to save the ONNX model
        input_shape: Input shape tuple (batch_size, features)
        hidden_size: Number of neurons in hidden layer
        output_size: Number of output classes
        activation: Activation function to use
        opset_version: ONNX opset version
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Inspect the model state dict to determine the correct architecture
        model_info = inspect_model_state_dict(model_path)
        if model_info is None:
            logger.error("Failed to inspect model state dict")
            return False
        
        # Use the architecture information from the state dict
        actual_input_size = model_info['input_size']
        actual_hidden_size = model_info['hidden_size']
        actual_output_size = model_info['output_size']
        actual_activation = model_info['activation']
        
        logger.info(f"Using architecture from state dict:")
        logger.info(f"  - Input size: {actual_input_size}")
        logger.info(f"  - Hidden size: {actual_hidden_size}")
        logger.info(f"  - Output size: {actual_output_size}")
        logger.info(f"  - Activation: {actual_activation}")
        
        # Load model with correct parameters
        model = NeuralNetworkModel(
            input_size=actual_input_size,
            hidden_size=actual_hidden_size,
            output_size=actual_output_size,
            activation=actual_activation
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn((1, actual_input_size), dtype=torch.float32)
        
        # Export with improved settings
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
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
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare PyTorch and ONNX outputs
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()
        
        # Check if outputs match within tolerance
        np.testing.assert_allclose(ort_outputs[0], torch_output, rtol=1e-03, atol=1e-05)
        
        logger.info(f"Successfully converted model to ONNX format: {onnx_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        return False

def make_mcu_compatible(input_model_path, output_model_path):
    """
    Make an ONNX model MCU compatible by replacing FlexMatMul operations with standard MatMul
    """
    try:
        # Load the model
        logger.info(f"Loading model: {input_model_path}")
        model = onnx.load(input_model_path)
        
        # Check if the model contains FlexMatMul operations
        has_flex_matmul = False
        for node in model.graph.node:
            if node.op_type == "FlexMatMul":
                has_flex_matmul = True
                break
        
        if not has_flex_matmul:
            logger.info("No FlexMatMul operations found. Model is already MCU compatible.")
            # Copy the model to the output path
            onnx.save(model, output_model_path)
            return True
        
        logger.info("Found FlexMatMul operations. Replacing with standard MatMul...")
        
        # Create a new graph with replaced nodes
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == "FlexMatMul":
                # Create a new MatMul node
                new_node = onnx.helper.make_node(
                    "MatMul",
                    inputs=node.input,
                    outputs=node.output,
                    name=node.name + "_matmul"
                )
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)
        
        # Update the graph with the new nodes
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        
        # Save the modified model
        onnx.save(model, output_model_path)
        
        # Verify the modified model
        onnx.checker.check_model(onnx.load(output_model_path))
        
        logger.info(f"✅ Created MCU compatible model: {output_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error making model MCU compatible: {str(e)}")
        return False

def validate_onnx_model(onnx_path, test_data_path):
    """Validate the ONNX model using test data."""
    logger.info(f"Loading model: {onnx_path}")
    session = onnxruntime.InferenceSession(onnx_path)
    
    logger.info(f"Found test data at: {test_data_path}")
    test_data = np.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Run inference
    input_name = session.get_inputs()[0].name
    raw_predictions = session.run(None, {input_name: X_test.astype(np.float32)})[0]
    
    # For binary classification (output shape is [N, 1])
    if raw_predictions.shape[1] == 1:
        # The ONNX model already includes sigmoid, so we just need to threshold
        binary_predictions = (raw_predictions > 0.5).astype(int)
        accuracy = np.mean(binary_predictions.squeeze() == y_test.squeeze())
    # For multi-class classification (output shape is [N, num_classes])
    else:
        predicted_classes = np.argmax(raw_predictions, axis=1)
        accuracy = np.mean(predicted_classes == y_test)
    
    logger.info(f"Validation accuracy: {accuracy * 100:.2f}%")
    return accuracy

def process_model(model_dir, model_name, model_type, export_dir, input_features, hidden_size, output_size, activation='relu'):
    """
    Process a model directory: convert to ONNX and make MCU compatible
    
    Args:
        model_dir: Directory containing the model
        model_name: Name of the model
        model_type: Type of model (original/pruned)
        export_dir: Directory to export ONNX model
        input_features: Number of input features
        hidden_size: Number of neurons in hidden layer
        output_size: Number of output classes
        activation: Activation function to use (must match training config)
    """
    # Get the latest model directory
    model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not model_dirs:
        logger.error(f"❌ No model directories found in: {model_dir}")
        return None
    
    latest_model_dir = max(model_dirs, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model_dir, f"{model_name}.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return None
    
    # Create subdirectory for this model type
    model_export_dir = os.path.join(export_dir, model_type, f"{model_name}")
    os.makedirs(model_export_dir, exist_ok=True)
    
    # Log model configuration
    logger.info(f"Converting {model_type} model with configuration:")
    logger.info(f"  - Input features: {input_features}")
    logger.info(f"  - Hidden size: {hidden_size}")
    logger.info(f"  - Output size: {output_size}")
    logger.info(f"  - Activation: {activation}")
    
    # Convert to ONNX with all required parameters
    onnx_path = os.path.join(model_export_dir, f"{model_name}.onnx")
    success = convert_to_onnx(
        model_path=model_path,
        onnx_path=onnx_path,
        input_shape=(1, input_features),
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
        opset_version=11
    )
    
    if not success:
        logger.error(f"❌ Failed to convert {model_type} model to ONNX format")
        return None
    
    # Create MCU compatible version
    mcu_compatible_path = os.path.join(model_export_dir, f"{model_name}_mcu_compatible.onnx")
    mcu_success = make_mcu_compatible(onnx_path, mcu_compatible_path)
    
    if not mcu_success:
        logger.warning("Failed to create MCU compatible model")
    
    # Extract dataset name from model_name (e.g., "breast_cancer_model" -> "breast_cancer")
    dataset_name = model_name.replace("_model", "")
    
    # Try multiple possible locations for test data
    test_data_paths = [
        os.path.join(model_dir, latest_model_dir, f"{dataset_name}_test_data.npz"),  # In model directory
        os.path.join("data", f"{dataset_name}_test_data.npz"),  # In data directory
        os.path.join("models", "original", dataset_name, f"{dataset_name}_test_data.npz"),  # In models/original
        os.path.join("models", "pruned", dataset_name, f"{dataset_name}_test_data.npz")  # In models/pruned
    ]
    
    test_data_path = None
    for path in test_data_paths:
        if os.path.exists(path):
            test_data_path = path
            logger.info(f"Found test data at: {test_data_path}")
            break
    
    if test_data_path is None:
        logger.warning(f"❌ Test data not found in any of the expected locations for dataset: {dataset_name}")
        return {
            'onnx_path': onnx_path,
            'mcu_compatible_path': mcu_compatible_path if mcu_success else None,
            'validation_accuracy': None,
            'mcu_validation_accuracy': None,
            'test_data_path': None
        }
    
    validation_accuracy = validate_onnx_model(onnx_path, test_data_path)
    
    # Validate MCU compatible model if it exists
    mcu_validation_accuracy = None
    if mcu_success:
        mcu_validation_accuracy = validate_onnx_model(mcu_compatible_path, test_data_path)
    
    return {
        'onnx_path': onnx_path,
        'mcu_compatible_path': mcu_compatible_path if mcu_success else None,
        'validation_accuracy': validation_accuracy,
        'mcu_validation_accuracy': mcu_validation_accuracy,
        'test_data_path': test_data_path
    }

def export_models_to_onnx(config: Dict[str, Any]) -> None:
    """
    Export PyTorch models to ONNX format.
    
    Args:
        config: Configuration dictionary containing model paths and export settings
    """
    logger.info("Starting ONNX export...")
    
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
    
    # Verify paths exist
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(f"Original model not found at {original_model_path}")
    if not os.path.exists(pruned_model_path):
        raise FileNotFoundError(f"Pruned model not found at {pruned_model_path}")
    if not os.path.exists(original_model_info_path):
        raise FileNotFoundError(f"Original model info not found at {original_model_info_path}")
    if not os.path.exists(pruned_model_info_path):
        raise FileNotFoundError(f"Pruned model info not found at {pruned_model_info_path}")
    
    # Create directory structure for ONNX models
    original_onnx_dir = os.path.join("models", "original", activation, f"{dataset_name}_model")
    pruned_onnx_dir = os.path.join("models", "pruned", activation, f"{dataset_name}_model")
    
    # Create directories if they don't exist
    os.makedirs(original_onnx_dir, exist_ok=True)
    os.makedirs(pruned_onnx_dir, exist_ok=True)
    
    # Define ONNX model paths
    original_onnx_path = os.path.join(original_onnx_dir, f"{dataset_name}_model.onnx")
    pruned_onnx_path = os.path.join(pruned_onnx_dir, f"{dataset_name}_model.onnx")
    
    # Load models
    original_model = load_model(original_model_path)
    pruned_model = load_model(pruned_model_path)
    
    # Export original model to ONNX
    logger.info(f"Exporting original model to {original_onnx_path}")
    export_model_to_onnx(original_model, original_onnx_path)
    
    # Export pruned model to ONNX
    logger.info(f"Exporting pruned model to {pruned_onnx_path}")
    export_model_to_onnx(pruned_model, pruned_onnx_path)
    
    logger.info("ONNX export completed successfully")
    
    # Update config with ONNX paths
    config['export']['original_onnx_path'] = original_onnx_path
    config['export']['pruned_onnx_path'] = pruned_onnx_path

def inspect_model_state_dict(model_path):
    """
    Inspect the model state dict to determine the correct architecture
    
    Args:
        model_path: Path to the PyTorch model
        
    Returns:
        dict: Model architecture information
    """
    try:
        state_dict = torch.load(model_path)
        
        # Extract architecture information
        input_size = state_dict['hidden.weight'].shape[1]
        hidden_size = state_dict['hidden.weight'].shape[0]
        output_size = state_dict['output.weight'].shape[0]
        
        # Determine if this is a binary classification model
        is_binary = output_size == 1
        
        # Determine activation function (if possible)
        activation = 'relu'  # Default
        
        logger.info(f"Model architecture:")
        logger.info(f"  - Input size: {input_size}")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Output size: {output_size}")
        logger.info(f"  - Binary classification: {is_binary}")
        
        return {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'is_binary': is_binary,
            'activation': activation
        }
    except Exception as e:
        logger.error(f"Error inspecting model state dict: {str(e)}")
        return None

def is_mcu_compatible(model_path):
    """
    Check if an ONNX model is compatible with MCU deployment.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        bool: True if model is MCU compatible, False otherwise
    """
    try:
        # Load the model
        model = onnx.load(model_path)
        
        # Check for unsupported operations
        unsupported_ops = set()
        for node in model.graph.node:
            if node.op_type in ['FlexMatMul']:
                unsupported_ops.add(node.op_type)
        
        if unsupported_ops:
            logger.info(f"Model contains unsupported operations for MCU: {unsupported_ops}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking MCU compatibility: {str(e)}")
        return False

def convert_to_mcu_compatible(input_model_path, output_model_path):
    """
    Convert an ONNX model to be MCU compatible by replacing unsupported operations.
    
    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save MCU compatible model
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Load the model
        model = onnx.load(input_model_path)
        
        # Replace unsupported operations
        for node in model.graph.node:
            if node.op_type == 'FlexMatMul':
                # Replace FlexMatMul with standard MatMul
                node.op_type = 'MatMul'
        
        # Save the modified model
        onnx.save(model, output_model_path)
        
        # Verify the modified model
        onnx.checker.check_model(onnx.load(output_model_path))
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting to MCU compatible: {str(e)}")
        return False

def load_model(model_path: str) -> nn.Module:
    """
    Load a PyTorch model from a file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        nn.Module: Loaded PyTorch model
    """
    try:
        # Load the model state dict
        state_dict = torch.load(model_path)
        
        # Extract model architecture from state dict
        input_size = state_dict['hidden.weight'].shape[1]
        hidden_size = state_dict['hidden.weight'].shape[0]
        output_size = state_dict['output.weight'].shape[0]
        
        # Create a new model with the same architecture
        model = NeuralNetworkModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation='relu'  # Default to ReLU, will be overridden by state dict
        )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def export_model_to_onnx(model: nn.Module, onnx_path: str) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        onnx_path: Path to save the ONNX model
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(1, model.hidden.weight.shape[1])
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Successfully exported model to {onnx_path}")
        
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {str(e)}")
        raise

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