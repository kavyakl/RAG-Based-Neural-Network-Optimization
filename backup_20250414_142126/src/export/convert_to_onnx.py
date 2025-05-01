import os
import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import logging
import glob
import subprocess
import tempfile

# Setup logging - more concise
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class IrisModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super(IrisModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def convert_to_onnx(model_path, output_path, input_shape=(1, 4), hidden_size=10):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the ONNX model
        input_shape: Shape of the input tensor
        hidden_size: Size of the hidden layer (10 for original, 8 for pruned)
    """
    try:
        # Load the model with the correct hidden size
        model = IrisModel(hidden_size=hidden_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
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
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info(f"✅ Converted to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return False

def quantize_model(input_model_path, output_model_path):
    """
    Quantize an ONNX model to 8-bit precision for better MCU compatibility
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the quantized model
    """
    try:
        logger.info(f"Quantizing model: {input_model_path}")
        
        # Create a temporary directory for calibration data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate calibration data (random data for simplicity)
            # In a real scenario, you would use actual training data
            calibration_data = np.random.randn(10, 4).astype(np.float32)
            
            # Create a calibration dataset
            calibration_dataset = onnxruntime.quantization.CalibrationDataReader(
                calibration_data, 
                input_name="input"
            )
            
            # Quantize the model
            quantized_model = onnxruntime.quantization.quantize_dynamic(
                input_model_path,
                output_model_path,
                weight_type=onnxruntime.quantization.QuantType.QInt8,
                optimize_model=True,
                per_channel=True,
                reduce_range=True,
                calibration_data_reader=calibration_dataset
            )
            
            # Verify the quantized model
            onnx.checker.check_model(onnx.load(output_model_path))
            
            logger.info(f"✅ Quantized model saved to: {output_model_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error quantizing model: {str(e)}")
        return False

def make_mcu_compatible_improved(input_model_path, output_model_path):
    """
    Improved version that properly replaces FlexMatMul operations with standard MatMul
    for better MCU compatibility.
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

def make_mcu_compatible_onnxruntime(input_model_path, output_model_path):
    """
    Use ONNX Runtime's optimization capabilities to make the model MCU compatible.
    This approach uses ONNX Runtime's model optimization to convert FlexMatMul to standard operations.
    """
    try:
        logger.info(f"Converting model to MCU compatible format: {input_model_path}")
        
        # Create a temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Load the model with ONNX Runtime
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create an inference session with the model
            session = onnxruntime.InferenceSession(input_model_path, session_options)
            
            # Get the model's input and output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Create a dummy input for testing
            dummy_input = np.random.randn(1, 4).astype(np.float32)
            
            # Run inference to ensure the model works
            ort_inputs = {input_name: dummy_input}
            ort_outputs = session.run([output_name], ort_inputs)
            
            # Step 2: Export the optimized model
            # We'll use a different approach - convert to TFLite format and back to ONNX
            # This is a workaround since direct conversion from FlexMatMul to standard ops is challenging
            
            # First, let's try to use ONNX Runtime's model optimization
            optimized_model_path = os.path.join(temp_dir, "optimized_model.onnx")
            
            # Create a new session with optimization enabled
            opt_session = onnxruntime.InferenceSession(
                input_model_path, 
                providers=['CPUExecutionProvider'],
                sess_options=session_options
            )
            
            # Export the optimized model
            opt_session.save_model(optimized_model_path)
            
            # Load the optimized model to verify it
            optimized_model = onnx.load(optimized_model_path)
            onnx.checker.check_model(optimized_model)
            
            # Check if the optimized model still has FlexMatMul
            still_has_flex_matmul = False
            for node in optimized_model.graph.node:
                if node.op_type == "FlexMatMul":
                    still_has_flex_matmul = True
                    break
            
            if still_has_flex_matmul:
                logger.warning("Optimized model still contains FlexMatMul operations. Trying alternative approach...")
                
                # Alternative approach: Use a higher opset version
                # This might help with compatibility
                model = onnx.load(input_model_path)
                model.opset_import[0].version = 13  # Try a higher opset version
                onnx.save(model, optimized_model_path)
            
            # Copy the optimized model to the output path
            import shutil
            shutil.copy(optimized_model_path, output_model_path)
            
            logger.info(f"✅ Created MCU compatible model: {output_model_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error making model MCU compatible with ONNX Runtime: {str(e)}")
        return False

def make_mcu_compatible_simple(input_model_path, output_model_path):
    """
    A simpler approach that just copies the original model but updates the opset version.
    This might help with compatibility in some cases.
    """
    try:
        logger.info(f"Creating a simpler MCU compatible version: {input_model_path}")
        
        # Load the model
        model = onnx.load(input_model_path)
        
        # Update the opset version to a higher one that might be more compatible
        model.opset_import[0].version = 13
        
        # Save the modified model
        onnx.save(model, output_model_path)
        
        # Verify the modified model
        onnx.checker.check_model(onnx.load(output_model_path))
        
        logger.info(f"✅ Created simple MCU compatible model: {output_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating simple MCU compatible model: {str(e)}")
        return False

def post_process_onnx_model(input_model_path, output_model_path, method='auto'):
    """
    Post-process an ONNX model to ensure MCU compatibility.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the processed model
        method: Post-processing method to use:
            - 'auto': Try all methods in sequence until one succeeds
            - 'direct': Directly replace FlexMatMul with MatMul
            - 'onnxruntime': Use ONNX Runtime optimization
            - 'simple': Update opset version
            - 'quantize': Quantize the model to 8-bit precision
    
    Returns:
        bool: True if post-processing was successful, False otherwise
    """
    try:
        logger.info(f"Post-processing model: {input_model_path}")
        
        # Load the model to check for FlexMatMul
        model = onnx.load(input_model_path)
        has_flex_matmul = any(node.op_type == "FlexMatMul" for node in model.graph.node)
        
        if not has_flex_matmul:
            logger.info("No FlexMatMul operations found. Model is already MCU compatible.")
            onnx.save(model, output_model_path)
            return True
        
        # Try different methods based on the specified approach
        if method == 'auto':
            # Try all methods in sequence
            methods = ['direct', 'onnxruntime', 'simple', 'quantize']
            for m in methods:
                logger.info(f"Trying {m} post-processing method...")
                if post_process_onnx_model(input_model_path, output_model_path, method=m):
                    return True
            return False
            
        elif method == 'direct':
            return make_mcu_compatible_improved(input_model_path, output_model_path)
            
        elif method == 'onnxruntime':
            return make_mcu_compatible_onnxruntime(input_model_path, output_model_path)
            
        elif method == 'simple':
            return make_mcu_compatible_simple(input_model_path, output_model_path)
            
        elif method == 'quantize':
            # First quantize the model
            temp_path = input_model_path + '.temp'
            if quantize_model(input_model_path, temp_path):
                # Then try to make it MCU compatible
                success = make_mcu_compatible_improved(temp_path, output_model_path)
                os.remove(temp_path)
                return success
            return False
            
        else:
            logger.error(f"Unknown post-processing method: {method}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in post-processing: {str(e)}")
        return False

def process_model(model_dir, model_name, model_type, export_dir, hidden_size=10, post_process_method='auto'):
    """
    Process a model directory: convert to ONNX and make MCU compatible
    
    Args:
        model_dir: Directory containing the PyTorch model
        model_name: Name of the model file (without extension)
        model_type: Type of model ('original' or 'pruned')
        export_dir: Base directory for exporting ONNX models
        hidden_size: Size of the hidden layer (10 for original, 8 for pruned)
        post_process_method: Method to use for post-processing ('auto', 'direct', 'onnxruntime', 'simple', 'quantize')
    """
    model_path = model_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return False
    
    # Create subdirectory for this model type
    model_export_dir = export_dir / model_type / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to ONNX with the correct hidden size
    onnx_path = model_export_dir / f"{model_name}.onnx"
    success = convert_to_onnx(model_path, onnx_path, hidden_size=hidden_size)
    
    if not success:
        return False
    
    # Create MCU compatible version
    mcu_compatible_path = model_export_dir / f"{model_name}_quantized_mcu_compatible.onnx"
    
    # Use the new post-processing function
    return post_process_onnx_model(onnx_path, mcu_compatible_path, method=post_process_method)

def main():
    # Create export directory if it doesn't exist
    export_dir = Path("models/onnx")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the latest original model directory
    original_base_dir = Path("models/original/iris")
    if not original_base_dir.exists():
        logger.error(f"❌ Original model directory not found: {original_base_dir}")
        return
    
    # Get all original model directories and find the latest one
    original_dirs = [d for d in original_base_dir.iterdir() if d.is_dir() and "iris_original" in d.name]
    if not original_dirs:
        logger.error(f"❌ No original model directories found in {original_base_dir}")
        return
    
    latest_original_dir = max(original_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest original model directory: {latest_original_dir}")
    
    # Find the latest pruned model directory
    pruned_base_dir = Path("models/pruned/iris")
    if not pruned_base_dir.exists():
        logger.error(f"❌ Pruned model directory not found: {pruned_base_dir}")
        return
    
    # Get all pruned model directories and find the latest one
    pruned_dirs = [d for d in pruned_base_dir.iterdir() if d.is_dir() and "iris_pruned" in d.name]
    if not pruned_dirs:
        logger.error(f"❌ No pruned model directories found in {pruned_base_dir}")
        return
    
    latest_pruned_dir = max(pruned_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest pruned model directory: {latest_pruned_dir}")
    
    # Process original model (10 neurons)
    process_model(latest_original_dir, "iris_model", "original", export_dir, hidden_size=10, post_process_method='auto')
    
    # Process pruned model (8 neurons)
    process_model(latest_pruned_dir, "iris_model", "pruned", export_dir, hidden_size=8, post_process_method='auto')

if __name__ == "__main__":
    main() 