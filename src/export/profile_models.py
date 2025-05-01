import os
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from tabulate import tabulate
import onnx
import numpy as np
import time
import sys
from datetime import datetime
import glob

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Try importing Edge Impulse dependencies
EDGE_IMPULSE_AVAILABLE = False
try:
    from edgeimpulse.model import profile
    EDGE_IMPULSE_AVAILABLE = True
    logging.info("Edge Impulse SDK found. Profiling will be enabled.")
except ImportError:
    logging.warning("""
Edge Impulse SDK not found. Edge Impulse profiling will be disabled.
To enable Edge Impulse profiling, install the SDK:
pip install edgeimpulse edgeimpulse-api
""")

from src.utils.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_latest_model_dir(base_dir):
    """Get the latest model directory based on modification time."""
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory does not exist: {base_dir}")
        return None
        
    # Find all subdirectories
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        logger.warning(f"No subdirectories found in {base_dir}")
        return None
        
    # Get the most recently modified directory
    latest_dir = max(subdirs, key=os.path.getmtime)
    logger.info(f"Latest model directory: {latest_dir}")
    return latest_dir

def convert_to_native_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def analyze_model_size(model_path: str) -> dict:
    """
    Analyze ONNX model size and memory requirements for ESP8266.
    ESP8266 specs:
    - Flash memory: 4MB
    - RAM: 80KB
    - CPU: 80MHz/160MHz
    """
    model = onnx.load(model_path)
    
    # Calculate model size
    model_size = os.path.getsize(model_path)
    
    # Analyze weights and compute approximate RAM usage
    total_params = 0
    ram_usage = 0
    for tensor in model.graph.initializer:
        # Handle both static and dynamic dimensions
        shape = []
        for dim in tensor.dims:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                shape.append(dim.dim_value)
        
        num_params = int(np.prod(shape)) if shape else 1
        total_params += num_params
        
        # Calculate memory based on data type
        if tensor.data_type == onnx.TensorProto.FLOAT:
            bytes_per_param = 4  # float32
        elif tensor.data_type == onnx.TensorProto.DOUBLE:
            bytes_per_param = 8  # float64
        elif tensor.data_type in [onnx.TensorProto.INT32, onnx.TensorProto.UINT32]:
            bytes_per_param = 4  # int32/uint32
        elif tensor.data_type in [onnx.TensorProto.INT64, onnx.TensorProto.UINT64]:
            bytes_per_param = 8  # int64/uint64
        else:
            bytes_per_param = 4  # default to float32
            
        ram_usage += num_params * bytes_per_param
    
    # Add estimated runtime memory for activations (rough estimate: 20% of model size)
    runtime_memory = ram_usage * 0.2
    total_ram = ram_usage + runtime_memory
    
    # Estimate quantized model size (assuming 8-bit quantization)
    quantized_size = model_size * 0.25  # Rough estimate: 8-bit vs 32-bit
    
    return {
        'model_size_kb': float(model_size / 1024),
        'quantized_size_kb': float(quantized_size / 1024),
        'ram_usage_kb': float(total_ram / 1024),
        'total_parameters': int(total_params),
        'esp8266_compatible': {
            'flash': quantized_size < 4 * 1024 * 1024,  # 4MB flash limit
            'ram': total_ram < 80 * 1024,  # 80KB RAM limit
        }
    }

def profile_model(model_path: str, device: str, timeout_seconds: int = 300, api_key: str = None) -> Dict[str, Any]:
    """
    Profile a model on a specific device with timeout
    
    Args:
        model_path: Path to the ONNX model
        device: Target device for profiling
        timeout_seconds: Maximum time to wait for profiling results
        api_key: Edge Impulse API key
        
    Returns:
        Dict containing profiling results or None if profiling failed
    """
    if not EDGE_IMPULSE_AVAILABLE:
        logger.warning("Edge Impulse profiling is disabled. Skipping profiling.")
        return {
            'inference_time_ms': 0.0,
            'rom_size_bytes': 0,
            'ram_usage_bytes': 0,
            'mcu_supported': False,
            'mcu_support_error': 'Edge Impulse dependencies not available'
        }
        
    logger.info(f"\n{'='*50}")
    logger.info(f"Profiling model: {os.path.basename(model_path)} for {device}")
    logger.info(f"{'='*50}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return {
            'inference_time_ms': 0.0,
            'rom_size_bytes': 0,
            'ram_usage_bytes': 0,
            'mcu_supported': False,
            'mcu_support_error': f'Model file not found at: {model_path}'
        }
    
    try:
        # Get API key from environment
        if api_key is None:
            api_key = os.getenv("EDGE_IMPULSE_API_KEY")
            if not api_key:
                logger.error("Edge Impulse API key not found in environment variables. Please set EDGE_IMPULSE_API_KEY.")
                return {
                    'inference_time_ms': 0.0,
                    'rom_size_bytes': 0,
                    'ram_usage_bytes': 0,
                    'mcu_supported': False,
                    'mcu_support_error': 'API key not found'
                }
        
        # Start profiling
        start_time = time.time()
        result = profile(model=model_path, api_key=api_key, device=device)
        
        # Check if profiling took too long
        if isinstance(timeout_seconds, str):
            timeout_seconds = int(timeout_seconds)
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Profiling timed out after {timeout_seconds} seconds")
            return {
                'inference_time_ms': 0.0,
                'rom_size_bytes': 0,
                'ram_usage_bytes': 0,
                'mcu_supported': False,
                'mcu_support_error': f'Profiling timed out after {timeout_seconds} seconds'
            }
            
        # Extract the profile info from the result
        profile_info = result.model.profile_info.float32
        
        # Create results dictionary
        results = {
            'device': device,
            'inference_time_ms': profile_info.time_per_inference_ms,
            'rom_size_bytes': profile_info.memory.tflite.rom,
            'ram_usage_bytes': profile_info.memory.tflite.ram,
            'mcu_supported': profile_info.is_supported_on_mcu,
            'mcu_support_error': profile_info.mcu_support_error if not profile_info.is_supported_on_mcu else None
        }
        
        # Add device-specific details
        try:
            if device == "raspberry-pi-4":
                device_table = result.model.profile_info.table.mpu
                results.update({
                    'description': device_table.description,
                    'device_inference_time_ms': device_table.time_per_inference_ms,
                    'device_rom_size_bytes': device_table.rom,
                    'device_supported': device_table.supported
                })
            else:
                # For RP2040, check if mcu attribute exists
                if hasattr(result.model.profile_info.table, 'mcu'):
                    device_table = result.model.profile_info.table.mcu
                    results.update({
                        'description': device_table.description,
                        'device_inference_time_ms': device_table.time_per_inference_ms,
                        'device_rom_size_bytes': device_table.rom,
                        'device_supported': device_table.supported
                    })
        except AttributeError as e:
            logger.warning(f"Could not access device-specific details: {str(e)}")
        
        logger.info(f"\nProfiling Results for {device}:")
        logger.info(f"Inference time: {results['inference_time_ms']} ms")
        logger.info(f"ROM size: {results['rom_size_bytes']} bytes")
        logger.info(f"RAM usage: {results['ram_usage_bytes']} bytes")
        logger.info(f"MCU support: {'Supported' if results['mcu_supported'] else 'Not supported'}")
        
        if not results['mcu_supported'] and results['mcu_support_error']:
            logger.info(f"MCU support error: {results['mcu_support_error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'inference_time_ms': 0.0,
            'rom_size_bytes': 0,
            'ram_usage_bytes': 0,
            'mcu_supported': False,
            'mcu_support_error': str(e)
        }

def profile_single_model(model_path: str) -> Dict[str, Any]:
    """Profile a single model for Edge Impulse compatibility.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Dictionary containing profiling results
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Profiling model: {os.path.basename(model_path)}")
    logger.info(f"{'='*50}")
    
    # Get API key from environment
    api_key = os.getenv('EDGE_IMPULSE_API_KEY')
    if not api_key:
        logger.error("Edge Impulse API key not found in environment variables")
        return {
            'inference_time_ms': 0.0,
            'rom_size_bytes': 0,
            'ram_usage_bytes': 0,
            'mcu_supported': False,
            'mcu_support_error': 'API key not found'
        }
    
    # Analyze model size
    size_analysis = analyze_model_size(model_path)
    
    # Profile model on different devices
    devices = ["raspberry-pi-4", "raspberry-pi-rp2040"]
    device_results = {}
    
    for device in devices:
        try:
            logger.info(f"Profiling on {device}...")
            result = profile_model(model_path, device, api_key=api_key)
            if result:
                device_results[device] = result
            else:
                device_results[device] = {
                    'inference_time_ms': 0.0,
                    'rom_size_bytes': size_analysis['model_size_kb'] * 1024,
                    'ram_usage_bytes': size_analysis['ram_usage_kb'] * 1024,
                    'mcu_supported': False,
                    'mcu_support_error': 'Profiling failed'
                }
        except Exception as e:
            logger.error(f"Error profiling on {device}: {str(e)}")
            device_results[device] = {
                'inference_time_ms': 0.0,
                'rom_size_bytes': size_analysis['model_size_kb'] * 1024,
                'ram_usage_bytes': size_analysis['ram_usage_kb'] * 1024,
                'mcu_supported': False,
                'mcu_support_error': str(e)
            }
    
    return device_results

def profile_models(config):
    """Profile models for Edge Impulse compatibility"""
    logger.info("Starting model profiling for Edge Impulse compatibility...")
    
    # Get the latest model directories
    activation = config['model']['activation']
    dataset_name = config['dataset']['name']
    model_name = f"{dataset_name}_model"
    
    # Base directories - separate original and pruned
    original_base = os.path.join("models", "original", activation)
    pruned_base = os.path.join("models", "pruned", activation)
    
    # Model directories
    original_model_dir = os.path.join(original_base, model_name)
    pruned_model_dir = os.path.join(pruned_base, model_name)
    
    # Paths to ONNX models
    original_onnx_path = os.path.join(original_model_dir, f"{model_name}.onnx")
    pruned_onnx_path = os.path.join(pruned_model_dir, f"{model_name}.onnx")
    
    # Verify ONNX files exist
    if not os.path.exists(original_onnx_path):
        logger.error(f"Original ONNX model not found at {original_onnx_path}")
        return None
    if not os.path.exists(pruned_onnx_path):
        logger.error(f"Pruned ONNX model not found at {pruned_onnx_path}")
        return None
    
    # Profile the models
    logger.info(f"Profiling original model: {original_onnx_path}")
    original_profiling = profile_single_model(original_onnx_path)
    
    logger.info(f"Profiling pruned model: {pruned_onnx_path}")
    pruned_profiling = profile_single_model(pruned_onnx_path)
    
    # Save profiling results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", "profiling")
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, f"{dataset_name}_profiling_results_{timestamp}.json")
    
    results = {
        'original': original_profiling,
        'pruned': pruned_profiling,
        'timestamp': timestamp,
        'dataset': dataset_name,
        'activation': activation
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=convert_to_native_types)
    
    logger.info(f"Profiling results saved to {results_path}")
    
    return results

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile ONNX models')
    parser.add_argument('--config', type=str, default='config/config_iris.yaml',
                      help='Path to the configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    profile_models(config)

if __name__ == "__main__":
    main() 