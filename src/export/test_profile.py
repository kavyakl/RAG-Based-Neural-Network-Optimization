import os
import logging
import time
from dotenv import load_dotenv
from edgeimpulse.model import profile

# Setup logging with a simpler format
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Base path for models
BASE_PATH = "/Users/kavyakalyanam/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/Kavya_work/PHD_stuff/Problem_3/code/Experiment_6/neural_network_optimization_tagged_10_apr"

# Model paths for the two models we want to profile
ORIGINAL_MODEL_PATH = os.path.join(BASE_PATH, "models/onnx/original/iris_model/iris_model.onnx")
PRUNED_MODEL_PATH = os.path.join(BASE_PATH, "models/onnx/pruned/iris_model/iris_model.onnx")

# You can also hardcode your key if not using .env
API_KEY = os.getenv("EDGE_IMPULSE_API_KEY")
TARGET_DEVICES = ["raspberry-pi-4", "raspberry-pi-rp2040"]  # Both Raspberry Pi 4 and RP2040

def profile_model(model_path, device, timeout_seconds=300):  # Increased timeout to 5 minutes
    """Profile a model on a specific device with timeout"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Profiling model: {os.path.basename(model_path)} for {device}")
    logger.info(f"{'='*50}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return
    
    try:
        # Start profiling
        start_time = time.time()
        result = profile(model=model_path, api_key=API_KEY, device=device)
        
        # Check if profiling took too long
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Profiling timed out after {timeout_seconds} seconds")
            return
        
        # Extract the profile info from the result
        profile_info = result.model.profile_info.float32
        
        logger.info(f"\nProfiling Results for {device}:")
        logger.info(f"Inference time: {profile_info.time_per_inference_ms} ms")
        logger.info(f"ROM size: {profile_info.memory.tflite.rom} bytes")
        logger.info(f"RAM usage: {profile_info.memory.tflite.ram} bytes")
        logger.info(f"MCU support: {'Supported' if profile_info.is_supported_on_mcu else 'Not supported'}")
        
        if not profile_info.is_supported_on_mcu and profile_info.mcu_support_error:
            logger.info(f"MCU support error: {profile_info.mcu_support_error}")
            
        # Print device-specific table info
        try:
            if device == "raspberry-pi-4":
                device_table = result.model.profile_info.table.mpu
                logger.info(f"\nMPU (Raspberry Pi 4) Details:")
            else:
                # For RP2040, check if mcu attribute exists
                if hasattr(result.model.profile_info.table, 'mcu'):
                    device_table = result.model.profile_info.table.mcu
                    logger.info(f"\nMCU (RP2040) Details:")
                else:
                    # If mcu attribute doesn't exist, use the profile_info directly
                    logger.info(f"\nMCU (RP2040) Details:")
                    logger.info(f"Inference time: {profile_info.time_per_inference_ms} ms")
                    logger.info(f"ROM size: {profile_info.memory.tflite.rom} bytes")
                    logger.info(f"RAM usage: {profile_info.memory.tflite.ram} bytes")
                    logger.info(f"Supported: {'Yes' if profile_info.is_supported_on_mcu else 'No'}")
                    return
                
            logger.info(f"Description: {device_table.description}")
            logger.info(f"Inference time: {device_table.time_per_inference_ms} ms")
            logger.info(f"ROM size: {device_table.rom} bytes")
            logger.info(f"Supported: {'Yes' if device_table.supported else 'No'}")
        except AttributeError as e:
            logger.error(f"Error accessing device table: {str(e)}")
            logger.info("Using profile info instead:")
            logger.info(f"Inference time: {profile_info.time_per_inference_ms} ms")
            logger.info(f"ROM size: {profile_info.memory.tflite.rom} bytes")
            logger.info(f"RAM usage: {profile_info.memory.tflite.ram} bytes")
            logger.info(f"Supported: {'Yes' if profile_info.is_supported_on_mcu else 'No'}")

    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    if not API_KEY:
        logger.error("API key not found. Set EDGE_IMPULSE_API_KEY in your .env file.")
        return
    
    # Define the two models we want to profile
    models_to_profile = [
        ORIGINAL_MODEL_PATH,  # Original ONNX model
        PRUNED_MODEL_PATH     # Pruned ONNX model
    ]
    
    # Profile each model on each device
    for model_path in models_to_profile:
        for device in TARGET_DEVICES:
            logger.info(f"\n{'#'*70}")
            logger.info(f"PROFILING {os.path.basename(model_path)} FOR {device.upper()}")
            logger.info(f"{'#'*70}")
            profile_model(model_path, device)

if __name__ == "__main__":
    main()