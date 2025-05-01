import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.inference.inference import compare_models
from src.inference.quantized_inference import run_quantized_inference, compare_quantized_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('config/config_breast_cancer.yaml')
    
    # Create output directory
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    try:
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
        
        # Run standard inference
        logger.info("Running standard inference...")
        compare_models(
            data_path=config['dataset']['path'],
            original_model_dir=latest_original_dir,
            pruned_model_dir=latest_pruned_dir,
            output_dir=config['inference']['output_dir'],
            dataset_name=config['dataset']['name']
        )
        
        # Get paths to quantized models
        original_quant_dir = os.path.join(config['quantization']['save_dir'], 'original', config['dataset']['name'])
        pruned_quant_dir = os.path.join(config['quantization']['save_dir'], 'pruned', config['dataset']['name'])
        
        # Run quantized inference
        logger.info("Running quantized inference...")
        compare_quantized_models(
            data_path=config['dataset']['path'],
            original_quantized_path=os.path.join(original_quant_dir, f"{config['dataset']['name']}_original_quantized.pth"),
            original_info_path=os.path.join(original_quant_dir, f"{config['dataset']['name']}_original_quantized_info.json"),
            pruned_quantized_path=os.path.join(pruned_quant_dir, f"{config['dataset']['name']}_pruned_quantized.pth"),
            pruned_info_path=os.path.join(pruned_quant_dir, f"{config['dataset']['name']}_pruned_quantized_info.json"),
            output_dir=config['inference']['output_dir'],
            dataset_name=config['dataset']['name']
        )
        
        logger.info("Inference completed successfully")
        logger.info(f"Results saved to: {config['output']['results_dir']}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main() 