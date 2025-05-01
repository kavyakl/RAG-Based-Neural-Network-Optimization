import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.quantization.quantize import quantize_model

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
    
    # Extract paths from config
    original_model_dir = config['models']['original']['dir']
    pruned_model_dir = config['models']['pruned']['dir']
    
    # Get quantization parameters
    quantization_config = config.get('quantization', {})
    bits = quantization_config.get('bits', 8)
    method = quantization_config.get('method', 'uniform')
    
    try:
        # Quantize original model
        logger.info("Quantizing original model...")
        quantize_model(
            model_dir=original_model_dir,
            bits=bits,
            method=method
        )
        
        # Quantize pruned model
        logger.info("Quantizing pruned model...")
        quantize_model(
            model_dir=pruned_model_dir,
            bits=bits,
            method=method
        )
        
        logger.info("Model quantization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model quantization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 