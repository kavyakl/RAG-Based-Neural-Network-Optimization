#!/usr/bin/env python3
"""
Stand-alone script to export neural network models for Arduino/RP2040.
"""

import os
import logging
import argparse
from typing import Dict, Any
import yaml

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
    """Export models for Arduino/RP2040 deployment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Export neural network models for Arduino/RP2040')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--model_type', type=str, default='all', choices=['original', 'pruned', 'all'], 
                      help='Type of model to export')
    parser.add_argument('--output_dir', type=str, default='models/arduino', 
                      help='Directory to save exported models')
    parser.add_argument('--test_samples', type=int, default=5, 
                      help='Number of test samples to include')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Import deployment modules
    from export_arduino_model import export_model_to_c, export_test_data
    from generate_arduino_sketch import generate_arduino_sketch
    
    # Get dataset information
    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    
    # Determine model types to export
    model_types = []
    if args.model_type == 'all':
        model_types = ['original', 'pruned']
    else:
        model_types = [args.model_type]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_export_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_export_dir, exist_ok=True)
    
    # Export test data
    logger.info(f"Exporting test data for {dataset_name}...")
    test_data_path = export_test_data(
        data_path=dataset_path,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        num_samples=args.test_samples
    )
    
    # Export each model type
    exported_models = {}
    for model_type in model_types:
        logger.info(f"Exporting {model_type} model for {dataset_name}...")
        
        # Get quantized model path
        model_dir = os.path.join(config['quantization']['save_dir'], model_type, dataset_name)
        model_path = os.path.join(model_dir, f"{dataset_name}_{model_type}_quantized.pth")
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, skipping...")
            continue
        
        # Export model to C header
        export_result = export_model_to_c(
            model_path=model_path,
            output_dir=args.output_dir,
            model_type=model_type,
            dataset_name=dataset_name,
            bit_width=config['quantization']['bits']
        )
        
        exported_models[model_type] = export_result
        logger.info(f"Exported {model_type} model to {export_result['header_path']}")
    
    # Generate Arduino sketches
    if exported_models:
        logger.info("Generating Arduino sketches...")
        arduino_sketches = {}
        
        for model_type, export_info in exported_models.items():
            sketch_dir = generate_arduino_sketch(
                dataset_name=dataset_name,
                model_type=model_type,
                model_header_path=export_info['header_path'],
                test_data_header_path=test_data_path,
                output_dir=args.output_dir
            )
            
            arduino_sketches[model_type] = sketch_dir
            logger.info(f"Generated Arduino sketch for {model_type} model at {sketch_dir}")
        
        logger.info(f"Arduino deployment assets generated in {args.output_dir}")
    else:
        logger.error("No models were exported for deployment")

if __name__ == "__main__":
    main() 