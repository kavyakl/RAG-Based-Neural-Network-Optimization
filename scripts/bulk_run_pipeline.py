#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import argparse
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bulk_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_config_files(config_dir="configs"):
    """Find all YAML config files in the specified directory."""
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    return config_files

def run_pipeline(config_file):
    """Run the pipeline for a specific config file."""
    logger.info(f"Running pipeline for config: {config_file}")
    
    # Construct the command
    cmd = ["python3", "scripts/pipeline/pipeline_export_onnx_for_ei.py", "--config", config_file]
    
    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed for {config_file} in {duration:.2f} seconds")
        logger.info(f"Output: {result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed for {config_file}")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk run the neural network pipeline for all config files")
    parser.add_argument("--config-dir", default="configs", help="Directory containing config files")
    parser.add_argument("--configs", nargs="+", help="Specific config files to run (optional)")
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], 
                        help="Run only configs for specific activation function")
    args = parser.parse_args()
    
    # Get config files
    if args.configs:
        config_files = args.configs
        logger.info(f"Running specific configs: {config_files}")
    else:
        config_files = find_config_files(args.config_dir)
        
        # Filter by activation function if specified
        if args.activation:
            config_files = [f for f in config_files if f"_{args.activation}.yaml" in f]
            logger.info(f"Filtered to {len(config_files)} configs for activation: {args.activation}")
        
        logger.info(f"Found {len(config_files)} config files: {config_files}")
    
    # Run pipeline for each config
    results = {}
    for config_file in config_files:
        success = run_pipeline(config_file)
        results[config_file] = "Success" if success else "Failed"
    
    # Print summary
    logger.info("\n=== Pipeline Run Summary ===")
    for config_file, result in results.items():
        logger.info(f"{config_file}: {result}")
    
    # Check if all runs were successful
    all_successful = all(result == "Success" for result in results.values())
    if all_successful:
        logger.info("All pipeline runs completed successfully!")
    else:
        logger.warning("Some pipeline runs failed. Check the logs for details.")

if __name__ == "__main__":
    main() 