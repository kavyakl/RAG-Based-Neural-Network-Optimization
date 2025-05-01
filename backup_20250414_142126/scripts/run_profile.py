#!/usr/bin/env python3
"""
Script to run model profiling from the command line.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.export.profile_models import profile_models
from src.utils.config import load_config

def main():
    """Main function to run profiling"""
    parser = argparse.ArgumentParser(description='Profile ONNX models')
    parser.add_argument('--config', type=str, default='config/config_iris.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run profiling
    results = profile_models(config)
    
    if results:
        print("Profiling completed successfully!")
        print(f"Results: {results}")
    else:
        print("Profiling failed or was skipped.")

if __name__ == "__main__":
    main() 