#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path

def fix_yaml_file(config_file):
    """
    Check if a YAML file is malformed and fix it if necessary.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        str: Path to the fixed configuration file
    """
    try:
        # Try to load the file to see if it's valid
        with open(config_file, 'r') as f:
            yaml.safe_load(f)
        # If we get here, the file is valid
        return config_file
    except yaml.scanner.ScannerError as e:
        print(f"YAML error in {config_file}: {e}")
        print("Attempting to fix the file...")
        
        # Create a backup of the original file
        backup_file = f"{config_file}.bak"
        shutil.copy2(config_file, backup_file)
        
        # Read the file line by line and fix issues
        with open(config_file, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for i, line in enumerate(lines):
            # Check for lines that look like paths but aren't properly formatted as YAML
            if line.strip() and not line.strip().startswith('#') and ':' not in line and not line.startswith(' '):
                # This line looks like a path but isn't properly formatted
                # Add it as a value to the previous key or create a new key
                if i > 0 and fixed_lines and ':' in fixed_lines[-1]:
                    # Add as a value to the previous key
                    fixed_lines.append(f"  {line.strip()}\n")
                else:
                    # Create a new key
                    fixed_lines.append(f"data_path: {line.strip()}\n")
            else:
                fixed_lines.append(line)
        
        # Write the fixed content back to the file
        with open(config_file, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"Fixed YAML file saved to {config_file}")
        print(f"Original file backed up to {backup_file}")
        
        return config_file

def update_config(config_file, activation, enhanced_pruning, edge_impulse):
    """
    Update a YAML configuration file with the specified parameters.
    
    Args:
        config_file (str): Path to the configuration file
        activation (str): Activation function to use
        enhanced_pruning (bool): Whether to enable enhanced pruning
        edge_impulse (bool): Whether to enable Edge Impulse deployment
    
    Returns:
        str: Path to the updated configuration file
    """
    # Fix the YAML file if necessary
    config_file = fix_yaml_file(config_file)
    
    # Load the configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract dataset name from config file
    dataset_name = os.path.basename(config_file).replace('config_', '').replace('.yaml', '')
    
    # Update activation function
    if 'model' in config:
        config['model']['activation'] = activation
    
    # Update enhanced pruning
    if 'pruning' in config:
        config['pruning']['enhanced_pruning'] = enhanced_pruning
    
    # Update Edge Impulse deployment
    if 'export' in config and 'edge_impulse' in config['export']:
        config['export']['edge_impulse']['enabled'] = edge_impulse
    
    # Update paths
    if 'dataset' in config:
        config['dataset']['model_path'] = f"models/original/{activation}/{dataset_name}"
        # Add target_column for iris dataset
        if dataset_name == 'iris':
            config['dataset']['target_column'] = 'class'
    
    if 'training' in config:
        config['training']['save_dir'] = f"models/original/{activation}"
    
    if 'pruning' in config:
        config['pruning']['save_dir'] = f"models/pruned/{activation}"
    
    if 'inference' in config:
        config['inference']['output_dir'] = f"results/inference/{activation}"
    
    if 'output' in config:
        config['output']['save_dir'] = f"models/pruned/{activation}"
    
    # Create a temporary config file
    temp_config = f"temp_{dataset_name}_{activation}_{enhanced_pruning}_{edge_impulse}.yaml"
    
    # Save the updated configuration
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config

def main():
    parser = argparse.ArgumentParser(description='Update YAML configuration files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--activation', type=str, required=True, choices=['relu', 'sigmoid', 'tanh'], help='Activation function to use')
    parser.add_argument('--enhanced-pruning', type=str, required=True, choices=['true', 'false'], help='Whether to enable enhanced pruning')
    parser.add_argument('--edge-impulse', type=str, required=True, choices=['true', 'false'], help='Whether to enable Edge Impulse deployment')
    parser.add_argument('--run-pipeline', action='store_true', help='Run the pipeline with the updated configuration')
    
    args = parser.parse_args()
    
    # Convert string boolean to actual boolean
    enhanced_pruning = args.enhanced_pruning.lower() == 'true'
    edge_impulse = args.edge_impulse.lower() == 'true'
    
    # Update the configuration
    temp_config = update_config(args.config, args.activation, enhanced_pruning, edge_impulse)
    
    print(f"Updated configuration saved to {temp_config}")
    
    # Run the pipeline if requested
    if args.run_pipeline:
        print(f"Running pipeline with config: {temp_config}")
        os.system(f"python scripts/pipeline/pipeline_export_onnx_for_ei.py --config {temp_config}")
    
    return temp_config

if __name__ == '__main__':
    main() 