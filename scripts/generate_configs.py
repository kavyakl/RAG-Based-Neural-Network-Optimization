#!/usr/bin/env python3
import os
import yaml
import pandas as pd
from pathlib import Path

def get_dataset_info(csv_path):
    """Get dataset information from CSV file."""
    df = pd.read_csv(csv_path)
    input_features = len(df.columns) - 1  # Assuming last column is target
    num_classes = len(df[df.columns[-1]].unique())
    return input_features, num_classes, df.columns[-1]

def load_existing_config(dataset_name, activation="relu"):
    """Load existing config if it exists."""
    config_path = Path("config") / f"config_{dataset_name}.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def update_paths(config, dataset_name, activation):
    """Update paths in the config while preserving other settings."""
    # Update dataset paths
    config["dataset"]["data_path"] = f"data/{dataset_name}.csv"
    config["dataset"]["model_dir"] = f"models/{dataset_name}"
    config["dataset"]["path"] = f"models/original/{activation}/{dataset_name}"
    config["dataset"]["model_path"] = f"models/original/{activation}/{dataset_name}"
    
    # Update model paths
    config["model"]["model_save_path"] = f"models/{dataset_name}"
    
    # Update export paths
    config["export"]["save_dir"] = f"models/pruned/{activation}/{dataset_name}"
    
    # Update inference paths
    config["inference"]["output_dir"] = f"results/inference/{activation}/{dataset_name}"
    
    # Update output paths
    config["output"]["results_dir"] = f"results/{dataset_name}"
    config["output"]["save_dir"] = f"models/pruned/{activation}/{dataset_name}"
    
    # Update pruning paths
    config["pruning"]["save_dir"] = f"models/pruned/{activation}/{dataset_name}"
    
    # Update training paths
    config["training"]["save_dir"] = f"models/original/{activation}/{dataset_name}"
    
    # Disable Edge Impulse
    config["export"]["edge_impulse"]["enabled"] = False
    
    return config

def create_config(dataset_name, csv_path, activation):
    """Create a config file for a specific dataset using existing config as template."""
    # Load existing config if available
    config = load_existing_config(dataset_name, activation)
    
    # Get dataset information
    input_features, num_classes, target_column = get_dataset_info(csv_path)
    
    if config is None:
        # If no existing config, load template
        template_config = load_existing_config("template")
        if template_config is None:
            raise ValueError("Template config not found at config/config_template.yaml")
        
        # Create new config from template
        config = template_config.copy()
        
        # Update dataset-specific information
        config["dataset"].update({
            "input_features": input_features,
            "num_classes": num_classes,
            "target_column": target_column,
            "name": dataset_name
        })
        
        # Update model information
        config["model"].update({
            "input_features": input_features,
            "activation": activation,
            "hidden_neurons": max(20, input_features * 2)  # Ensure hidden_neurons is set
        })
        
        # Ensure Edge Impulse is disabled
        if "export" not in config:
            config["export"] = {}
        if "edge_impulse" not in config["export"]:
            config["export"]["edge_impulse"] = {}
        config["export"]["edge_impulse"]["enabled"] = False
        
    else:
        # Update only the necessary fields in existing config
        config["dataset"].update({
            "input_features": input_features,
            "num_classes": num_classes,
            "target_column": target_column
        })
        config["model"].update({
            "input_features": input_features,
            "activation": activation,
            "hidden_neurons": max(20, input_features * 2)  # Ensure hidden_neurons is set
        })
        # Ensure Edge Impulse is disabled
        config["export"]["edge_impulse"]["enabled"] = False
    
    # Update all paths
    config = update_paths(config, dataset_name, activation)
    
    return config

def main():
    # Create configs directory if it doesn't exist
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Get all CSV files in data directory
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    
    # Skip binary digit dataset as it's a special case
    csv_files = [f for f in csv_files if "binary" not in f.name]
    
    # Define activation functions
    activations = ["relu", "tanh", "sigmoid"]
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem
        
        for activation in activations:
            # Create config for this dataset and activation
            config = create_config(dataset_name, csv_file, activation)
            
            # Save config file
            config_path = configs_dir / f"config_{dataset_name}_{activation}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"Created/Updated config for {dataset_name} with {activation} at {config_path}")

if __name__ == "__main__":
    main() 