#!/usr/bin/env python3
import os
import sys
import yaml
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.pruning.structural_prune import structural_prune
from src.training.train_model import train_model
from src.utils.data_handling import prepare_data_for_training
from src.utils.model_io import load_model_and_info
from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_iris_model(config, activation):
    """Train the iris model with the specified activation function."""
    logger.info(f"Training iris model with {activation} activation function...")
    
    # Update activation in config
    config['model']['activation'] = activation
    
    # Set up training config
    training_config = {
        'dataset_path': config['dataset']['path'],
        'hidden_neurons': config['model']['hidden_neurons'],
        'activation': activation,
        'epochs': config['model'].get('epochs', 1000),
        'learning_rate': config['model'].get('learning_rate', 0.001),
        'test_size': config['dataset'].get('test_size', 0.2),
        'random_state': config['dataset'].get('random_state', 42),
        'save_dir': f"models/original/{activation}",
        'model_name': "iris_model"
    }
    
    # Train model
    results = train_model(training_config)
    
    logger.info(f"Training completed for iris model with {activation} activation function.")
    logger.info(f"Test accuracy: {results['metrics']['test_accuracy']:.2f}%")
    
    return results

def main():
    """Run the training and pruning process for the iris dataset with enhanced pruning enabled."""
    # Load configuration
    config_path = "config/config_iris.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Set enhanced pruning to true
    if 'validation' not in config:
        config['validation'] = {}
    config['validation']['enhanced_pruning'] = True
    
    # Set target pruning rate to 35% for iris dataset
    config['validation']['target_pruning_rate'] = 0.35
    
    # Set validation strategy
    config['validation']['validation_strategy'] = 'single'
    
    # Run training and pruning for each activation function
    activation_functions = ['relu', 'sigmoid', 'tanh']
    
    for activation in activation_functions:
        logger.info(f"Processing iris dataset with {activation} activation function...")
        
        # Train model
        train_results = train_iris_model(config, activation)
        
        # Set activation-specific max accuracy drop
        if activation == 'sigmoid':
            config['validation']['max_accuracy_drop'] = 0.08
        elif activation == 'tanh':
            config['validation']['max_accuracy_drop'] = 0.09
        else:  # ReLU
            config['validation']['max_accuracy_drop'] = 0.12
        
        # Update paths in config
        config['model_path'] = f"models/original/{activation}/iris"
        config['save_dir'] = f"models/pruned/{activation}"
        config['output_dir'] = f"results/inference/{activation}"
        
        # Run pruning
        structural_prune(config)
        
        logger.info(f"Pruning completed for iris dataset with {activation} activation function.")

if __name__ == "__main__":
    main() 