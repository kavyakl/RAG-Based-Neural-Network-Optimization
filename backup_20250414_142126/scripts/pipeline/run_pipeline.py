#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import logging
import json
import time
import numpy as np
from datetime import datetime

from src.training.train_model import train_model
from src.pruning.structural_prune import prune_model
from src.inference.inference import compare_models


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple) and all(isinstance(i, np.integer) for i in obj):
            return tuple(int(i) for i in obj)
        return super(NumpyEncoder, self).default(obj)

def setup_logging(config):
    """
    Set up logging based on the configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper())
    log_file = config.get('logging', {}).get('file', 'results/pipeline.log')
    
    # Create directory for log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_config(config):
    """
    Validate the configuration and set default values.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Validated configuration dictionary
    """
    # Check required fields
    if 'dataset' not in config or 'path' not in config['dataset']:
        raise ValueError("Configuration must include 'dataset.path'")
    
    if 'model' not in config or 'hidden_neurons' not in config['model']:
        raise ValueError("Configuration must include 'model.hidden_neurons'")
    
    # Set default values if not present
    if 'dataset' not in config:
        config['dataset'] = {}
    if 'name' not in config['dataset']:
        config['dataset']['name'] = os.path.splitext(os.path.basename(config['dataset']['path']))[0]
    
    # Set default training configuration
    if 'training' not in config:
        config['training'] = {'enabled': True}
    
    # Set default pruning configuration
    if 'pruning' not in config:
        config['pruning'] = {'enabled': True}
    
    # Set default inference configuration
    if 'inference' not in config:
        config['inference'] = {'enabled': True}
    
    return config

def run_pipeline(config):
    """
    Run the neural network optimization pipeline.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Results of the pipeline
    """
    logger = logging.getLogger(__name__)
    results = {
        'dataset': config['dataset']['name'],
        'hidden_neurons': config['model']['hidden_neurons'],
        'activation': config['model'].get('activation', 'relu'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }
    
    start_time = time.time()
    
    # Step 1: Training
    if config['training'].get('enabled', True):
        logger.info("Step 1: Training")
        train_config = {
            'dataset_path': config['dataset']['path'],
            'hidden_neurons': config['model']['hidden_neurons'],
            'activation': config['model'].get('activation', 'relu'),
            'epochs': config['model'].get('epochs', 1000),
            'learning_rate': config['model'].get('learning_rate', 0.001),
            'test_size': config['dataset'].get('test_size', 0.2),
            'random_state': config['dataset'].get('random_state', 42),
            'save_dir': config['training'].get('save_dir', 'models/original')
        }
        train_results = train_model(train_config)
        results['results']['training'] = {
            'train_accuracy': train_results['metrics']['train_accuracy'],
            'test_accuracy': train_results['metrics']['test_accuracy'],
            'model_dir': train_results['paths']['model_dir'],
            'model_path': train_results['paths']['model_path'],
            'train_data_path': train_results['paths']['train_data_path'],
            'test_data_path': train_results['paths']['test_data_path']
        }
        logger.info(f"Training completed. Test accuracy: {train_results['metrics']['test_accuracy']:.2f}%")
    else:
        logger.info("Training step disabled. Using existing model.")
        # Find the most recent model directory
        model_dir = os.path.join(config['training'].get('save_dir', 'models/original'), config['dataset']['name'])
        if not os.path.exists(model_dir):
            raise ValueError(f"No existing model found in {model_dir}")
        model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not model_dirs:
            raise ValueError(f"No model directories found in {model_dir}")
        latest_model = max(model_dirs, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
        model_dir = os.path.join(model_dir, latest_model)
        
        results['results']['training'] = {
            'model_dir': model_dir,
            'model_path': os.path.join(model_dir, 'model.pth'),
            'train_data_path': os.path.join(model_dir, 'train_data.npz'),
            'test_data_path': os.path.join(model_dir, 'test_data.npz')
        }
    
    # Step 2: Pruning
    if config['pruning'].get('enabled', True):
        if 'training' not in results['results']:
            logger.error("Cannot run pruning without training results")
        else:
            logger.info("Step 2: Pruning")
            prune_config = {
                'model_dir': results['results']['training']['model_dir'],
                'train_data_path': results['results']['training']['train_data_path'],
                'test_data_path': results['results']['training']['test_data_path'],
                'hidden_neurons': config['model']['hidden_neurons'],
                'dataset_name': config['dataset']['name'],
                'activation': config['model'].get('activation', 'relu'),
                'correlation_threshold': config['pruning'].get('correlation_threshold', 0.8),
                'activity_threshold': config['pruning'].get('activity_threshold', 0.1),
                'save_dir': config['pruning'].get('save_dir', 'models/pruned')
            }
            prune_results = prune_model(prune_config)
            results['results']['pruning'] = {
                'original_train_accuracy': prune_results['metrics']['original_train_accuracy'],
                'original_test_accuracy': prune_results['metrics']['original_test_accuracy'],
                'pruned_train_accuracy': prune_results['metrics']['pruned_train_accuracy'],
                'pruned_test_accuracy': prune_results['metrics']['pruned_test_accuracy'],
                'original_hidden_neurons': prune_results['metrics']['original_hidden_neurons'],
                'pruned_hidden_neurons': prune_results['metrics']['pruned_hidden_neurons'],
                'neurons_pruned': prune_results['metrics']['neurons_pruned'],
                'model_dir': prune_results['paths']['model_dir'],
                'model_path': prune_results['paths']['model_path'],
                'pruning_info_pickle_path': prune_results['paths']['pruning_info_pickle_path'],
                'pruning_info_yaml_path': prune_results['paths']['pruning_info_yaml_path']
            }
            logger.info(f"Pruning completed. Pruned neurons: {prune_results['metrics']['neurons_pruned']}. "
                        f"Test accuracy: {prune_results['metrics']['pruned_test_accuracy']:.2f}%")
    else:
        logger.info("Pruning step disabled. Using existing pruned model.")
        # Find the most recent pruned model directory
        pruned_dir = os.path.join(config['pruning'].get('save_dir', 'models/pruned'), config['dataset']['name'])
        if not os.path.exists(pruned_dir):
            raise ValueError(f"No existing pruned model found in {pruned_dir}")
        pruned_dirs = [d for d in os.listdir(pruned_dir) if os.path.isdir(os.path.join(pruned_dir, d))]
        if not pruned_dirs:
            raise ValueError(f"No pruned model directories found in {pruned_dir}")
        latest_pruned = max(pruned_dirs, key=lambda x: os.path.getctime(os.path.join(pruned_dir, x)))
        pruned_dir = os.path.join(pruned_dir, latest_pruned)
        
        results['results']['pruning'] = {
            'model_dir': pruned_dir,
            'model_path': os.path.join(pruned_dir, 'model.pth'),
            'pruning_info_pickle_path': os.path.join(pruned_dir, f"{config['dataset']['name']}_pruning_info.pkl"),
            'pruning_info_yaml_path': os.path.join(pruned_dir, f"{config['dataset']['name']}_pruning_info.yaml")
        }
    
    # Step 3: Inference
    if config['inference'].get('enabled', True):
        if 'pruning' not in results['results']:
            logger.error("Cannot run inference without pruning results")
        else:
            logger.info("Step 3: Inference")
            inference_config = {
                'data_path': config['dataset']['path'],
                'original_model_dir': results['results']['training']['model_dir'],
                'pruned_model_dir': results['results']['pruning']['model_dir'],
                'output_dir': config['inference'].get('output_dir', 'results/inference'),
                'dataset_name': config['dataset']['name']
            }
            inference_results = compare_models(**inference_config)
            results['results']['inference'] = {
                'original_accuracy': inference_results['original']['accuracy'],
                'pruned_accuracy': inference_results['pruned']['accuracy'],
                'accuracy_difference': inference_results['accuracy_difference'],
                'size_reduction': inference_results['size_reduction'],
                'results_path': inference_results['original']['results_path']
            }
            logger.info(f"Inference completed. Accuracy difference: {inference_results['accuracy_difference']:.2f}%")
    else:
        logger.info("Inference step disabled")
    
    # Calculate total time
    end_time = time.time()
    results['execution_time'] = end_time - start_time
    
    # Print summary
    logger.info("Pipeline completed")
    logger.info(f"Total execution time: {results['execution_time']:.2f} seconds")
    
    # Save results
    results_dir = config.get('output', {}).get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f"{config['dataset']['name']}_pipeline_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_file}")
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run neural network optimization pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    config = validate_config(config)
    
    # Set up logging
    setup_logging(config)
    
    # Run pipeline
    run_pipeline(config)

if __name__ == '__main__':
    main() 