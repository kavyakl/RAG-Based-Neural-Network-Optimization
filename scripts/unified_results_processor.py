#!/usr/bin/env python3

import os
import json
import pandas as pd
import glob
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_result_files(dataset_name=None):
    """Find all result files and annotate them with their activation function."""
    results_dir = Path("results")
    annotated_files = []
    result_files = []
    
    # First scan results directory for datasets
    if not results_dir.exists():
        logger.warning("Results directory not found")
        return annotated_files, result_files
        
    # Get all dataset directories
    dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    # Filter by dataset name if specified
    if dataset_name:
        dataset_dirs = [d for d in dataset_dirs if d.name == dataset_name]
    
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        logger.info(f"Processing dataset: {dataset}")
        
        # First check for original model results in the dataset directory
        original_files = list(dataset_dir.glob("*.json"))
        for file_path in original_files:
            if "metrics" in file_path.name or "results" in file_path.name:
                logger.info(f"Found original model file: {file_path}")
                annotated_files.append({
                    "file_path": str(file_path),
                    "dataset": dataset,
                    "activation": "original"
                })
                result_files.append(str(file_path))
        
        # Then check for pruned model results in the nested directory
        nested_dir = dataset_dir / dataset
        if nested_dir.exists():
            pruned_files = list(nested_dir.glob("*.json"))
            for file_path in pruned_files:
                if "metrics" in file_path.name or "results" in file_path.name:
                    logger.info(f"Found pruned model file: {file_path}")
                    annotated_files.append({
                        "file_path": str(file_path),
                        "dataset": dataset,
                        "activation": "pruned"
                    })
                    result_files.append(str(file_path))
        
        # Check for inference results in the inference directory
        inference_dir = dataset_dir / "inference"
        if inference_dir.exists():
            logger.info(f"Found inference directory for {dataset}")
            # Look for inference results in both original and pruned subdirectories
            for model_type in ["original", "pruned"]:
                model_dir = inference_dir / model_type
                if model_dir.exists():
                    inference_files = list(model_dir.glob("*_inference_results.json"))
                    for file_path in inference_files:
                        logger.info(f"Found {model_type} inference file: {file_path}")
                        # Try to extract activation from filename
                        activation = "unknown"
                        for act in ["relu", "sigmoid", "tanh"]:
                            if act in file_path.name.lower():
                                activation = act
                                break
                        annotated_files.append({
                            "file_path": str(file_path),
                            "dataset": dataset,
                            "activation": activation,
                            "model_type": model_type,
                            "is_inference": True
                        })
                        result_files.append(str(file_path))
    
    return annotated_files, result_files

def find_profiling_files(dataset_name: Optional[str] = None, activation: Optional[str] = None) -> List[str]:
    """Find all profiling result files matching the criteria"""
    profiling_dir = Path("results/profiling")
    
    if not profiling_dir.exists():
        return []
    
    # First try specific profiling files
    if dataset_name and activation:
        pattern = f"{profiling_dir}/{dataset_name}_{activation}_*profiling*.json"
        files = glob.glob(pattern)
        if not files:
            # Try with dataset name only
            pattern = f"{profiling_dir}/{dataset_name}*profiling*.json"
            files = glob.glob(pattern)
    elif dataset_name:
        pattern = f"{profiling_dir}/{dataset_name}*profiling*.json"
        files = glob.glob(pattern)
    elif activation:
        pattern = f"{profiling_dir}/*{activation}*profiling*.json"
        files = glob.glob(pattern)
    else:
        pattern = f"{profiling_dir}/*profiling*.json"
        files = glob.glob(pattern)
    
    return files

def extract_metadata(file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metadata from result file path and content"""
    try:
        file_path = file_info["file_path"]
        activation = file_info["activation"]
        is_inference = file_info.get("is_inference", False)
        
        # Extract info from path
        parts = Path(file_path).parts
        dataset_name = file_info["dataset"]
        model_type = file_info.get("model_type", "unknown")
        
        # Determine model type based on filename pattern if not already set
        if model_type == "unknown":
            if 'pruned' in file_path:
                model_type = 'pruned'
            elif 'original' in file_path:
                model_type = 'original'
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Initialize metadata dictionary
        metadata = {
            'file_path': file_path,
            'dataset_name': dataset_name,
            'activation': activation,
            'model_type': model_type,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }
        
        if is_inference:
            # Extract inference-specific metrics
            metadata.update({
                'accuracy': data.get('metrics', {}).get('accuracy', 0),
                'precision': data.get('metrics', {}).get('precision', 0),
                'recall': data.get('metrics', {}).get('recall', 0),
                'f1': data.get('metrics', {}).get('f1', 0),
                'size_bytes': data.get('size_bytes', 0),
                'inference_time_ms': data.get('inference_time_ms', 0)
            })
        else:
            # Get accuracy values
            accuracy = 0
            if model_type == 'pruned':
                accuracy = data.get('pruned_accuracy',
                           data.get('test_accuracy',
                           data.get('accuracy', 0)))
            else:  # original
                accuracy = data.get('original_accuracy',
                           data.get('test_accuracy',
                           data.get('accuracy', 0)))
            
            # Scale accuracy if it's between 0 and 1
            if isinstance(accuracy, (int, float)) and accuracy <= 1.0 and accuracy > 0:
                normalized_accuracy = accuracy * 100
            else:
                normalized_accuracy = accuracy if isinstance(accuracy, (int, float)) else 0
            
            # Extract detailed metrics
            metadata.update({
                'accuracy': normalized_accuracy,
                'hidden_neurons': data.get('hidden_neurons', 
                                data.get('original_hidden_neurons', 0)),
                'pruned_neurons': data.get('pruned_hidden_neurons', 0) if model_type == 'pruned' else 0,
                'pruning_rate': data.get('pruning_rate', 0) if model_type == 'pruned' else 0,
                'accuracy_drop': data.get('accuracy_drop', 0) if model_type == 'pruned' else 0
            })
        
        return metadata
    
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {str(e)}")
        return None

def extract_profiling_data(profiling_files: List[str], dataset_name: Optional[str] = None, activation: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Extract ROM/RAM details from profiling files"""
    profiling_data = {}
    
    for file_path in profiling_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract dataset and activation from the file
            file_dataset = data.get('dataset', '')
            file_activation = data.get('activation', '')
            
            # Skip unknown activations
            if file_activation == 'unknown':
                continue
                
            # Check if this is for the right dataset
            if (dataset_name is None or 
                dataset_name.lower() in file_path.lower() or 
                dataset_name.lower() == file_dataset.lower() or
                dataset_name.replace('_', '') == file_dataset):
                
                # If activation is specified, check for a match
                if activation and activation != 'unknown' and activation != file_activation:
                    continue
                
                # Initialize dataset entry if needed
                if file_dataset not in profiling_data:
                    profiling_data[file_dataset] = {}
                
                # Initialize activation entry if needed
                if file_activation not in profiling_data[file_dataset]:
                    profiling_data[file_dataset][file_activation] = {}
                
                # Extract original model data
                if 'original' in data:
                    profiling_data[file_dataset][file_activation]['original'] = data['original']
                
                # Extract pruned model data
                if 'pruned' in data:
                    profiling_data[file_dataset][file_activation]['pruned'] = data['pruned']
                
                print(f"Extracted profiling data for {file_dataset} with activation {file_activation}")
        except Exception as e:
            print(f"Error extracting profiling data from {file_path}: {str(e)}")
    
    return profiling_data

def create_results_table(result_files: List[str]) -> pd.DataFrame:
    """Create a table with metadata from result files"""
    metadata_list = []
    
    for file_path in result_files:
        metadata = extract_metadata(file_path)
        if metadata:
            metadata_list.append(metadata)
    
    if not metadata_list:
        print("No results found.")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    return df

def create_activation_comparison(df: pd.DataFrame, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Create a comparison table across activation functions"""
    # Filter by dataset if specified
    if dataset_name:
        df = df[df['dataset_name'] == dataset_name]
    
    # Group by dataset and activation
    comparison = []
    
    # Find all unique datasets
    datasets = df['dataset_name'].unique().tolist()
    # Remove None values
    datasets = [d for d in datasets if d is not None]
    print(f"\nDatasets found: {datasets}")
    
    if not datasets:
        print("No valid datasets found in the results.")
        return pd.DataFrame()
    
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        
        # Get all metrics files for this dataset
        metrics_files = glob.glob(f"results/{dataset}/**/*metrics*.json", recursive=True)
        
        # Process each activation function
        for activation in ['relu', 'tanh', 'sigmoid']:
            metrics = {
                'dataset': dataset,
                'activation': activation,
                'original_accuracy': 0,
                'pruned_accuracy': 0,
                'accuracy_drop': 0,
                'pruning_rate': 0,
                'original_neurons': 0,
                'pruned_neurons': 0,
                'neurons_removed': 0
            }
            
            # Find original metrics file for this activation
            original_files = [f for f in metrics_files if 'original' in f]
            if original_files:
                with open(original_files[-1], 'r') as f:  # Use latest file
                    data = json.load(f)
                    metrics['original_accuracy'] = data.get('test_accuracy', 0) * 100
                    metrics['original_neurons'] = data.get('hidden_neurons', 0)
            
            # Find pruned metrics file for this activation
            pruned_files = [f for f in metrics_files if 'pruned' in f]
            if pruned_files:
                with open(pruned_files[-1], 'r') as f:  # Use latest file
                    data = json.load(f)
                    if data.get('activation', '').lower() == activation:
                        metrics['pruned_accuracy'] = data.get('pruned_accuracy', 0)
                        metrics['pruned_neurons'] = data.get('pruned_hidden_neurons', 0)
                        metrics['accuracy_drop'] = data.get('accuracy_drop', 0)
                        if metrics['original_neurons'] > 0:
                            metrics['pruning_rate'] = (metrics['original_neurons'] - metrics['pruned_neurons']) / metrics['original_neurons']
                        metrics['neurons_removed'] = metrics['original_neurons'] - metrics['pruned_neurons']
            
            # Find inference results if available
            inference_dir = f"results/{dataset}/inference"
            if os.path.exists(inference_dir):
                # Original model inference
                orig_inf_files = glob.glob(f"{inference_dir}/original/*{activation}*_inference_results.json")
                if orig_inf_files:
                    with open(orig_inf_files[-1], 'r') as f:
                        data = json.load(f)
                        metrics.update({
                            'original_rom_bytes': data.get('rom_size_bytes', 0),
                            'original_ram_bytes': data.get('ram_usage_bytes', 0),
                            'original_inference_ms': data.get('inference_time_ms', 0)
                        })
                
                # Pruned model inference
                pruned_inf_files = glob.glob(f"{inference_dir}/pruned/*{activation}*_inference_results.json")
                if pruned_inf_files:
                    with open(pruned_inf_files[-1], 'r') as f:
                        data = json.load(f)
                        metrics.update({
                            'pruned_rom_bytes': data.get('rom_size_bytes', 0),
                            'pruned_ram_bytes': data.get('ram_usage_bytes', 0),
                            'pruned_inference_ms': data.get('inference_time_ms', 0)
                        })
                        
                        # Calculate reductions if we have both original and pruned data
                        if metrics.get('original_rom_bytes', 0) > 0:
                            metrics['rom_reduction'] = metrics['original_rom_bytes'] - metrics['pruned_rom_bytes']
                            metrics['rom_reduction_percent'] = (metrics['rom_reduction'] / metrics['original_rom_bytes']) * 100
                        
                        if metrics.get('original_ram_bytes', 0) > 0:
                            metrics['ram_reduction'] = metrics['original_ram_bytes'] - metrics['pruned_ram_bytes']
                            metrics['ram_reduction_percent'] = (metrics['ram_reduction'] / metrics['original_ram_bytes']) * 100
                        
                        if metrics.get('original_inference_ms', 0) > 0:
                            metrics['inference_speedup'] = metrics['original_inference_ms'] / metrics['pruned_inference_ms']
            
            comparison.append(metrics)
    
    comparison_df = pd.DataFrame(comparison) if comparison else pd.DataFrame()
    
    # Save comparison if we have data
    if len(comparison_df) > 0:
        comparison_path = 'results/activation_comparison.csv'
        if dataset_name:
            comparison_path = f'results/{dataset_name}_activation_comparison.csv'
        
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Activation comparison saved to {comparison_path}")
    
    return comparison_df

def crosscheck_accuracy_metrics(results_df: pd.DataFrame, metrics_data: Dict) -> List[Tuple[str, str, float, float]]:
    """Crosscheck accuracy metrics between results index and metrics files"""
    discrepancies = []
    
    for dataset in metrics_data:
        dataset_results = results_df[results_df['dataset_name'] == dataset]
        
        # Check original model accuracies
        for orig_metrics in metrics_data[dataset]['original']:
            orig_acc = orig_metrics.get('test_accuracy', 0)
            matching_results = dataset_results[
                (dataset_results['model_type'] == 'original') & 
                (abs(dataset_results['accuracy'] - orig_acc) < 0.01)
            ]
            
            if len(matching_results) == 0:
                discrepancies.append((
                    dataset,
                    'original',
                    orig_acc,
                    dataset_results[dataset_results['model_type'] == 'original']['accuracy'].mean() if len(dataset_results) > 0 else 0
                ))
        
        # Check pruned model accuracies
        for pruned_metrics in metrics_data[dataset]['pruned']:
            pruned_acc = pruned_metrics.get('test_accuracy', 0)
            matching_results = dataset_results[
                (dataset_results['model_type'] == 'pruned') & 
                (abs(dataset_results['accuracy'] - pruned_acc) < 0.01)
            ]
            
            if len(matching_results) == 0:
                discrepancies.append((
                    dataset,
                    'pruned',
                    pruned_acc,
                    dataset_results[dataset_results['model_type'] == 'pruned']['accuracy'].mean() if len(dataset_results) > 0 else 0
                ))
    
    return discrepancies

def crosscheck_profiling_metrics(profiling_data: Dict, activation_comparison_df: pd.DataFrame) -> List[Tuple[str, str, str, float, float]]:
    """Crosscheck profiling metrics between profiling files and activation comparison"""
    discrepancies = []
    
    for dataset in profiling_data:
        dataset_comparison = activation_comparison_df[activation_comparison_df['dataset'] == dataset]
        
        for activation, profiling_entry in profiling_data[dataset].items():
            # Skip unknown activations
            if activation == 'unknown':
                continue
                
            for model_type, device_data in profiling_entry.items():
                if model_type in ['original', 'pruned']:
                    # Check for various devices, prioritize RP2040
                    device = None
                    if 'raspberry-pi-rp2040' in device_data:
                        device = 'raspberry-pi-rp2040'
                    elif 'arduino-nano-33' in device_data:
                        device = 'arduino-nano-33'
                    elif 'raspberry-pi-4' in device_data:
                        device = 'raspberry-pi-4'
                        
                    if device:
                        inference_time = device_data[device].get('inference_time_ms', 0)
                        rom_size = device_data[device].get('rom_size_bytes', 0)
                        ram_usage = device_data[device].get('ram_usage_bytes', 0)
                        
                        # Find matching entry in comparison DataFrame
                        matching_entries = dataset_comparison[
                            (dataset_comparison['activation'] == activation) &
                            (abs(dataset_comparison[f'{model_type}_inference_ms'] - inference_time) < 0.1) &
                            (abs(dataset_comparison[f'{model_type}_rom_bytes'] - rom_size) < 1) &
                            (abs(dataset_comparison[f'{model_type}_ram_bytes'] - ram_usage) < 1)
                        ]
                        
                        if len(matching_entries) == 0:
                            discrepancies.append((
                                dataset,
                                device,
                                'profiling',
                                inference_time,
                                dataset_comparison[f'{model_type}_inference_ms'].mean() if len(dataset_comparison) > 0 else 0
                            ))
    
    return discrepancies

def load_metrics_data(dataset_name: Optional[str] = None) -> Dict:
    """Load metrics data from JSON files"""
    metrics_data = {}
    pattern = f'results/{dataset_name}/*_metrics_*.json' if dataset_name else 'results/*/*_metrics_*.json'
    
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract dataset name from path
                dataset = os.path.basename(os.path.dirname(file_path))
                if dataset not in metrics_data:
                    metrics_data[dataset] = {'original': [], 'pruned': []}
                # Determine if original or pruned based on filename
                if 'original' in file_path:
                    metrics_data[dataset]['original'].append(data)
                else:
                    metrics_data[dataset]['pruned'].append(data)
        except Exception as e:
            print(f"Error loading metrics file {file_path}: {str(e)}")
    
    return metrics_data

def process_pruned_metrics(file_path):
    """Process pruned metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        metrics = {
            'activation': data.get('activation'),
            'pruned_accuracy': data.get('pruned_accuracy', 0),
            'original_accuracy': data.get('original_accuracy', 0),
            'accuracy_drop': data.get('accuracy_drop', 0),
            'original_hidden_neurons': data.get('original_hidden_neurons', 0),
            'pruned_hidden_neurons': data.get('pruned_hidden_neurons', 0),
            'pruning_rate': 0  # Will calculate below
        }
        
        # Calculate pruning rate if we have both neuron counts
        if metrics['original_hidden_neurons'] and metrics['pruned_hidden_neurons']:
            metrics['pruning_rate'] = (
                (metrics['original_hidden_neurons'] - metrics['pruned_hidden_neurons']) 
                / metrics['original_hidden_neurons'] * 100
            )
            
        return metrics
    except Exception as e:
        print(f"Error processing pruned metrics from {file_path}: {str(e)}")
        return None

def generate_report(dataset_name, metrics_data):
    """Generate analysis report from processed metrics."""
    report = f"# Neural Network Analysis Report for {dataset_name}\n\n"
    
    def format_value(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    
    if metrics_data.get('original_metrics'):
        report += "## Original Model Performance\n"
        report += f"- Training Accuracy: {format_value(metrics_data['original_metrics'].get('train_accuracy'))}\n"
        report += f"- Test Accuracy: {format_value(metrics_data['original_metrics'].get('test_accuracy'))}\n\n"
    
    if metrics_data.get('pruned_metrics'):
        report += "## Pruning Analysis\n"
        pruned = metrics_data['pruned_metrics']
        report += f"- Original Hidden Neurons: {format_value(pruned.get('original_neurons'))}\n"
        report += f"- Pruned Hidden Neurons: {format_value(pruned.get('pruned_neurons'))}\n"
        report += f"- Pruning Rate: {format_value(pruned.get('pruning_rate')*100)}%\n"
        report += f"- Activation Function: {format_value(pruned.get('activation'))}\n"
        report += f"- Original Accuracy: {format_value(pruned.get('original_accuracy'))}%\n"
        report += f"- Pruned Accuracy: {format_value(pruned.get('pruned_accuracy'))}%\n"
        report += f"- Accuracy Drop: {format_value(pruned.get('accuracy_drop'))}%\n"
        report += f"- Average Importance Score: {format_value(pruned.get('avg_importance_score'))}\n\n"
    
    if not metrics_data.get('original_metrics') and not metrics_data.get('pruned_metrics'):
        report += "\nNo metrics data available for analysis.\n"
    
    return report

def print_summary(results_df: pd.DataFrame, activation_comparison_df: pd.DataFrame):
    """Print a summary of the results"""
    # Count results by activation and model type
    counts = results_df.groupby(['dataset_name', 'activation', 'model_type']).size().unstack()
    
    print("\n=== Results Summary ===")
    print(counts)
    
    if len(activation_comparison_df) > 0:
        print("\n=== Activation Function Comparison ===")
        
        # Format for pretty printing
        for dataset, dataset_df in activation_comparison_df.groupby('dataset'):
            print(f"\nDataset: {dataset}")
            # Check if we have memory data
            has_memory_data = dataset_df['original_rom_bytes'].sum() > 0 or dataset_df['pruned_rom_bytes'].sum() > 0
            
            if has_memory_data:
                print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
                print(f"{'Activation':<10} | {'Orig Acc':<10} | {'Pruned Acc':<10} | {'Acc Drop':<10} | {'Orig ROM':<10} | {'Pruned ROM':<10} | {'ROM Save%':<10} | {'Orig RAM':<10} | {'Pruned RAM':<10} | {'Orig Inf':<10} | {'Pruned Inf':<10}")
                print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
                
                for _, row in dataset_df.iterrows():
                    print(f"{row['activation']:<10} | "
                          f"{row['original_accuracy']:<10.2f} | "
                          f"{row['pruned_accuracy']:<10.2f} | "
                          f"{row['accuracy_drop']:<10.2f} | "
                          f"{row['original_rom_bytes']:<10} | "
                          f"{row['pruned_rom_bytes']:<10} | "
                          f"{row['rom_reduction_percent']:<10.2f} | "
                          f"{row['original_ram_bytes']:<10} | "
                          f"{row['pruned_ram_bytes']:<10} | "
                          f"{row['original_inference_ms']:<10} | "
                          f"{row['pruned_inference_ms']:<10}")
            else:
                print(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*12}")
                print(f"{'Activation':<10} | {'Orig Acc':<12} | {'Pruned Acc':<12} | {'Acc Drop':<10} | {'Pruning Rate':<12}")
                print(f"{'-'*10} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*12}")
                
                for _, row in dataset_df.iterrows():
                    print(f"{row['activation']:<10} | {row['original_accuracy']:<12.2f} | {row['pruned_accuracy']:<12.2f} | {row['accuracy_drop']:<10.2f} | {row['pruning_rate']*100:<12.2f}")

def analyze_inference_time(profiling_data: Dict, activation_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze inference time across different models and devices"""
    inference_analysis = []
    
    for dataset in profiling_data:
        dataset_comparison = activation_comparison_df[activation_comparison_df['dataset'] == dataset]
        
        for activation, profiling_entry in profiling_data[dataset].items():
            # Skip unknown activations
            if activation == 'unknown':
                continue
                
            # Get original and pruned data
            original_data = profiling_entry.get('original', {})
            pruned_data = profiling_entry.get('pruned', {})
            
            # Process each device
            devices = set(original_data.keys()) | set(pruned_data.keys())
            for device in devices:
                if device not in ['raspberry-pi-rp2040', 'arduino-nano-33', 'raspberry-pi-4']:
                    continue
                    
                # Get metrics for original model
                orig_metrics = original_data.get(device, {})
                orig_inference = orig_metrics.get('inference_time_ms', 0)
                orig_rom = orig_metrics.get('rom_size_bytes', 0)
                orig_ram = orig_metrics.get('ram_usage_bytes', 0)
                
                # Get metrics for pruned model
                pruned_metrics = pruned_data.get(device, {})
                pruned_inference = pruned_metrics.get('inference_time_ms', 0)
                pruned_rom = pruned_metrics.get('rom_size_bytes', 0)
                pruned_ram = pruned_metrics.get('ram_usage_bytes', 0)
                
                # Calculate speedup and memory savings
                speedup = ((orig_inference - pruned_inference) / orig_inference * 100) if orig_inference > 0 else 0
                rom_savings = ((orig_rom - pruned_rom) / orig_rom * 100) if orig_rom > 0 else 0
                ram_savings = ((orig_ram - pruned_ram) / orig_ram * 100) if orig_ram > 0 else 0
                
                # Add both original and pruned entries
                inference_analysis.append({
                    'dataset': dataset,
                    'activation': activation,
                    'device': device,
                    'model_type': 'original',
                    'inference_time_ms': orig_inference,
                    'rom_size_bytes': orig_rom,
                    'ram_usage_bytes': orig_ram,
                    'speedup_percent': 0,  # Original model has no speedup
                    'rom_savings_percent': 0,  # Original model has no savings
                    'ram_savings_percent': 0  # Original model has no savings
                })
                
                inference_analysis.append({
                    'dataset': dataset,
                    'activation': activation,
                    'device': device,
                    'model_type': 'pruned',
                    'inference_time_ms': pruned_inference,
                    'rom_size_bytes': pruned_rom,
                    'ram_usage_bytes': pruned_ram,
                    'speedup_percent': speedup,
                    'rom_savings_percent': rom_savings,
                    'ram_savings_percent': ram_savings
                })
    
    inference_df = pd.DataFrame(inference_analysis) if inference_analysis else pd.DataFrame()
    
    # Save inference analysis if we have data
    if len(inference_df) > 0:
        inference_path = 'results/inference_analysis.csv'
        inference_df.to_csv(inference_path, index=False)
        print(f"Inference analysis saved to {inference_path}")
        
        # Print summary of inference analysis
        print("\n=== Inference Time Analysis ===")
        for device in inference_df['device'].unique():
            print(f"\nDevice: {device}")
            device_df = inference_df[inference_df['device'] == device]
            
            # Group by dataset and activation
            for dataset in device_df['dataset'].unique():
                dataset_df = device_df[device_df['dataset'] == dataset]
                print(f"\nDataset: {dataset}")
                print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
                print(f"{'Activation':<10} | {'Orig Inf':<10} | {'Pruned Inf':<10} | {'Speedup%':<10} | {'ROM Save%':<10} | {'RAM Save%':<10}")
                print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
                
                for activation in dataset_df['activation'].unique():
                    activation_df = dataset_df[dataset_df['activation'] == activation]
                    orig = activation_df[activation_df['model_type'] == 'original']
                    pruned = activation_df[activation_df['model_type'] == 'pruned']
                    
                    if len(orig) > 0 and len(pruned) > 0:
                        print(f"{activation:<10} | "
                              f"{orig['inference_time_ms'].iloc[0]:<10.2f} | "
                              f"{pruned['inference_time_ms'].iloc[0]:<10.2f} | "
                              f"{pruned['speedup_percent'].iloc[0]:<10.2f} | "
                              f"{pruned['rom_savings_percent'].iloc[0]:<10.2f} | "
                              f"{pruned['ram_savings_percent'].iloc[0]:<10.2f}")
    
    return inference_df

def main(args):
    """Main function to process results"""
    print("Starting unified results processor...")
    
    # Find result files
    print(f"Searching for result files...")
    annotated_files, all_files = find_result_files(args.dataset)
    print(f"Found {len(annotated_files)} annotated result files")
    print(f"Found {len(all_files)} total result files")
    
    # Process each file
    results = []
    for file_info in annotated_files:
        file_path = file_info["file_path"]
        dataset = file_info["dataset"]
        activation = file_info["activation"]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract metrics
            metrics = {
                'dataset': dataset,
                'activation': activation,
                'file': file_path
            }
            
            # Add metrics from file
            if isinstance(data, dict):
                metrics.update(data)
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Save consolidated results
    if results:
        output_file = "consolidated_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved consolidated results to {output_file}")
    else:
        print("No results to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and consolidate results")
    parser.add_argument("--dataset", type=str, help="Dataset to process")
    args = parser.parse_args()
    main(args) 