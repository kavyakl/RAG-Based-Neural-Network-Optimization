#!/usr/bin/env python3

import os
import json
import pandas as pd
import glob
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any

def find_result_files(dataset_name: Optional[str] = None, activation: Optional[str] = None) -> List[str]:
    """Find all result files matching the criteria"""
    results_dir = Path("results")
    
    # Construct pattern based on provided filters
    if dataset_name and activation:
        pattern = f"{results_dir}/{dataset_name}/{activation}/{dataset_name}/*_metrics_*.json"
    elif dataset_name:
        pattern = f"{results_dir}/{dataset_name}/*/{dataset_name}/*_metrics_*.json"
    elif activation:
        pattern = f"{results_dir}/*/{activation}/*/*_metrics_*.json"
    else:
        pattern = f"{results_dir}/*/*/*/*_metrics_*.json"
    
    files = glob.glob(pattern)
    
    # Also check for results in the flat structure
    if dataset_name:
        flat_pattern = f"{results_dir}/{dataset_name}/*_metrics_*.json"
        files.extend(glob.glob(flat_pattern))
    
    return files

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

def extract_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """Extract metadata from result file path and content"""
    try:
        # Extract info from path
        parts = Path(file_path).parts
        dataset_name = None
        activation = None
        model_type = 'unknown'
        
        # Determine model type based on filename pattern
        if 'pruned' in file_path:
            model_type = 'pruned'
        elif 'original' in file_path:
            model_type = 'original'
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract dataset name - first try from file path
        file_name = Path(file_path).name
        dataset_parts = file_name.split('_')
        
        # More robust dataset name extraction
        # First look in the filename for the dataset name
        for part in parts:
            if part not in ['results', 'relu', 'sigmoid', 'tanh', 'original', 'pruned', 'metrics']:
                if part not in ['json', 'csv', 'py', 'yaml', 'md']:
                    # Use the most specific dataset name (longest match)
                    if dataset_name is None or len(part) > len(dataset_name):
                        dataset_name = part

        # Normalize dataset name - often there's an underscore in dataset_name
        if dataset_name and '_' in dataset_name:
            dataset_name = dataset_name.replace('_', '')
            
        # If we're in a specific dataset folder, use that as the dataset name
        if len(parts) >= 2 and parts[1] not in ['relu', 'sigmoid', 'tanh', 'original', 'pruned']:
            dataset_name = parts[1]
        
        # Try to extract activation from data more aggressively
        # First check if it's directly in the data
        if 'activation' in data:
            activation = data['activation']
        elif model_type == 'original' and 'pruned' in file_path:
            # For original models associated with pruned models, check if there's a matching pruned file
            # This would be handled by the timestamp matching logic in create_activation_comparison
            pass
        else:
            # Look for activation in path components
            for part in parts:
                if part in ['relu', 'sigmoid', 'tanh']:
                    activation = part
                    break
            
            # Still not found? Search deeper in the data structure
            if activation is None and isinstance(data, dict):
                # Search in model_config if it exists
                if 'model_config' in data and isinstance(data['model_config'], dict):
                    if 'activation' in data['model_config']:
                        activation = data['model_config']['activation']
                
                # Search in layers if they exist
                if activation is None and 'layers' in data and isinstance(data['layers'], list):
                    for layer in data['layers']:
                        if isinstance(layer, dict) and 'activation' in layer:
                            activation = layer['activation']
                            break
                
                # Recursive search for activation in nested dictionaries
                if activation is None:
                    for k, v in data.items():
                        if k.endswith('activation') and isinstance(v, str) and v in ['relu', 'sigmoid', 'tanh']:
                            activation = v
                            break
                        # Check one level deeper
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                if (k2.endswith('activation') or k2 == 'activation') and v2 in ['relu', 'sigmoid', 'tanh']:
                                    activation = v2
                                    break
        
        # If activation is still unknown, skip this file
        if activation is None or activation == 'unknown':
            print(f"Skipping file with unknown activation: {file_path}")
            return None
        
        # Get accuracy values - try different fields that might exist
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
        
        # Default other metadata values
        hidden_neurons = 0
        pruned_neurons = 0
        pruning_rate = 0
        accuracy_drop = 0
        
        # Extract detailed metrics if available
        if model_type == 'pruned':
            hidden_neurons = data.get('pruned_hidden_neurons', 0)
            # Calculate neurons pruned if we have original and pruned counts
            if 'original_hidden_neurons' in data and 'pruned_hidden_neurons' in data:
                pruned_neurons = data['original_hidden_neurons'] - data['pruned_hidden_neurons']
            else:
                pruned_neurons = data.get('neurons_pruned', 0)
                
            # Calculate pruning rate
            if pruned_neurons > 0 and 'original_hidden_neurons' in data and data['original_hidden_neurons'] > 0:
                pruning_rate = pruned_neurons / data['original_hidden_neurons']
            else:
                pruning_rate = data.get('pruning_rate', 0)
                
            # Get accuracy drop
            if 'original_accuracy' in data and 'pruned_accuracy' in data:
                accuracy_drop = data['original_accuracy'] - data['pruned_accuracy']
            else:
                accuracy_drop = data.get('accuracy_drop', 0)
        else:  # original
            hidden_neurons = data.get('hidden_neurons', 
                            data.get('original_hidden_neurons', 0))
            
        # Create metadata dictionary
        metadata = {
            'file_path': file_path,
            'dataset_name': dataset_name,
            'activation': activation,
            'model_type': model_type,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'accuracy': normalized_accuracy,  # Normalize to percentage
            'hidden_neurons': hidden_neurons,
            'pruned_neurons': pruned_neurons,
            'pruning_rate': pruning_rate,
            'accuracy_drop': accuracy_drop
        }
        
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

def create_activation_comparison(df: pd.DataFrame, dataset_name: Optional[str] = None, profiling_data: Optional[Dict] = None) -> pd.DataFrame:
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
    
    # Try to link original models to their corresponding activation
    # First, create a map of original models by timestamp
    original_map = {}
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        # Get original models from this dataset
        originals = dataset_df[dataset_df['model_type'] == 'original']
        
        # Look at corresponding pruned models to get activation
        for _, orig in originals.iterrows():
            orig_timestamp = orig['timestamp']
            # Find pruned models with similar timestamp (within 1 minute)
            # First extract just the date and first part of time
            time_base = orig_timestamp.split(':')[0] + ':'
            
            matched_pruned = dataset_df[
                (dataset_df['model_type'] == 'pruned') & 
                (dataset_df['timestamp'].str.startswith(time_base))
            ]
            
            if len(matched_pruned) > 0:
                # Get activation from matched pruned model
                activation = matched_pruned.iloc[0]['activation']
                # Skip if activation is unknown
                if activation == 'unknown':
                    continue
                # Store original model with this activation
                if dataset not in original_map:
                    original_map[dataset] = {}
                if activation not in original_map[dataset]:
                    original_map[dataset][activation] = []
                
                original_map[dataset][activation].append(orig)
    
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        activations = dataset_df['activation'].unique().tolist()
        # Remove unknown activations
        activations = [a for a in activations if a != 'unknown']
        
        print(f"\nDataset: {dataset}")
        print(f"  Activations found: {activations}")
        
        # For each activation, find original and pruned models
        for activation in activations:
            activation_df = dataset_df[dataset_df['activation'] == activation]
            
            # For original models, try to use the mapped ones
            if dataset in original_map and activation in original_map[dataset]:
                original_models = pd.DataFrame(original_map[dataset][activation])
            else:
                # Fallback to direct filter for backward compatibility
                original_models = activation_df[activation_df['model_type'] == 'original']
                # If no original models with this activation, try to match by timestamp
                if len(original_models) == 0:
                    # Get pruned models with this activation
                    pruned_with_activation = activation_df[activation_df['model_type'] == 'pruned']
                    if len(pruned_with_activation) > 0:
                        # Get all original models for this dataset
                        all_originals = dataset_df[dataset_df['model_type'] == 'original']
                        # Match by similar timestamp
                        matched_originals = []
                        for _, pruned in pruned_with_activation.iterrows():
                            pruned_time = pruned['timestamp'].split(':')[0] + ':'
                            for _, orig in all_originals.iterrows():
                                if orig['timestamp'].startswith(pruned_time):
                                    matched_originals.append(orig)
                        if matched_originals:
                            original_models = pd.DataFrame(matched_originals)
            
            pruned_models = activation_df[activation_df['model_type'] == 'pruned']
            
            print(f"  Activation: {activation}")
            print(f"    Original models: {len(original_models) if 'original_models' in locals() else 0}")
            print(f"    Pruned models: {len(pruned_models)}")
            
            # Skip if we don't have any models for this activation
            if (('original_models' not in locals() or len(original_models) == 0) and 
                len(pruned_models) == 0):
                continue
                
            # Get most recent of each if available
            latest_orig_acc = 0
            latest_pruned_acc = 0
            accuracy_drop = 0
            pruning_rate = 0
            neurons_removed = 0
            original_neurons = 0
            pruned_neurons = 0
            
            # Memory metrics (default to 0)
            orig_rom_bytes = 0
            orig_ram_bytes = 0
            pruned_rom_bytes = 0
            pruned_ram_bytes = 0
            orig_inference_ms = 0
            pruned_inference_ms = 0
            
            # Try to get original model data
            if 'original_models' in locals() and len(original_models) > 0:
                # If we have multiple, sort by timestamp and get the latest
                if hasattr(original_models, 'sort_values'):
                    latest_orig = original_models.sort_values('timestamp').iloc[-1]
                else:
                    latest_orig = original_models.iloc[-1]
                
                latest_orig_acc = latest_orig['accuracy']
                original_neurons = latest_orig['hidden_neurons']
            
            # Try to get pruned model data
            if len(pruned_models) > 0:
                latest_pruned = pruned_models.sort_values('timestamp').iloc[-1]
                latest_pruned_acc = latest_pruned['accuracy']
                pruned_neurons = latest_pruned['hidden_neurons']
                
                # Get accuracy drop either from the file or calculate it
                if 'accuracy_drop' in latest_pruned and latest_pruned['accuracy_drop'] > 0:
                    accuracy_drop = latest_pruned['accuracy_drop']
                elif latest_orig_acc > 0:
                    accuracy_drop = abs(latest_orig_acc - latest_pruned_acc)
                
                # Get pruning rate
                if 'pruning_rate' in latest_pruned and latest_pruned['pruning_rate'] > 0:
                    pruning_rate = latest_pruned['pruning_rate']
                elif original_neurons > 0:
                    pruning_rate = (original_neurons - pruned_neurons) / original_neurons
                
                # Get neurons removed
                if 'pruned_neurons' in latest_pruned and latest_pruned['pruned_neurons'] > 0:
                    neurons_removed = latest_pruned['pruned_neurons']
                elif original_neurons > 0:
                    neurons_removed = original_neurons - pruned_neurons
                
                # If we don't have original model data, try to extract from pruned model
                if latest_orig_acc == 0:
                    if 'original_accuracy' in latest_pruned:
                        latest_orig_acc = latest_pruned['original_accuracy']
                    elif 'accuracy_drop' in latest_pruned:
                        latest_orig_acc = latest_pruned_acc + latest_pruned['accuracy_drop']
            
            # Try to get memory metrics from profiling data
            if profiling_data and dataset in profiling_data and activation in profiling_data[dataset]:
                if 'original' in profiling_data[dataset][activation]:
                    orig_rom_bytes = profiling_data[dataset][activation]['original'].get('rom_size_bytes', 0)
                    orig_ram_bytes = profiling_data[dataset][activation]['original'].get('ram_usage_bytes', 0)
                    orig_inference_ms = profiling_data[dataset][activation]['original'].get('inference_time_ms', 0)
                
                if 'pruned' in profiling_data[dataset][activation]:
                    pruned_rom_bytes = profiling_data[dataset][activation]['pruned'].get('rom_size_bytes', 0)
                    pruned_ram_bytes = profiling_data[dataset][activation]['pruned'].get('ram_usage_bytes', 0)
                    pruned_inference_ms = profiling_data[dataset][activation]['pruned'].get('inference_time_ms', 0)
            
            comparison.append({
                'dataset': dataset,
                'activation': activation,
                'original_accuracy': latest_orig_acc,
                'pruned_accuracy': latest_pruned_acc,
                'accuracy_drop': accuracy_drop,
                'pruning_rate': pruning_rate,
                'original_neurons': original_neurons,
                'pruned_neurons': pruned_neurons,
                'neurons_removed': neurons_removed,
                'original_rom_bytes': orig_rom_bytes,
                'original_ram_bytes': orig_ram_bytes,
                'pruned_rom_bytes': pruned_rom_bytes,
                'pruned_ram_bytes': pruned_ram_bytes,
                'original_inference_ms': orig_inference_ms,
                'pruned_inference_ms': pruned_inference_ms,
                'rom_reduction': (orig_rom_bytes - pruned_rom_bytes) if orig_rom_bytes > 0 else 0,
                'ram_reduction': (orig_ram_bytes - pruned_ram_bytes) if orig_ram_bytes > 0 else 0,
                'inference_speedup': ((orig_inference_ms - pruned_inference_ms) / orig_inference_ms * 100) if orig_inference_ms > 0 else 0,
                'rom_reduction_percent': ((orig_rom_bytes - pruned_rom_bytes) / orig_rom_bytes * 100) if orig_rom_bytes > 0 else 0,
                'ram_reduction_percent': ((orig_ram_bytes - pruned_ram_bytes) / orig_ram_bytes * 100) if orig_ram_bytes > 0 else 0
            })
    
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

def generate_report(results_df: pd.DataFrame, activation_comparison_df: pd.DataFrame, 
                   accuracy_discrepancies: List[Tuple], profiling_discrepancies: List[Tuple]) -> str:
    """Generate a detailed report of all discrepancies found"""
    report = []
    report.append("# Data Crosscheck Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add dataset summary
    report.append("## Dataset Summary")
    report.append("| Dataset | Activations | Original Accuracy | Pruned Accuracy | Accuracy Drop | Pruning Rate | Original Neurons | Pruned Neurons |")
    report.append("|---------|-------------|------------------|-----------------|---------------|--------------|------------------|----------------|")
    
    # Group by dataset
    for dataset, group in activation_comparison_df.groupby('dataset'):
        activations = ', '.join(group['activation'].unique())
        orig_acc = group['original_accuracy'].mean()
        pruned_acc = group['pruned_accuracy'].mean()
        acc_drop = group['accuracy_drop'].mean()
        pruning_rate = group['pruning_rate'].mean()
        orig_neurons = group['original_neurons'].mean()
        pruned_neurons = group['pruned_neurons'].mean()
        
        report.append(f"| {dataset} | {activations} | {orig_acc:.2f} | {pruned_acc:.2f} | {acc_drop:.2f} | {pruning_rate:.2f} | {orig_neurons:.0f} | {pruned_neurons:.0f} |")
    
    if not accuracy_discrepancies and not profiling_discrepancies:
        report.append("\n✅ No discrepancies found! All data is consistent across files.")
        return "\n".join(report)
    
    if accuracy_discrepancies:
        report.append("\n## Accuracy Metric Discrepancies")
        report.append("| Dataset | Model Type | Metrics File | Results Index | Difference |")
        report.append("|---------|------------|--------------|---------------|------------|")
        
        for dataset, model_type, metrics_acc, results_acc in accuracy_discrepancies:
            diff = abs(metrics_acc - results_acc)
            report.append(f"| {dataset} | {model_type} | {metrics_acc:.4f} | {results_acc:.4f} | {diff:.4f} |")
    
    if profiling_discrepancies:
        report.append("\n## Profiling Metric Discrepancies")
        report.append("| Dataset | Device | Metric Type | Profiling File | Comparison File | Difference |")
        report.append("|---------|--------|-------------|----------------|-----------------|------------|")
        
        for dataset, device, metric_type, profiling_val, comparison_val in profiling_discrepancies:
            diff = abs(profiling_val - comparison_val)
            report.append(f"| {dataset} | {device} | {metric_type} | {profiling_val:.2f} | {comparison_val:.2f} | {diff:.2f} |")
    
    return "\n".join(report)

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

def main():
    """Main function to process and validate results"""
    parser = argparse.ArgumentParser(description='Process and validate neural network results')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], help='Filter by activation function')
    parser.add_argument('--output', type=str, default='results/results_index.csv', help='Output path for tagged results')
    parser.add_argument('--report', type=str, default='reports/crosscheck_report.md', help='Output path for the validation report')
    parser.add_argument('--memory', action='store_true', help='Include memory metrics in the output')
    parser.add_argument('--inference', action='store_true', help='Analyze inference time')
    args = parser.parse_args()
    
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    
    # Find result files
    print(f"Searching for result files...")
    result_files = find_result_files(args.dataset, args.activation)
    print(f"Found {len(result_files)} result files")
    
    if not result_files:
        print("No result files found. Make sure you've run the experiments.")
        return
    
    # Create results table
    results_df = create_results_table(result_files)
    
    if results_df.empty:
        print("Could not create results table.")
        return
    
    # Save results index
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Results tagged and saved to {args.output}")
    
    # Find profiling files
    profiling_files = find_profiling_files(args.dataset, args.activation)
    print(f"Found {len(profiling_files)} profiling files")
    
    # Extract profiling data
    profiling_data = extract_profiling_data(profiling_files, args.dataset, args.activation)
    
    # Create activation comparison
    activation_comparison_df = create_activation_comparison(results_df, args.dataset, profiling_data)
    
    # If inference analysis is requested, do that and exit
    if args.inference:
        print("\nAnalyzing inference time...")
        inference_df = analyze_inference_time(profiling_data, activation_comparison_df)
        if inference_df.empty:
            print("No inference data found.")
        return
    
    # Load metrics data for crosschecking
    metrics_data = load_metrics_data(args.dataset)
    
    # Perform crosschecks
    print("Performing crosschecks...")
    accuracy_discrepancies = crosscheck_accuracy_metrics(results_df, metrics_data)
    profiling_discrepancies = crosscheck_profiling_metrics(profiling_data, activation_comparison_df)
    
    # Generate and save report
    report = generate_report(results_df, activation_comparison_df, accuracy_discrepancies, profiling_discrepancies)
    with open(args.report, 'w') as f:
        f.write(report)
    
    print(f"\nCrosscheck complete! Report saved to {args.report}")
    if accuracy_discrepancies or profiling_discrepancies:
        print(f"Found {len(accuracy_discrepancies)} accuracy discrepancies and {len(profiling_discrepancies)} profiling discrepancies.")
    else:
        print("No discrepancies found!")
    
    # Print summary
    print_summary(results_df, activation_comparison_df)

if __name__ == "__main__":
    main() 