#!/usr/bin/env python3
import os
import sys
import json
import glob
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

def load_results_index(dataset_name=None):
    """Load the results index CSV file, optionally filtered by dataset"""
    results_index_path = "results/results_index.csv"
    if not os.path.exists(results_index_path):
        print(f"Results index file not found: {results_index_path}")
        return None
    
    df = pd.read_csv(results_index_path)
    
    if dataset_name:
        df = df[df['dataset_name'] == dataset_name]
    
    return df

def load_activation_comparison(dataset_name):
    """Load the activation comparison CSV file for a specific dataset"""
    # Try dataset-specific activation comparison file
    dataset_csv_path = f"results/{dataset_name}_activation_comparison.csv"
    
    if os.path.exists(dataset_csv_path):
        df = pd.read_csv(dataset_csv_path)
        return df
    
    # If not found, try to extract from the general activation comparison file
    general_csv_path = "results/activation_comparison.csv"
    
    if os.path.exists(general_csv_path):
        df = pd.read_csv(general_csv_path)
        df = df[df['dataset'] == dataset_name]
        return df
    
    print(f"No activation comparison data found for dataset: {dataset_name}")
    return None

def load_profiling_data(dataset_name):
    """Load all profiling data for a specific dataset"""
    profiling_dir = Path("results/profiling")
    if not profiling_dir.exists():
        print(f"Profiling directory not found: {profiling_dir}")
        return {}
    
    profiling_files = glob.glob(f"{profiling_dir}/{dataset_name}_profiling*.json")
    profiling_data = []
    
    for file_path in profiling_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                profiling_data.append(data)
        except Exception as e:
            print(f"Error loading profiling data from {file_path}: {str(e)}")
    
    return profiling_data

def load_metrics_data(dataset_name):
    """Load all metrics data for a specific dataset"""
    metrics_dir = Path(f"results/{dataset_name}")
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return {"original": [], "pruned": []}
    
    original_metrics_files = glob.glob(f"{metrics_dir}/{dataset_name}_original_metrics_*.json")
    pruned_metrics_files = glob.glob(f"{metrics_dir}/{dataset_name}_pruned_metrics_*.json")
    
    original_metrics = []
    pruned_metrics = []
    
    for file_path in original_metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                original_metrics.append(data)
        except Exception as e:
            print(f"Error loading original metrics from {file_path}: {str(e)}")
    
    for file_path in pruned_metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                pruned_metrics.append(data)
        except Exception as e:
            print(f"Error loading pruned metrics from {file_path}: {str(e)}")
    
    return {
        "original": original_metrics,
        "pruned": pruned_metrics
    }

def crosscheck_data(dataset_name):
    """Crosscheck data from different sources for a specific dataset"""
    print(f"Crosschecking data for dataset: {dataset_name}")
    
    # Load data from different sources
    results_index_df = load_results_index(dataset_name)
    activation_comparison_df = load_activation_comparison(dataset_name)
    profiling_data = load_profiling_data(dataset_name)
    metrics_data = load_metrics_data(dataset_name)
    
    # Print summary of loaded data
    print("\n=== Data Summary ===")
    print(f"Results Index: {len(results_index_df) if results_index_df is not None else 0} entries")
    print(f"Activation Comparison: {len(activation_comparison_df) if activation_comparison_df is not None else 0} entries")
    print(f"Profiling Data: {len(profiling_data)} files")
    print(f"Original Metrics: {len(metrics_data['original'])} files")
    print(f"Pruned Metrics: {len(metrics_data['pruned'])} files")
    
    # Crosscheck activation functions
    if results_index_df is not None and activation_comparison_df is not None:
        print("\n=== Activation Function Crosscheck ===")
        
        # Get unique activation functions from results index
        results_index_activations = set(results_index_df['activation'].unique())
        print(f"Activation functions in Results Index: {results_index_activations}")
        
        # Get unique activation functions from activation comparison
        activation_comparison_activations = set(activation_comparison_df['activation'].unique())
        print(f"Activation functions in Activation Comparison: {activation_comparison_activations}")
        
        # Check for discrepancies
        if results_index_activations != activation_comparison_activations:
            print("WARNING: Discrepancy in activation functions between Results Index and Activation Comparison")
            print(f"Only in Results Index: {results_index_activations - activation_comparison_activations}")
            print(f"Only in Activation Comparison: {activation_comparison_activations - results_index_activations}")
        else:
            print("Activation functions match between Results Index and Activation Comparison")
    
    # Crosscheck accuracy values
    if results_index_df is not None and activation_comparison_df is not None:
        print("\n=== Accuracy Crosscheck ===")
        
        # For each activation function, compare accuracy values
        for activation in results_index_activations:
            results_index_accuracy = results_index_df[results_index_df['activation'] == activation]['accuracy'].mean()
            activation_comparison_row = activation_comparison_df[activation_comparison_df['activation'] == activation]
            
            if not activation_comparison_row.empty:
                activation_comparison_accuracy = activation_comparison_row['pruned_accuracy'].iloc[0]
                
                print(f"Activation: {activation}")
                print(f"  Results Index Accuracy: {results_index_accuracy:.2f}%")
                print(f"  Activation Comparison Accuracy: {activation_comparison_accuracy:.2f}%")
                
                if abs(results_index_accuracy - activation_comparison_accuracy) > 1.0:
                    print(f"  WARNING: Significant difference in accuracy values ({abs(results_index_accuracy - activation_comparison_accuracy):.2f}%)")
    
    # Crosscheck pruning rates
    if results_index_df is not None and activation_comparison_df is not None:
        print("\n=== Pruning Rate Crosscheck ===")
        
        # For each activation function, compare pruning rates
        for activation in results_index_activations:
            results_index_pruning_rate = results_index_df[results_index_df['activation'] == activation]['pruning_rate'].mean() * 100
            activation_comparison_row = activation_comparison_df[activation_comparison_df['activation'] == activation]
            
            if not activation_comparison_row.empty:
                activation_comparison_pruning_rate = activation_comparison_row['pruning_rate'].iloc[0] * 100
                
                print(f"Activation: {activation}")
                print(f"  Results Index Pruning Rate: {results_index_pruning_rate:.2f}%")
                print(f"  Activation Comparison Pruning Rate: {activation_comparison_pruning_rate:.2f}%")
                
                if abs(results_index_pruning_rate - activation_comparison_pruning_rate) > 1.0:
                    print(f"  WARNING: Significant difference in pruning rates ({abs(results_index_pruning_rate - activation_comparison_pruning_rate):.2f}%)")
    
    # Crosscheck profiling data
    if profiling_data and metrics_data['pruned']:
        print("\n=== Profiling Data Crosscheck ===")
        
        # Check if profiling data matches metrics data
        profiling_activations = set(data.get('activation', '') for data in profiling_data)
        metrics_activations = set()
        
        for metric in metrics_data['pruned']:
            if 'activation' in metric:
                metrics_activations.add(metric['activation'])
        
        print(f"Activation functions in Profiling Data: {profiling_activations}")
        print(f"Activation functions in Metrics Data: {metrics_activations}")
        
        if profiling_activations != metrics_activations:
            print("WARNING: Discrepancy in activation functions between Profiling Data and Metrics Data")
            print(f"Only in Profiling Data: {profiling_activations - metrics_activations}")
            print(f"Only in Metrics Data: {metrics_activations - profiling_activations}")
        else:
            print("Activation functions match between Profiling Data and Metrics Data")
    
    return {
        "results_index": results_index_df,
        "activation_comparison": activation_comparison_df,
        "profiling_data": profiling_data,
        "metrics_data": metrics_data
    }

def main():
    """Main function to run the crosscheck"""
    parser = argparse.ArgumentParser(description='Load and crosscheck data for a specific dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name to analyze')
    parser.add_argument('--output', type=str, help='Output path for the crosscheck report')
    args = parser.parse_args()
    
    # Run crosscheck
    data = crosscheck_data(args.dataset)
    
    # Generate report if output path is provided
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Dataset Crosscheck Report: {args.dataset}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Data Summary\n\n")
            f.write(f"- Results Index: {len(data['results_index']) if data['results_index'] is not None else 0} entries\n")
            f.write(f"- Activation Comparison: {len(data['activation_comparison']) if data['activation_comparison'] is not None else 0} entries\n")
            f.write(f"- Profiling Data: {len(data['profiling_data'])} files\n")
            f.write(f"- Original Metrics: {len(data['metrics_data']['original'])} files\n")
            f.write(f"- Pruned Metrics: {len(data['metrics_data']['pruned'])} files\n\n")
            
            # Write activation function comparison
            if data['results_index'] is not None and data['activation_comparison'] is not None:
                f.write("## Activation Function Comparison\n\n")
                
                results_index_activations = set(data['results_index']['activation'].unique())
                activation_comparison_activations = set(data['activation_comparison']['activation'].unique())
                
                f.write("| Source | Activation Functions |\n")
                f.write("|--------|---------------------|\n")
                f.write(f"| Results Index | {', '.join(results_index_activations)} |\n")
                f.write(f"| Activation Comparison | {', '.join(activation_comparison_activations)} |\n\n")
                
                if results_index_activations != activation_comparison_activations:
                    f.write("**WARNING**: Discrepancy in activation functions between Results Index and Activation Comparison\n\n")
                    f.write(f"- Only in Results Index: {', '.join(results_index_activations - activation_comparison_activations)}\n")
                    f.write(f"- Only in Activation Comparison: {', '.join(activation_comparison_activations - results_index_activations)}\n\n")
                else:
                    f.write("Activation functions match between Results Index and Activation Comparison\n\n")
            
            # Write accuracy comparison
            if data['results_index'] is not None and data['activation_comparison'] is not None:
                f.write("## Accuracy Comparison\n\n")
                
                f.write("| Activation | Results Index Accuracy | Activation Comparison Accuracy | Difference |\n")
                f.write("|------------|------------------------|------------------------------|------------|\n")
                
                for activation in results_index_activations:
                    results_index_accuracy = data['results_index'][data['results_index']['activation'] == activation]['accuracy'].mean()
                    activation_comparison_row = data['activation_comparison'][data['activation_comparison']['activation'] == activation]
                    
                    if not activation_comparison_row.empty:
                        activation_comparison_accuracy = activation_comparison_row['pruned_accuracy'].iloc[0]
                        difference = abs(results_index_accuracy - activation_comparison_accuracy)
                        
                        f.write(f"| {activation} | {results_index_accuracy:.2f}% | {activation_comparison_accuracy:.2f}% | {difference:.2f}% |\n")
                
                f.write("\n")
            
            # Write pruning rate comparison
            if data['results_index'] is not None and data['activation_comparison'] is not None:
                f.write("## Pruning Rate Comparison\n\n")
                
                f.write("| Activation | Results Index Pruning Rate | Activation Comparison Pruning Rate | Difference |\n")
                f.write("|------------|----------------------------|----------------------------------|------------|\n")
                
                for activation in results_index_activations:
                    results_index_pruning_rate = data['results_index'][data['results_index']['activation'] == activation]['pruning_rate'].mean() * 100
                    activation_comparison_row = data['activation_comparison'][data['activation_comparison']['activation'] == activation]
                    
                    if not activation_comparison_row.empty:
                        activation_comparison_pruning_rate = activation_comparison_row['pruning_rate'].iloc[0] * 100
                        difference = abs(results_index_pruning_rate - activation_comparison_pruning_rate)
                        
                        f.write(f"| {activation} | {results_index_pruning_rate:.2f}% | {activation_comparison_pruning_rate:.2f}% | {difference:.2f}% |\n")
                
                f.write("\n")
            
            # Write profiling data summary
            if data['profiling_data']:
                f.write("## Profiling Data Summary\n\n")
                
                f.write("| Activation | Device | Inference Time (ms) | ROM Size (bytes) | RAM Usage (bytes) |\n")
                f.write("|------------|--------|---------------------|------------------|-------------------|\n")
                
                for profiling in data['profiling_data']:
                    activation = profiling.get('activation', '')
                    
                    for model_type in ['original', 'pruned']:
                        if model_type in profiling:
                            for device, metrics in profiling[model_type].items():
                                inference_time = metrics.get('inference_time_ms', 0)
                                rom_size = metrics.get('rom_size_bytes', 0)
                                ram_usage = metrics.get('ram_usage_bytes', 0)
                                
                                f.write(f"| {activation} ({model_type}) | {device} | {inference_time:.2f} | {rom_size} | {ram_usage} |\n")
                
                f.write("\n")
        
        print(f"Crosscheck report saved to: {args.output}")
    
    return data

if __name__ == "__main__":
    main() 