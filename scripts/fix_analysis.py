#!/usr/bin/env python3
import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_results_index():
    """Load the results index CSV file"""
    results_index_path = "results/results_index.csv"
    if not os.path.exists(results_index_path):
        print(f"Results index file not found: {results_index_path}")
        return None
    
    df = pd.read_csv(results_index_path)
    return df

def load_profiling_data():
    """Load all profiling data from the profiling directory"""
    profiling_dir = Path("results/profiling")
    if not profiling_dir.exists():
        print(f"Profiling directory not found: {profiling_dir}")
        return {}
    
    profiling_files = glob.glob(f"{profiling_dir}/*profiling*.json")
    profiling_data = {}
    
    for file_path in profiling_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            dataset = data.get('dataset', '')
            activation = data.get('activation', '')
            
            if dataset and activation:
                if dataset not in profiling_data:
                    profiling_data[dataset] = {}
                
                if activation not in profiling_data[dataset]:
                    profiling_data[dataset][activation] = {}
                
                # Store original and pruned model data
                if 'original' in data:
                    profiling_data[dataset][activation]['original'] = data['original']
                
                if 'pruned' in data:
                    profiling_data[dataset][activation]['pruned'] = data['pruned']
        except Exception as e:
            print(f"Error loading profiling data from {file_path}: {str(e)}")
    
    return profiling_data

def analyze_activation_functions(df):
    """Analyze the performance of different activation functions"""
    activation_analysis = {
        'summary': {},
        'dataset_details': {},
        'recommendations': {}
    }
    
    # Get all activation functions
    activation_functions = df['activation'].unique().tolist()
    
    # Initialize summary for each activation
    for activation in activation_functions:
        activation_analysis['summary'][activation] = {
            'datasets': 0,
            'best_performing_datasets': [],
            'worst_performing_datasets': [],
            'avg_accuracy': 0,
            'avg_accuracy_drop': 0
        }
    
    # Analyze each activation function
    for activation in activation_functions:
        activation_df = df[df['activation'] == activation]
        
        # Calculate average accuracy and accuracy drop
        avg_accuracy = activation_df['accuracy'].mean()
        avg_accuracy_drop = activation_df['accuracy_drop'].mean()
        
        activation_analysis['summary'][activation]['avg_accuracy'] = avg_accuracy
        activation_analysis['summary'][activation]['avg_accuracy_drop'] = avg_accuracy_drop
        activation_analysis['summary'][activation]['datasets'] = len(activation_df['dataset_name'].unique())
        
        # Find best and worst performing datasets
        for dataset in activation_df['dataset_name'].unique():
            dataset_df = activation_df[activation_df['dataset_name'] == dataset]
            accuracy = dataset_df['accuracy'].iloc[0]
            
            if accuracy >= 90:
                activation_analysis['summary'][activation]['best_performing_datasets'].append(dataset)
            elif accuracy <= 60:
                activation_analysis['summary'][activation]['worst_performing_datasets'].append(dataset)
    
    # Generate recommendations
    for activation in activation_functions:
        summary = activation_analysis['summary'][activation]
        
        if summary['avg_accuracy'] >= 85:
            recommendation = f"{activation.capitalize()} performs well across datasets with an average accuracy of {summary['avg_accuracy']:.2f}%. It's particularly effective for {', '.join(summary['best_performing_datasets'])}."
        elif summary['avg_accuracy'] >= 70:
            recommendation = f"{activation.capitalize()} has moderate performance with an average accuracy of {summary['avg_accuracy']:.2f}%. Consider using it for {', '.join(summary['best_performing_datasets'])}."
        else:
            recommendation = f"{activation.capitalize()} has poor performance with an average accuracy of {summary['avg_accuracy']:.2f}%. Avoid using it for most datasets."
        
        activation_analysis['recommendations'][activation] = recommendation
    
    # Analyze each dataset
    for dataset in df['dataset_name'].unique():
        dataset_df = df[df['dataset_name'] == dataset]
        
        activation_analysis['dataset_details'][dataset] = {
            'activation_comparison': {}
        }
        
        for activation in activation_functions:
            activation_df = dataset_df[dataset_df['activation'] == activation]
            if not activation_df.empty:
                activation_analysis['dataset_details'][dataset]['activation_comparison'][activation] = {
                    'accuracy': activation_df['accuracy'].iloc[0],
                    'accuracy_drop': activation_df['accuracy_drop'].iloc[0],
                    'pruning_rate': activation_df['pruning_rate'].iloc[0]
                }
    
    return activation_analysis

def analyze_pruning_impact(df):
    """Analyze the impact of pruning on model performance"""
    pruning_analysis = {
        'summary': {
            'avg_pruning_rate': 0,
            'avg_accuracy_drop': 0,
            'datasets_with_positive_impact': [],
            'datasets_with_negative_impact': []
        },
        'dataset_details': {},
        'recommendations': {}
    }
    
    # Calculate average pruning rate and accuracy drop
    pruning_analysis['summary']['avg_pruning_rate'] = df['pruning_rate'].mean() * 100
    pruning_analysis['summary']['avg_accuracy_drop'] = df['accuracy_drop'].mean()
    
    # Analyze each dataset
    for dataset in df['dataset_name'].unique():
        dataset_df = df[df['dataset_name'] == dataset]
        
        pruning_analysis['dataset_details'][dataset] = {
            'best_pruning': None,
            'pruning_details': {}
        }
        
        # Find best pruning configuration
        best_pruning = None
        best_score = float('-inf')
        
        for _, row in dataset_df.iterrows():
            pruning_rate = row['pruning_rate']
            accuracy_drop = row['accuracy_drop']
            
            # Score based on pruning rate and accuracy drop
            # Higher pruning rate is better, but we want to minimize accuracy drop
            score = pruning_rate * (1 - abs(accuracy_drop) / 100)
            
            if score > best_score:
                best_score = score
                best_pruning = {
                    'activation': row['activation'],
                    'pruning_rate': pruning_rate,
                    'accuracy_drop': accuracy_drop
                }
        
        pruning_analysis['dataset_details'][dataset]['best_pruning'] = best_pruning
        
        # Store pruning details for each activation
        for activation in dataset_df['activation'].unique():
            activation_df = dataset_df[dataset_df['activation'] == activation]
            if not activation_df.empty:
                pruning_analysis['dataset_details'][dataset]['pruning_details'][activation] = {
                    'pruning_rate': activation_df['pruning_rate'].iloc[0],
                    'accuracy_drop': activation_df['accuracy_drop'].iloc[0],
                    'neurons_removed': activation_df['pruned_neurons'].iloc[0],
                    'original_neurons': activation_df['hidden_neurons'].iloc[0] + activation_df['pruned_neurons'].iloc[0]
                }
        
        # Determine if pruning had a positive or negative impact
        if best_pruning:
            accuracy_drop = best_pruning['accuracy_drop']
            
            if accuracy_drop <= 5:  # Less than 5% accuracy drop
                pruning_analysis['summary']['datasets_with_positive_impact'].append(dataset)
            elif accuracy_drop > 15:  # More than 15% accuracy drop
                pruning_analysis['summary']['datasets_with_negative_impact'].append(dataset)
            
            # Generate recommendation
            activation = best_pruning['activation']
            pruning_rate = best_pruning['pruning_rate'] * 100
            
            if accuracy_drop <= 5:
                recommendation = f"For {dataset}, pruning with {activation} activation achieved {pruning_rate:.1f}% reduction in model size with minimal accuracy drop ({accuracy_drop:.2f}%)."
            elif accuracy_drop <= 15:
                recommendation = f"For {dataset}, pruning with {activation} activation achieved {pruning_rate:.1f}% reduction in model size with moderate accuracy drop ({accuracy_drop:.2f}%)."
            else:
                recommendation = f"For {dataset}, pruning with {activation} activation achieved {pruning_rate:.1f}% reduction in model size but with significant accuracy drop ({accuracy_drop:.2f}%). Consider using a different activation function or pruning less aggressively."
            
            pruning_analysis['recommendations'][dataset] = recommendation
    
    return pruning_analysis

def analyze_device_performance(profiling_data):
    """Analyze device performance metrics"""
    device_analysis = {
        'summary': {
            'devices': [],
            'best_performing_models': []
        },
        'dataset_details': {},
        'recommendations': {}
    }
    
    # Collect all devices
    devices = set()
    for dataset, dataset_data in profiling_data.items():
        for activation, activation_data in dataset_data.items():
            for model_type, device_data in activation_data.items():
                for device in device_data.keys():
                    devices.add(device)
    
    device_analysis['summary']['devices'] = list(devices)
    
    # Analyze each dataset
    for dataset, dataset_data in profiling_data.items():
        device_analysis['dataset_details'][dataset] = {
            'device_data': {},
            'best_device': None,
            'best_model': None
        }
        
        best_inference_time = float('inf')
        
        # Analyze each activation function
        for activation, activation_data in dataset_data.items():
            for model_type, device_data in activation_data.items():
                for device, metrics in device_data.items():
                    # Initialize device data if not exists
                    if device not in device_analysis['dataset_details'][dataset]['device_data']:
                        device_analysis['dataset_details'][dataset]['device_data'][device] = {
                            'models': []
                        }
                    
                    # Extract metrics
                    inference_time = metrics.get('inference_time_ms', 0)
                    rom_size = metrics.get('rom_size_bytes', 0)
                    ram_usage = metrics.get('ram_usage_bytes', 0)
                    
                    # Store model data
                    model_data = {
                        'activation': activation,
                        'model_type': model_type,
                        'inference_time_ms': inference_time,
                        'rom_size_bytes': rom_size,
                        'ram_usage_bytes': ram_usage
                    }
                    
                    device_analysis['dataset_details'][dataset]['device_data'][device]['models'].append(model_data)
                    
                    # Track best performing model
                    if inference_time > 0 and inference_time < best_inference_time:
                        best_inference_time = inference_time
                        device_analysis['dataset_details'][dataset]['best_device'] = device
                        device_analysis['dataset_details'][dataset]['best_model'] = {
                            'activation': activation,
                            'model_type': model_type,
                            'inference_time_ms': inference_time,
                            'rom_size_bytes': rom_size,
                            'ram_usage_bytes': ram_usage
                        }
        
        # Add to summary if we found a best model
        if device_analysis['dataset_details'][dataset]['best_model']:
            device_analysis['summary']['best_performing_models'].append({
                'dataset': dataset,
                'device': device_analysis['dataset_details'][dataset]['best_device'],
                'activation': device_analysis['dataset_details'][dataset]['best_model']['activation'],
                'model_type': device_analysis['dataset_details'][dataset]['best_model']['model_type'],
                'inference_time_ms': device_analysis['dataset_details'][dataset]['best_model']['inference_time_ms']
            })
    
    # Generate recommendations
    for dataset, dataset_data in device_analysis['dataset_details'].items():
        if dataset_data['best_model']:
            device = dataset_data['best_device']
            model = dataset_data['best_model']
            
            recommendation = f"For {dataset}, the best performing model on {device} uses {model['activation']} activation and {model['model_type']} model type, achieving {model['inference_time_ms']:.2f}ms inference time with {model['rom_size_bytes']} bytes ROM and {model['ram_usage_bytes']} bytes RAM."
            
            device_analysis['recommendations'][dataset] = recommendation
    
    return device_analysis

def generate_comprehensive_report(df, activation_analysis, pruning_analysis, device_analysis):
    """Generate a comprehensive report combining all analyses"""
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Generate report path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/comprehensive_analysis_{timestamp}.md'
    
    with open(report_path, 'w') as f:
        # Write header
        f.write('# Neural Network Optimization Analysis Report\n\n')
        f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        # Write summary
        f.write('## Executive Summary\n\n')
        f.write(f'This report analyzes {len(df["dataset_name"].unique())} datasets with {len(df)} models using {", ".join(df["activation"].unique())} activation functions.\n\n')
        
        # Write activation function analysis
        f.write('## Activation Function Analysis\n\n')
        f.write('### Summary\n\n')
        f.write('| Activation | Datasets | Avg Accuracy | Avg Accuracy Drop | Best Datasets |\n')
        f.write('|------------|----------|--------------|-------------------|---------------|\n')
        
        for activation, summary in activation_analysis['summary'].items():
            best_datasets = ', '.join(summary['best_performing_datasets'][:3])
            if len(summary['best_performing_datasets']) > 3:
                best_datasets += '...'
                
            f.write(f"| {activation} | {summary['datasets']} | {summary['avg_accuracy']:.2f}% | {summary['avg_accuracy_drop']:.2f}% | {best_datasets} |\n")
        
        f.write('\n### Recommendations\n\n')
        for activation, recommendation in activation_analysis['recommendations'].items():
            f.write(f'**{activation.capitalize()}**: {recommendation}\n\n')
        
        # Write pruning impact analysis
        f.write('## Pruning Impact Analysis\n\n')
        f.write('### Summary\n\n')
        f.write(f'Average Pruning Rate: {pruning_analysis["summary"]["avg_pruning_rate"]:.2f}%\n')
        f.write(f'Average Accuracy Drop: {pruning_analysis["summary"]["avg_accuracy_drop"]:.2f}%\n\n')
        
        f.write('Datasets with Positive Impact (≤5% accuracy drop):\n')
        for dataset in pruning_analysis['summary']['datasets_with_positive_impact']:
            f.write(f'- {dataset}\n')
        
        f.write('\nDatasets with Negative Impact (>15% accuracy drop):\n')
        for dataset in pruning_analysis['summary']['datasets_with_negative_impact']:
            f.write(f'- {dataset}\n')
        
        f.write('\n### Recommendations\n\n')
        for dataset, recommendation in pruning_analysis['recommendations'].items():
            f.write(f'**{dataset}**: {recommendation}\n\n')
        
        # Write device performance analysis
        f.write('## Device Performance Analysis\n\n')
        f.write('### Summary\n\n')
        f.write(f'Devices Analyzed: {", ".join(device_analysis["summary"]["devices"])}\n\n')
        
        f.write('Best Performing Models:\n')
        for model in device_analysis['summary']['best_performing_models']:
            f.write(f'- {model["dataset"]}: {model["activation"]} activation, {model["model_type"]} model on {model["device"]} ({model["inference_time_ms"]:.2f}ms)\n')
        
        f.write('\n### Recommendations\n\n')
        for dataset, recommendation in device_analysis['recommendations'].items():
            f.write(f'**{dataset}**: {recommendation}\n\n')
        
        # Write dataset-specific details
        f.write('## Dataset-Specific Details\n\n')
        
        for dataset in df['dataset_name'].unique():
            f.write(f'### {dataset}\n\n')
            
            # Activation function comparison
            f.write('#### Activation Function Comparison\n\n')
            f.write('| Activation | Accuracy | Accuracy Drop | Pruning Rate |\n')
            f.write('|------------|----------|---------------|--------------|\n')
            
            for activation, details in activation_analysis['dataset_details'][dataset]['activation_comparison'].items():
                f.write(f"| {activation} | {details['accuracy']:.2f}% | {details['accuracy_drop']:.2f}% | {details['pruning_rate']*100:.2f}% |\n")
            
            f.write('\n')
            
            # Best pruning configuration
            best_pruning = pruning_analysis['dataset_details'][dataset]['best_pruning']
            if best_pruning:
                f.write('#### Best Pruning Configuration\n\n')
                f.write(f'- Activation: {best_pruning["activation"]}\n')
                f.write(f'- Pruning Rate: {best_pruning["pruning_rate"]*100:.2f}%\n')
                f.write(f'- Accuracy Drop: {best_pruning["accuracy_drop"]:.2f}%\n\n')
            
            # Device performance
            if device_analysis['dataset_details'][dataset]['best_model']:
                f.write('#### Device Performance\n\n')
                f.write(f'- Best Device: {device_analysis["dataset_details"][dataset]["best_device"]}\n')
                f.write(f'- Best Model: {device_analysis["dataset_details"][dataset]["best_model"]["activation"]} activation, {device_analysis["dataset_details"][dataset]["best_model"]["model_type"]} model\n')
                f.write(f'- Inference Time: {device_analysis["dataset_details"][dataset]["best_model"]["inference_time_ms"]:.2f}ms\n')
                f.write(f'- ROM Size: {device_analysis["dataset_details"][dataset]["best_model"]["rom_size_bytes"]} bytes\n')
                f.write(f'- RAM Usage: {device_analysis["dataset_details"][dataset]["best_model"]["ram_usage_bytes"]} bytes\n\n')
    
    print(f"Comprehensive report generated: {report_path}")
    return report_path

def generate_visualizations(df, activation_analysis, pruning_analysis, device_analysis):
    """Generate visualizations for the analysis results"""
    # Create visualizations directory if it doesn't exist
    os.makedirs('reports/visualizations', exist_ok=True)
    
    visualization_paths = []
    
    # 1. Activation function comparison
    plt.figure(figsize=(12, 8))
    
    activations = list(activation_analysis['summary'].keys())
    accuracies = [activation_analysis['summary'][a]['avg_accuracy'] for a in activations]
    accuracy_drops = [activation_analysis['summary'][a]['avg_accuracy_drop'] for a in activations]
    
    x = np.arange(len(activations))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, accuracy_drops, width, label='Accuracy Drop')
    
    plt.xlabel('Activation Function')
    plt.ylabel('Percentage')
    plt.title('Activation Function Performance Comparison')
    plt.xticks(x, activations)
    plt.legend()
    
    plt.tight_layout()
    
    viz_path = 'reports/visualizations/activation_comparison.png'
    plt.savefig(viz_path)
    plt.close()
    
    visualization_paths.append(viz_path)
    
    # 2. Pruning impact
    plt.figure(figsize=(12, 8))
    
    datasets = list(pruning_analysis['dataset_details'].keys())
    pruning_rates = []
    accuracy_drops = []
    
    for dataset in datasets:
        best_pruning = pruning_analysis['dataset_details'][dataset]['best_pruning']
        if best_pruning:
            pruning_rates.append(best_pruning['pruning_rate'] * 100)
            accuracy_drops.append(best_pruning['accuracy_drop'])
    
    plt.scatter(pruning_rates, accuracy_drops)
    
    for i, dataset in enumerate(datasets):
        if i < len(pruning_rates):
            plt.annotate(dataset, (pruning_rates[i], accuracy_drops[i]))
    
    plt.xlabel('Pruning Rate (%)')
    plt.ylabel('Accuracy Drop (%)')
    plt.title('Pruning Impact: Rate vs. Accuracy Drop')
    
    plt.tight_layout()
    
    viz_path = 'reports/visualizations/pruning_impact.png'
    plt.savefig(viz_path)
    plt.close()
    
    visualization_paths.append(viz_path)
    
    # 3. Device performance (if available)
    if device_analysis['summary']['devices']:
        plt.figure(figsize=(12, 8))
        
        datasets = []
        devices = []
        inference_times = []
        
        for model in device_analysis['summary']['best_performing_models']:
            datasets.append(model['dataset'])
            devices.append(model['device'])
            inference_times.append(model['inference_time_ms'])
        
        plt.bar(datasets, inference_times)
        plt.xlabel('Dataset')
        plt.ylabel('Inference Time (ms)')
        plt.title('Device Performance: Inference Time by Dataset')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        viz_path = 'reports/visualizations/device_performance.png'
        plt.savefig(viz_path)
        plt.close()
        
        visualization_paths.append(viz_path)
    
    print(f"Generated {len(visualization_paths)} visualizations")
    return visualization_paths

def main():
    """Main function to run the analysis pipeline"""
    parser = argparse.ArgumentParser(description='Run the neural network optimization analysis pipeline')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], help='Filter by activation function')
    parser.add_argument('--output', type=str, default='reports/comprehensive_analysis.md', help='Output path for the report')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    # Load data
    df = load_results_index()
    if df is None:
        print("Failed to load results index. Exiting.")
        return
    
    # Filter by dataset and activation if specified
    if args.dataset:
        df = df[df['dataset_name'] == args.dataset]
    
    if args.activation:
        df = df[df['activation'] == args.activation]
    
    if df.empty:
        print("No data found matching the specified filters.")
        return
    
    # Load profiling data
    profiling_data = load_profiling_data()
    
    # Run analysis
    activation_analysis = analyze_activation_functions(df)
    pruning_analysis = analyze_pruning_impact(df)
    device_analysis = analyze_device_performance(profiling_data)
    
    # Generate report
    report_path = generate_comprehensive_report(df, activation_analysis, pruning_analysis, device_analysis)
    
    # Generate visualizations if requested
    if args.visualize:
        visualization_paths = generate_visualizations(df, activation_analysis, pruning_analysis, device_analysis)
        print(f"Visualizations saved to: {', '.join(visualization_paths)}")
    
    print(f"Analysis complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main() 