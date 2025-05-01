import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_json_metrics(file_path):
    """Load and parse a JSON metrics file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_csv_file(file_path):
    """Load and parse a CSV file."""
    try:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Warning: Empty file {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: No data in {file_path}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_dataset_results(results_dir):
    """Analyze results for a specific dataset."""
    dataset_name = os.path.basename(results_dir)
    metrics_files = list(Path(results_dir).glob('*_metrics_*.json'))
    
    results = {
        'dataset': dataset_name,
        'metrics_files': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'timestamps': []
    }
    
    for metrics_file in metrics_files:
        metrics = load_json_metrics(metrics_file)
        if metrics:
            results['metrics_files'].append(str(metrics_file))
            results['train_accuracies'].append(metrics.get('train_accuracy', None))
            results['test_accuracies'].append(metrics.get('test_accuracy', None))
            results['timestamps'].append(metrics.get('timestamp', None))
    
    return results

def analyze_activation_comparisons(results_dir):
    """Analyze activation function comparison results."""
    comparison_files = list(Path(results_dir).glob('*_activation_comparison.csv'))
    
    all_comparisons = []
    for file in comparison_files:
        df = load_csv_file(file)
        if df is not None and not df.empty:
            all_comparisons.append(df)
    
    if all_comparisons:
        return pd.concat(all_comparisons, ignore_index=True)
    return None

def analyze_inference_results(results_dir):
    """Analyze inference timing and memory usage results."""
    inference_file = os.path.join(results_dir, 'inference_analysis.csv')
    if os.path.exists(inference_file):
        return load_csv_file(inference_file)
    return None

def generate_summary_report(results_dir):
    """Generate a comprehensive summary report."""
    # Analyze all datasets
    dataset_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    dataset_results = []
    
    for dataset_dir in dataset_dirs:
        if dataset_dir.name not in ['inference', 'profiling', 'tagging', 'tagging_index']:
            results = analyze_dataset_results(dataset_dir)
            dataset_results.append(results)
    
    # Analyze activation comparisons
    activation_comparisons = analyze_activation_comparisons(results_dir)
    
    # Analyze inference results
    inference_results = analyze_inference_results(results_dir)
    
    # Generate summary statistics
    summary = {
        'total_datasets': len(dataset_results),
        'datasets_with_metrics': sum(1 for r in dataset_results if r['metrics_files']),
        'datasets_with_inference': 1 if inference_results is not None else 0,
        'activation_comparisons': len(activation_comparisons) if activation_comparisons is not None else 0
    }
    
    # Generate visualizations
    if activation_comparisons is not None and not activation_comparisons.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=activation_comparisons, x='dataset', y='accuracy_drop')
        plt.xticks(rotation=45)
        plt.title('Accuracy Drop by Dataset and Activation Function')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'accuracy_drop_analysis.png'))
    
    # Save summary report
    report_path = os.path.join(results_dir, 'analysis_summary.md')
    with open(report_path, 'w') as f:
        f.write('# Results Analysis Summary\n\n')
        f.write('## Overview\n')
        f.write(f'- Total datasets analyzed: {summary["total_datasets"]}\n')
        f.write(f'- Datasets with metrics: {summary["datasets_with_metrics"]}\n')
        f.write(f'- Datasets with inference data: {summary["datasets_with_inference"]}\n')
        f.write(f'- Activation comparisons available: {summary["activation_comparisons"]}\n\n')
        
        f.write('## Dataset-specific Results\n')
        for result in dataset_results:
            f.write(f'\n### {result["dataset"]}\n')
            f.write(f'- Number of metric files: {len(result["metrics_files"])}\n')
            if result['test_accuracies']:
                f.write(f'- Average test accuracy: {np.mean(result["test_accuracies"]):.2f}%\n')
                f.write(f'- Best test accuracy: {np.max(result["test_accuracies"]):.2f}%\n')
        
        if inference_results is not None and not inference_results.empty:
            f.write('\n## Inference Analysis\n')
            f.write('```\n')
            f.write(inference_results.to_string())
            f.write('\n```\n')
        
        if activation_comparisons is not None and not activation_comparisons.empty:
            f.write('\n## Activation Function Comparison\n')
            f.write('```\n')
            f.write(activation_comparisons.to_string())
            f.write('\n```\n')
    
    return report_path

if __name__ == '__main__':
    results_dir = 'results'
    report_path = generate_summary_report(results_dir)
    print(f"Analysis complete. Report saved to: {report_path}") 