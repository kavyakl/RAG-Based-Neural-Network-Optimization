"""
Script to generate comprehensive visualizations for neural network optimization results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_data():
    """Load data from CSV files."""
    results_dir = Path("results")
    
    # Load summary files
    original_summary = pd.read_csv(results_dir / "original_models_summary.csv")
    pruned_summary = pd.read_csv(results_dir / "pruned_models_summary.csv")
    
    # Load detailed metrics if available
    detailed_metrics = pd.read_csv(results_dir / "detailed_metrics.csv") if (results_dir / "detailed_metrics.csv").exists() else None
    
    return original_summary, pruned_summary, detailed_metrics

def plot_accuracy_comparison(original_summary, pruned_summary):
    """Plot accuracy comparison between original and pruned models."""
    plt.figure(figsize=(12, 6))
    
    datasets = original_summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, original_summary['test_accuracy_mean'], width, label='Original', color='skyblue')
    
    # For each activation function in pruned models
    colors = {'relu': 'lightcoral', 'sigmoid': 'lightgreen', 'tanh': 'orange'}
    for i, activation in enumerate(['relu', 'sigmoid', 'tanh']):
        mask = pruned_summary['activation'] == activation
        plt.bar(x + width/2, pruned_summary[mask]['pruned_accuracy_mean'], 
                width, label=f'Pruned ({activation})', color=colors[activation], alpha=0.7)
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison: Original vs Pruned Models')
    plt.xticks(x, datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png')
    plt.close()

def plot_pruning_impact(pruned_summary):
    """Plot pruning impact analysis."""
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot
    for activation in ['relu', 'sigmoid', 'tanh']:
        mask = pruned_summary['activation'] == activation
        plt.scatter(pruned_summary[mask]['pruning_rate_mean'], 
                   pruned_summary[mask]['accuracy_drop_mean'],
                   label=activation, alpha=0.6)
    
    plt.xlabel('Pruning Rate (%)')
    plt.ylabel('Accuracy Drop (%)')
    plt.title('Pruning Impact: Rate vs Accuracy Drop')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pruning_impact.png')
    plt.close()

def plot_activation_comparison(pruned_summary):
    """Plot activation function comparison."""
    # Prepare data for boxplot
    data = []
    for _, row in pruned_summary.iterrows():
        data.append({
            'dataset': row['dataset'],
            'activation': row['activation'],
            'accuracy_drop': row['accuracy_drop_mean'],
            'pruning_rate': row['pruning_rate_mean']
        })
    
    df = pd.DataFrame(data)
    
    # Create subplot for accuracy drop
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='activation', y='accuracy_drop')
    plt.title('Accuracy Drop by Activation Function')
    plt.ylabel('Accuracy Drop (%)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='activation', y='pruning_rate')
    plt.title('Pruning Rate by Activation Function')
    plt.ylabel('Pruning Rate (%)')
    
    plt.tight_layout()
    plt.savefig('results/activation_analysis.png')
    plt.close()

def plot_importance_scores(pruned_summary):
    """Plot importance score distribution."""
    plt.figure(figsize=(10, 6))
    
    for activation in ['relu', 'sigmoid', 'tanh']:
        mask = pruned_summary['activation'] == activation
        scores = pruned_summary[mask]['avg_importance_score_mean']
        if not scores.empty and not scores.isna().all():
            sns.kdeplot(data=scores, label=activation)
    
    plt.xlabel('Average Importance Score')
    plt.ylabel('Density')
    plt.title('Distribution of Neuron Importance Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/importance_scores.png')
    plt.close()

def plot_inference_time_comparison(pruned_summary):
    """Plot inference time comparison between activation functions."""
    plt.figure(figsize=(12, 6))
    
    for activation in ['relu', 'sigmoid', 'tanh']:
        mask = pruned_summary['activation'] == activation
        plt.bar(
            pruned_summary[mask]['dataset'],
            pruned_summary[mask]['pruned_inference_time_mean'],
            label=activation,
            alpha=0.7
        )
    
    plt.xlabel('Dataset')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time by Activation Function')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/inference_time_comparison.png')
    plt.close()

def plot_dataset_activation_heatmap(pruned_summary):
    """Generate a heatmap of accuracy drop per activation per dataset."""
    pivot_df = pruned_summary.pivot_table(
        index='dataset',
        columns='activation',
        values='accuracy_drop_mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title('Accuracy Drop Heatmap (Activation vs Dataset)')
    plt.ylabel('Dataset')
    plt.xlabel('Activation Function')
    plt.tight_layout()
    plt.savefig('results/activation_heatmap.png')
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations."""
    print("Loading data...")
    original_summary, pruned_summary, detailed_metrics = load_data()
    
    print("Generating visualizations...")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate all plots
    plot_accuracy_comparison(original_summary, pruned_summary)
    plot_pruning_impact(pruned_summary)
    plot_activation_comparison(pruned_summary)
    plot_importance_scores(pruned_summary)
    plot_inference_time_comparison(pruned_summary)
    plot_dataset_activation_heatmap(pruned_summary)
    
    print("Visualizations generated successfully!")
    print("Files saved:")
    print("- results/accuracy_comparison.png")
    print("- results/pruning_impact.png")
    print("- results/activation_analysis.png")
    print("- results/importance_scores.png")
    print("- results/inference_time_comparison.png")
    print("- results/activation_heatmap.png")

if __name__ == "__main__":
    generate_all_visualizations()