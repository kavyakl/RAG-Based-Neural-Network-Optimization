import json
import pandas as pd
from pathlib import Path

def load_consolidated_results(file_path='consolidated_results.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(results):
    metrics = []
    for result in results:
        metric = {
            'dataset': result['dataset'],
            'activation': result['activation'],
            'timestamp': result.get('timestamp', ''),
        }
        
        # Original model metrics
        if result['activation'] == 'original':
            metric.update({
                'train_accuracy': result.get('train_accuracy', None),
                'test_accuracy': result.get('test_accuracy', None),
                'type': 'original'
            })
        else:
            # Pruned model metrics
            metric.update({
                'original_accuracy': result.get('original_accuracy', None),
                'pruned_accuracy': result.get('pruned_accuracy', None),
                'accuracy_drop': result.get('accuracy_drop', None),
                'original_hidden_neurons': result.get('original_hidden_neurons', None),
                'pruned_hidden_neurons': result.get('pruned_hidden_neurons', None),
                'pruning_rate': round((1 - result.get('pruned_hidden_neurons', 0) / 
                                    result.get('original_hidden_neurons', 1)) * 100, 2) 
                                    if result.get('original_hidden_neurons') else None,
                'type': 'pruned'
            })
            
            # Extract correlation analysis metrics if available
            if 'correlation_results' in result:
                corr = result['correlation_results']
                metric.update({
                    'num_correlated_pairs': len(corr.get('correlated_pairs', [])),
                    'avg_importance_score': sum(corr.get('importance_scores', [0])) / 
                                         len(corr.get('importance_scores', [1]))
                })
        
        metrics.append(metric)
    return pd.DataFrame(metrics)

def generate_summary(df):
    # Convert test_accuracy to percentage for consistency
    if 'test_accuracy' in df.columns:
        df['test_accuracy'] = df['test_accuracy'] * 100
    
    # Group by dataset and activation for original models
    original = df[df['type'] == 'original'].groupby(['dataset', 'activation']).agg({
        'test_accuracy': ['mean', 'std', 'count']
    }).round(4)
    
    # Group by dataset and activation for pruned models
    pruned = df[df['type'] == 'pruned'].groupby(['dataset', 'activation']).agg({
        'pruned_accuracy': ['mean', 'std', 'count'],
        'accuracy_drop': ['mean', 'std'],
        'pruning_rate': ['mean', 'std'],
        'avg_importance_score': ['mean', 'std']
    }).round(4)
    
    # Ensure consistent column names and structure
    original.columns = ['test_accuracy_mean', 'test_accuracy_std', 'test_accuracy_count']
    pruned.columns = ['pruned_accuracy_mean', 'pruned_accuracy_std', 'pruned_accuracy_count', 
                      'accuracy_drop_mean', 'accuracy_drop_std', 
                      'pruning_rate_mean', 'pruning_rate_std',
                      'avg_importance_score_mean', 'avg_importance_score_std']
    
    return original, pruned

def main():
    results = load_consolidated_results()
    df = extract_metrics(results)
    
    # Save detailed metrics
    df.to_csv('results/detailed_metrics.csv', index=False)
    print(f"Saved detailed metrics to results/detailed_metrics.csv")
    
    # Generate and save summary
    original_summary, pruned_summary = generate_summary(df)
    
    # Save with consistent formatting
    original_summary.to_csv('results/original_models_summary.csv')
    pruned_summary.to_csv('results/pruned_models_summary.csv')
    print(f"Saved summary metrics to results/original_models_summary.csv and results/pruned_models_summary.csv")
    
    # Print quick stats
    print("\nQuick Statistics:")
    print(f"Total datasets: {df['dataset'].nunique()}")
    print(f"Total experiments: {len(df)}")
    print(f"Unique activations: {df[df['type']=='pruned']['activation'].unique().tolist()}")

if __name__ == '__main__':
    main() 