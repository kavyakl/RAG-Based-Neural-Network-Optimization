"""
Analyze profiling results from unified results processor.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

def load_dataset_results(dataset_name: str) -> pd.DataFrame:
    """Load results for a specific dataset from consolidated results."""
    results_file = Path("consolidated_results.json")
    if not results_file.exists():
        raise FileNotFoundError("consolidated_results.json not found")
        
    with open(results_file) as f:
        results = json.load(f)
    
    # Filter results for this dataset
    dataset_results = [r for r in results if r['dataset'] == dataset_name]
    
    # Add model type if not present
    for result in dataset_results:
        if 'model_type' not in result:
            result['model_type'] = 'pruned' if 'pruned' in str(result.get('file_path', '')).lower() else 'original'
    
    return pd.DataFrame(dataset_results)

def generate_profiling_report(results_df: pd.DataFrame) -> str:
    """Generate profiling report from results data."""
    if results_df.empty:
        return "No profiling results available."
    
    dataset_name = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else 'Unknown'
    
    # Get original model metrics
    original_df = results_df[results_df['model_type'] == 'original']
    original = original_df.iloc[0].to_dict() if not original_df.empty else {}
    
    # Get pruned model metrics
    pruned_df = results_df[results_df['model_type'] == 'pruned']
    pruned = pruned_df.iloc[0].to_dict() if not pruned_df.empty else {}
    
    report = f"""# Profiling Report for {dataset_name}

## Model Performance Metrics

### Original Model
- Accuracy: {original.get('accuracy', 'N/A'):.2f}% if isinstance(original.get('accuracy'), (int, float)) else 'N/A'
- Hidden Neurons: {original.get('hidden_neurons', 'N/A')}

### Pruned Model
- Accuracy: {pruned.get('accuracy', 'N/A'):.2f}% if isinstance(pruned.get('accuracy'), (int, float)) else 'N/A'
- Hidden Neurons: {pruned.get('pruned_neurons', 'N/A')}
- Pruning Rate: {pruned.get('pruning_rate', 0)*100:.2f}% if isinstance(pruned.get('pruning_rate'), (int, float)) else 'N/A'
- Accuracy Drop: {pruned.get('accuracy_drop', 0):.2f}% if isinstance(pruned.get('accuracy_drop'), (int, float)) else 'N/A'

### Resource Usage
- Original Inference Time: {original.get('inference_time_ms', 'N/A')} ms
- Pruned Inference Time: {pruned.get('inference_time_ms', 'N/A')} ms
- Original Memory: {original.get('size_bytes', 'N/A')} bytes
- Pruned Memory: {pruned.get('size_bytes', 'N/A')} bytes

## Analysis

"""
    # Add analysis based on available metrics
    if all(key in original for key in ['accuracy']) and all(key in pruned for key in ['accuracy']):
        acc_change = pruned['accuracy'] - original['accuracy']
        report += f"The pruned model shows a {'decrease' if acc_change < 0 else 'increase'} "
        report += f"in accuracy of {abs(acc_change):.2f}%.\n\n"
    
    if 'pruning_rate' in pruned:
        report += f"Model size was reduced by {pruned['pruning_rate']*100:.2f}% through pruning.\n\n"
    
    if all(key in original for key in ['inference_time_ms']) and all(key in pruned for key in ['inference_time_ms']):
        time_change = ((original['inference_time_ms'] - pruned['inference_time_ms']) / original['inference_time_ms']) * 100
        report += f"Inference time {'improved' if time_change > 0 else 'increased'} by {abs(time_change):.2f}%.\n\n"
    
    return report

def main(dataset_name: str):
    """Main function to analyze profiling results."""
    try:
        # Load results
        results_df = load_dataset_results(dataset_name)
        
        # Generate report
        report = generate_profiling_report(results_df)
        
        # Save report
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/{dataset_name}_profiling_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Profiling report generated: {report_path}")
        
    except Exception as e:
        print(f"Error analyzing profiling results: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze profiling results")
    parser.add_argument("dataset", help="Dataset name to analyze")
    args = parser.parse_args()
    main(args.dataset) 