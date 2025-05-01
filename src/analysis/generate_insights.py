"""
Generate insights from model optimization results.
"""

import os
import json
import pandas as pd
from pathlib import Path
from analyze_profiling import load_dataset_results, generate_profiling_report

def analyze_dataset(dataset_name: str, results_df: pd.DataFrame) -> dict:
    """Analyze results for a specific dataset."""
    insights = {
        'dataset': dataset_name,
        'metrics': {},
        'insights': []
    }
    
    # Filter for the specific dataset
    dataset_df = results_df[results_df['dataset_name'] == dataset_name].copy()
    
    if dataset_df.empty:
        insights['insights'].append("No data found for this dataset")
        return insights
    
    # Add model type if not present
    if 'model_type' not in dataset_df.columns:
        dataset_df['model_type'] = dataset_df.apply(
            lambda row: 'pruned' if 'pruned' in str(row.get('file_path', '')).lower() 
            else 'original', axis=1
        )
    
    # Get original and pruned model data
    original = dataset_df[dataset_df['model_type'] == 'original']
    pruned = dataset_df[dataset_df['model_type'] == 'pruned']
    
    if not original.empty and not pruned.empty:
        # Extract key metrics
        insights['metrics'] = {
            'original_accuracy': original['accuracy'].iloc[0] if 'accuracy' in original.columns else 0,
            'pruned_accuracy': pruned['accuracy'].iloc[0] if 'accuracy' in pruned.columns else 0,
            'pruning_rate': pruned['pruning_rate'].iloc[0] if 'pruning_rate' in pruned.columns else 0,
            'accuracy_drop': (
                original['accuracy'].iloc[0] - pruned['accuracy'].iloc[0]
                if 'accuracy' in original.columns and 'accuracy' in pruned.columns
                else 0
            )
        }
        
        # Generate insights
        if insights['metrics']['accuracy_drop'] < 2:
            insights['insights'].append("Excellent pruning results - minimal accuracy impact")
        elif insights['metrics']['accuracy_drop'] < 5:
            insights['insights'].append("Good balance of pruning and accuracy retention")
        else:
            insights['insights'].append("High accuracy drop - consider reducing pruning rate")
            
        if insights['metrics']['pruning_rate'] > 0.5:
            insights['insights'].append("Significant model size reduction achieved")
    else:
        insights['insights'].append("Missing either original or pruned model data")
            
    return insights

def generate_insights_report(dataset_name: str, insights: dict) -> str:
    """Generate a report from the insights."""
    report = f"""# Optimization Insights for {dataset_name}

## Performance Metrics
- Original Accuracy: {insights['metrics'].get('original_accuracy', 'N/A'):.2f}%
- Pruned Accuracy: {insights['metrics'].get('pruned_accuracy', 'N/A'):.2f}%
- Pruning Rate: {insights['metrics'].get('pruning_rate', 'N/A')*100:.2f}%
- Accuracy Drop: {insights['metrics'].get('accuracy_drop', 'N/A'):.2f}%

## Key Insights
"""
    for insight in insights['insights']:
        report += f"- {insight}\n"
        
    return report

def main():
    """Main function to generate insights."""
    try:
        # Load consolidated results
        results_file = Path("consolidated_results.json")
        if not results_file.exists():
            print("No consolidated results found. Please run unified_results_processor.py first.")
            return
            
        with open(results_file) as f:
            results = json.load(f)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Get unique datasets
        datasets = results_df['dataset'].unique().tolist() if 'dataset' in results_df.columns else []
        
        if not datasets:
            print("No datasets found in results.")
            return
        
        # Process each dataset
        for dataset_name in datasets:
            print(f"\nAnalyzing {dataset_name}...")
            
            # Generate insights
            insights = analyze_dataset(dataset_name, results_df)
            
            # Generate report
            report = generate_insights_report(dataset_name, insights)
            
            # Save report
            os.makedirs("reports", exist_ok=True)
            report_path = f"reports/{dataset_name}_insights.md"
            with open(report_path, "w") as f:
                f.write(report)
                
            print(f"Insights report generated: {report_path}")
            
    except Exception as e:
        print(f"Error generating insights: {str(e)}")

if __name__ == "__main__":
    main() 