"""
Script to consolidate activation comparison results into a single CSV file.
"""

import os
import pandas as pd
import glob
from typing import List, Dict, Any

def find_activation_comparisons() -> List[Dict[str, Any]]:
    """Find all activation comparison files in the results directory."""
    comparisons = []
    
    # Search in results directory
    for dataset_dir in glob.glob("results/*/"):
        dataset_name = os.path.basename(os.path.dirname(dataset_dir))
        
        # Look for activation comparison files
        for file in glob.glob(os.path.join(dataset_dir, "*_activation_comparison.csv")):
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    # Add dataset name if not present
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                    comparisons.append(df)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
    
    return comparisons

def consolidate_results(comparisons: List[pd.DataFrame]) -> pd.DataFrame:
    """Consolidate all activation comparison results into a single DataFrame."""
    if not comparisons:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    consolidated = pd.concat(comparisons, ignore_index=True)
    
    # Ensure consistent column names
    required_columns = ['dataset', 'activation', 'accuracy', 'model_size', 'inference_time']
    for col in required_columns:
        if col not in consolidated.columns:
            consolidated[col] = None
    
    # Select and order columns
    consolidated = consolidated[required_columns]
    
    return consolidated

def main():
    # Find all activation comparisons
    print("Finding activation comparison files...")
    comparisons = find_activation_comparisons()
    
    if not comparisons:
        print("No activation comparison files found.")
        return 1
    
    # Consolidate results
    print("Consolidating results...")
    consolidated = consolidate_results(comparisons)
    
    # Save consolidated results
    output_path = "results/activation_comparison.csv"
    consolidated.to_csv(output_path, index=False)
    print(f"Consolidated results saved to: {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total datasets analyzed: {len(consolidated['dataset'].unique())}")
    print(f"Total comparisons: {len(consolidated)}")
    print("\nDatasets included:")
    for dataset in consolidated['dataset'].unique():
        print(f"- {dataset}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 