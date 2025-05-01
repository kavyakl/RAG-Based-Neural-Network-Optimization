"""
Script to generate insights and analysis reports using LangChain.
"""

import os
from pathlib import Path
from llm_analysis import LLMAnalyzer
from analyze_profiling import ProfilingAnalyzer

def check_dataset_results(dataset: str) -> bool:
    """Check if results exist for the given dataset."""
    results_dir = Path("results") / dataset
    if not results_dir.exists():
        print(f"No results directory found for {dataset} dataset")
        return False
    
    # Check for metrics files
    original_pattern = f"{dataset}_original_metrics_*.json"
    pruned_pattern = f"{dataset}_pruned_metrics_*.json"
    
    has_original = any(results_dir.glob(original_pattern))
    has_pruned = any(results_dir.glob(pruned_pattern))
    
    if not has_original:
        print(f"No original metrics file found for {dataset} dataset")
    if not has_pruned:
        print(f"No pruned metrics file found for {dataset} dataset")
    
    return has_original and has_pruned

def check_profiling_results(dataset: str) -> bool:
    """Check if profiling results exist for the given dataset."""
    profiling_dir = Path("results/profiling")
    if not profiling_dir.exists():
        print(f"No profiling results directory found")
        return False
    
    # Check for profiling files
    profiling_pattern = f"{dataset}_profiling_results_*.json"
    has_profiling = any(profiling_dir.glob(profiling_pattern))
    
    if not has_profiling:
        print(f"No profiling results found for {dataset} dataset")
    
    return has_profiling

def main():
    # Initialize analyzers
    try:
        llm_analyzer = LLMAnalyzer()
        profiling_analyzer = ProfilingAnalyzer()
    except ValueError as e:
        print(f"Error initializing analyzers: {str(e)}")
        return
    
    # List of datasets to analyze
    datasets = ["iris", "breast_cancer", "raisin", "yeast"]
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate reports for each dataset
    for dataset in datasets:
        print(f"\nChecking {dataset} dataset...")
        
        # Check for optimization results
        has_optimization = check_dataset_results(dataset)
        
        # Check for profiling results
        has_profiling = check_profiling_results(dataset)
        
        # Skip if no results exist
        if not has_optimization and not has_profiling:
            print(f"No results found for {dataset} dataset, skipping...")
            continue
        
        try:
            # Generate optimization analysis if available
            if has_optimization:
                print(f"Generating optimization analysis for {dataset} dataset...")
                optimization_report_path = llm_analyzer.generate_report(dataset)
                print(f"Optimization report generated: {optimization_report_path}")
                
                # Example queries
                queries = [
                    "What is the impact of pruning on model performance?",
                    "How effective was the optimization process?",
                    "What are the main trade-offs between model size and accuracy?"
                ]
                
                print("\nAnswering example queries:")
                for query in queries:
                    try:
                        answer = llm_analyzer.answer_query(dataset, query)
                        print(f"\nQ: {query}")
                        print(f"A: {answer}")
                    except Exception as e:
                        print(f"Error answering query '{query}': {str(e)}")
            
            # Generate profiling analysis if available
            if has_profiling:
                print(f"Generating profiling analysis for {dataset} dataset...")
                profiling_report_path = profiling_analyzer.generate_report(dataset)
                print(f"Profiling report generated: {profiling_report_path}")
                
        except FileNotFoundError as e:
            print(f"Error: Could not find required files for {dataset}: {str(e)}")
        except Exception as e:
            print(f"Error analyzing {dataset}: {str(e)}")
            print("Please check if the metrics files are in the correct format")

if __name__ == "__main__":
    main() 