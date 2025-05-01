"""
Script to run all analysis tools and generate comprehensive reports.
"""

import os
import json
import argparse
from pathlib import Path
from generate_insights import main as generate_insights_main
from analyze_profiling import main as analyze_profiling_main, load_dataset_results, generate_profiling_report
from generate_comprehensive_report import generate_comprehensive_report
from analyze_optimization import analyze_optimization_results
from llm_analysis import LLMAnalyzer
import pandas as pd

# Add parent directory to path to allow imports from scripts
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from scripts.unified_results_processor import find_result_files, extract_metadata, create_activation_comparison, main as process_results

def load_consolidated_results():
    """Load the consolidated results from unified_results_processor.py"""
    results_file = Path("consolidated_results.json")
    if not results_file.exists():
        print("Consolidated results file not found. Please run unified_results_processor.py first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def run_llm_analysis(dataset_name: str, metrics_data: dict):
    """Run LLM analysis on the dataset results"""
    try:
        llm_analyzer = LLMAnalyzer()
        
        # Extract metrics from the data
        metrics = {
            'original_accuracy': metrics_data.get('original', {}).get('accuracy', 0),
            'pruned_accuracy': metrics_data.get('pruned', {}).get('accuracy', 0),
            'size_reduction': metrics_data.get('pruned', {}).get('pruning_rate', 0) * 100,
            'key_insights': [],
            'recommendations': []
        }
        
        # Save LLM analysis to reports directory
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/{dataset_name}_llm_analysis.md"
        
        with open(report_path, "w") as f:
            f.write(f"# LLM Analysis for {dataset_name}\n\n")
            f.write(f"## Model Performance\n")
            f.write(f"- Original Accuracy: {metrics['original_accuracy']:.2f}%\n")
            f.write(f"- Pruned Accuracy: {metrics['pruned_accuracy']:.2f}%\n")
            f.write(f"- Size Reduction: {metrics['size_reduction']:.2f}%\n\n")
            
            f.write("## Key Insights\n")
            for insight in metrics['key_insights']:
                f.write(f"- {insight}\n")
            
            f.write("\n## Recommendations\n")
            for rec in metrics['recommendations']:
                f.write(f"- {rec}\n")
        
        print(f"LLM analysis saved to {report_path}")
        return metrics
        
    except Exception as e:
        print(f"Error running LLM analysis for {dataset_name}: {str(e)}")
        return None

def run_analysis(dataset_name):
    """Run all analysis for a given dataset."""
    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)
    
    # Generate comprehensive report
    comprehensive_report_path = generate_comprehensive_report(dataset_name)
    print(f"Comprehensive report generated at: {comprehensive_report_path}")
    
    # Generate profiling report
    profiling_results = load_dataset_results(dataset_name)
    if profiling_results:
        profiling_report = generate_profiling_report(profiling_results)
        profiling_report_path = Path("reports") / f"{dataset_name}_profiling_report.md"
        profiling_report_path.write_text(profiling_report)
        print(f"Profiling report generated at: {profiling_report_path}")
    else:
        print("No profiling results found")
    
    # Generate optimization report
    optimization_results = analyze_optimization_results(dataset_name)
    if optimization_results:
        # Create optimization report
        optimization_report = f"""# Optimization Analysis for {dataset_name}

## Pruning Analysis
{optimization_results['pruning_results']['pruning_analysis']}

## Activation Analysis
{optimization_results['activation_results']['activation_analysis']}
"""
        optimization_report_path = Path("reports") / f"{dataset_name}_optimization_report.md"
        optimization_report_path.write_text(optimization_report)
        print(f"Optimization report generated at: {optimization_report_path}")
    else:
        print("No optimization results found")

def main():
    """Run all analysis tools and generate comprehensive reports."""
    parser = argparse.ArgumentParser(description="Run all analysis tools")
    parser.add_argument("--dataset", type=str, help="Specific dataset to analyze (optional)")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization analysis")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling analysis")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive analysis")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM analysis")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Running Neural Network Analysis Tools")
    print("=" * 80)
    
    # First run unified results processor if needed
    if not os.path.exists("consolidated_results.json"):
        print("\nRunning Unified Results Processor...")
        process_args = argparse.Namespace()
        process_args.dataset = args.dataset
        process_results(process_args)
    
    # Load consolidated results
    consolidated_results = load_consolidated_results()
    if consolidated_results is None:
        return
    
    # Get unique datasets from consolidated results
    datasets = list(set(result['dataset'] for result in consolidated_results))
    
    # Filter by dataset if specified
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Dataset {args.dataset} not found in consolidated results.")
            return
        datasets = [args.dataset]
    
    # Run optimization analysis
    if not args.skip_optimization:
        print("\n" + "=" * 40)
        print("Running Optimization Analysis")
        print("=" * 40)
        try:
            generate_insights_main()
        except Exception as e:
            print(f"Error running optimization analysis: {str(e)}")
    
    # Run profiling analysis
    if not args.skip_profiling:
        print("\n" + "=" * 40)
        print("Running Profiling Analysis")
        print("=" * 40)
        for dataset in datasets:
            print(f"\nAnalyzing profiling results for {dataset} dataset...")
            try:
                run_analysis(dataset)
            except Exception as e:
                print(f"Error analyzing {dataset}: {str(e)}")
    
    # Run LLM analysis with unified results
    if not args.skip_llm:
        print("\n" + "=" * 40)
        print("Running LLM Analysis")
        print("=" * 40)
        
        from llm_analysis import LLMAnalyzer
        analyzer = LLMAnalyzer()
        
        for dataset in datasets:
            print(f"\nAnalyzing {dataset} dataset...")
            try:
                # Prepare data for LLM analysis
                dataset_results = [r for r in consolidated_results if r['dataset'] == dataset]
                
                # Extract activation comparison data
                comparison_data = pd.DataFrame([
                    {
                        'activation': r.get('activation'),
                        'accuracy': r.get('accuracy', 0),
                        'pruning_rate': r.get('pruning_rate', 0),
                        'inference_time_ms': r.get('inference_time_ms', 0)
                    }
                    for r in dataset_results
                ])
                
                # Extract pruning patterns
                pruning_data = {
                    r.get('activation'): r.get('pruned_neurons', [])
                    for r in dataset_results if r.get('model_type') == 'pruned'
                }
                
                # Prepare metrics data structure
                metrics_data = {
                    'activation_comparison': comparison_data.to_dict('records'),
                    'pruning_patterns': pruning_data,
                    'configurations': dataset_results
                }
                
                # Generate comprehensive report
                report = analyzer.generate_comprehensive_report(dataset, metrics_data)
                
                # Save report
                os.makedirs("reports", exist_ok=True)
                report_path = f"reports/{dataset}_llm_analysis.md"
                with open(report_path, "w") as f:
                    f.write(report)
                
                print(f"LLM analysis report saved to {report_path}")
                
            except Exception as e:
                print(f"Error running LLM analysis for {dataset}: {str(e)}")
    
    # Run comprehensive analysis
    if not args.skip_comprehensive:
        print("\n" + "=" * 40)
        print("Running Comprehensive Analysis")
        print("=" * 40)
        for dataset in datasets:
            print(f"\nGenerating comprehensive report for {dataset} dataset...")
            try:
                run_analysis(dataset)
            except Exception as e:
                print(f"Error generating report for {dataset}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"Reports generated in: {os.path.abspath('reports')}")
    print("\nGenerated reports:")
    for report in os.listdir("reports"):
        print(f"- {report}")

if __name__ == "__main__":
    main() 