"""
Script to run all analysis tools and generate comprehensive reports.
"""

import os
import argparse
from pathlib import Path
from generate_insights import main as generate_insights_main
from analyze_profiling import main as analyze_profiling_main
from generate_comprehensive_report import main as generate_comprehensive_report_main

def main():
    """Run all analysis tools and generate comprehensive reports."""
    parser = argparse.ArgumentParser(description="Run all analysis tools")
    parser.add_argument("--dataset", type=str, help="Specific dataset to analyze (optional)")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization analysis")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling analysis")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive analysis")
    args = parser.parse_args()
    
    # List of datasets to analyze
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["iris", "breast_cancer", "raisin", "yeast"]
    
    print("=" * 80)
    print("Running Neural Network Analysis Tools")
    print("=" * 80)
    
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
        try:
            analyze_profiling_main()
        except Exception as e:
            print(f"Error running profiling analysis: {str(e)}")
    
    # Run comprehensive analysis
    if not args.skip_comprehensive:
        print("\n" + "=" * 40)
        print("Running Comprehensive Analysis")
        print("=" * 40)
        try:
            generate_comprehensive_report_main()
        except Exception as e:
            print(f"Error running comprehensive analysis: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"Reports generated in: {os.path.abspath('reports')}")

if __name__ == "__main__":
    main() 