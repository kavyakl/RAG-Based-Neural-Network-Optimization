"""
Script to run smart RAG analysis on consolidated activation comparison data.
"""

import os
import argparse
from smart_rag_analysis import SmartRAGAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run smart RAG analysis on activation comparison data")
    parser.add_argument("--dataset", type=str, help="Specific dataset to analyze")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SmartRAGAnalyzer()
    
    if args.dataset:
        # Analyze specific dataset
        report = analyzer.generate_comprehensive_report(args.dataset)
        output_path = os.path.join(args.output_dir, f"{args.dataset}_smart_rag_analysis.md")
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Generated smart RAG analysis report for {args.dataset}")
        print(f"Report saved to: {output_path}")
    else:
        # Analyze all datasets in the activation comparison data
        datasets = analyzer.activation_data['dataset'].unique()
        for dataset in datasets:
            report = analyzer.generate_comprehensive_report(dataset)
            output_path = os.path.join(args.output_dir, f"{dataset}_smart_rag_analysis.md")
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Generated smart RAG analysis report for {dataset}")
            print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main() 