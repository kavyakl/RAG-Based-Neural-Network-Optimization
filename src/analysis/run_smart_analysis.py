"""
Script to run smart RAG-based analysis for neural network optimization results.
"""

import os
import argparse
from smart_rag_analysis import SmartRAGAnalyzer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run smart RAG analysis for neural network optimization results")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to analyze")
    parser.add_argument("--device", type=str, default="raspberry-pi-4", help="Target deployment device")
    parser.add_argument("--constraints", type=str, default='{"max_accuracy_drop": 0.05, "min_speedup": 2.0}', 
                      help="JSON string of constraints")
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = SmartRAGAnalyzer()
        
        # Generate comprehensive report
        report_path = analyzer.generate_comprehensive_report(args.dataset)
        print(f"\nComprehensive analysis report generated: {report_path}")
        
        # Get constraint-based recommendations
        import json
        constraints = json.loads(args.constraints)
        recommendation = analyzer.get_constraint_recommendations(
            args.dataset,
            args.device,
            constraints
        )
        
        # Print recommendations
        print("\nConstraint-based Recommendations:")
        print(f"Device: {recommendation.device}")
        print(f"Recommended Activation: {recommendation.recommended_activation}")
        print(f"Expected Accuracy Drop: {recommendation.accuracy_drop:.2%}")
        print(f"Expected Speedup: {recommendation.inference_speedup:.2f}x")
        print(f"Expected Size Reduction: {recommendation.size_reduction:.2%}")
        print(f"\nRationale: {recommendation.rationale}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 