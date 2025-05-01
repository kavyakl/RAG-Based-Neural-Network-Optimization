"""
Script to generate a comprehensive report combining optimization and profiling analyses.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import argparse
import pandas as pd
from analyze_profiling import load_dataset_results, generate_profiling_report
from analyze_optimization import analyze_optimization_results
from llm_analysis import LLMAnalyzer

class ComprehensiveAnalysis(BaseModel):
    """Structured output for comprehensive analysis."""
    dataset_name: str = Field(description="Name of the dataset analyzed")
    optimization_summary: str = Field(description="Summary of optimization results")
    profiling_summary: str = Field(description="Summary of profiling results")
    deployment_strategy: str = Field(description="Recommended deployment strategy")
    key_insights: List[str] = Field(description="Key insights from the analysis")
    recommendations: List[str] = Field(description="Recommendations for further improvements")

class ComprehensiveAnalyzer:
    """Generates comprehensive analysis combining optimization and profiling results."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your-api-key-here":
            raise ValueError("Please set a valid OpenAI API key in the .env file")
        
        # Initialize analyzers
        self.llm_analyzer = LLMAnalyzer()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=ComprehensiveAnalysis)
        
        # Define analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization and deployment. 
            Analyze the provided optimization and profiling results, and provide a comprehensive 
            analysis with deployment recommendations.
            
            Your response must be a JSON object with the following structure:
            {{
                "dataset_name": "name of the dataset",
                "optimization_summary": "summary of optimization results",
                "profiling_summary": "summary of profiling results",
                "deployment_strategy": "recommended deployment strategy",
                "key_insights": ["insight 1", "insight 2", ...],
                "recommendations": ["recommendation 1", "recommendation 2", ...]
            }}
            
            Focus on:
            1. Overall performance improvements from optimization
            2. Device-specific performance characteristics
            3. Deployment recommendations based on profiling results
            4. Key insights and recommendations for further improvements"""),
            ("human", """Dataset: {dataset_name}
            
            Optimization Results:
            {optimization_results}
            
            Profiling Results:
            {profiling_results}
            
            Please provide a comprehensive analysis in the specified JSON format.""")
        ])
    
    def check_results(self, dataset_name: str) -> Dict[str, bool]:
        """Check if optimization and profiling results exist."""
        # Check for optimization results
        results_dir = Path("results") / dataset_name
        has_optimization = results_dir.exists() and (
            any(results_dir.glob(f"{dataset_name}_original_metrics_*.json")) and
            any(results_dir.glob(f"{dataset_name}_pruned_metrics_*.json"))
        )
        
        # Check for inference results
        inference_dir = Path("results/inference") / dataset_name
        has_inference = inference_dir.exists() and any(
            inference_dir.glob("inference_results_*.json")
        )
        
        # Check for activation comparison
        has_activation_comparison = Path("results").glob(f"{dataset_name}_activation_comparison.csv")
        
        return {
            "has_optimization": has_optimization,
            "has_inference": has_inference,
            "has_activation_comparison": bool(list(has_activation_comparison))
        }
    
    def load_optimization_results(self, dataset_name: str) -> str:
        """Load and format optimization results."""
        try:
            results = analyze_optimization_results(dataset_name)
            
            # Format pruning results
            pruning_text = "No pruning results available."
            if results["pruning_results"]["pruning_analysis"]:
                pruning_text = results["pruning_results"]["pruning_analysis"]
            
            # Format activation results
            activation_text = "No activation comparison results available."
            if results["activation_results"]["activation_analysis"]:
                activation_text = results["activation_results"]["activation_analysis"]
            
            return f"""
            Pruning Analysis:
            {pruning_text}
            
            Activation Analysis:
            {activation_text}
            """
        except Exception as e:
            print(f"Error loading optimization results: {str(e)}")
            return "Optimization results not available."
    
    def load_profiling_results(self, dataset_name: str) -> str:
        """Load and format profiling results."""
        try:
            profiling_results = load_dataset_results(dataset_name)
            if not profiling_results:
                return "Profiling results not available."
                
            return generate_profiling_report(profiling_results)
        except Exception as e:
            print(f"Error loading profiling results: {str(e)}")
            return "Profiling results not available."
    
    def generate_comprehensive_analysis(self, dataset_name: str) -> ComprehensiveAnalysis:
        """Generate comprehensive analysis combining optimization and profiling results."""
        # Check if results exist
        results_check = self.check_results(dataset_name)
        
        if not results_check["has_optimization"] and not results_check["has_inference"] and not results_check["has_activation_comparison"]:
            raise FileNotFoundError(f"No results found for dataset: {dataset_name}")
        
        # Load results
        optimization_results = self.load_optimization_results(dataset_name) if results_check["has_optimization"] else "Optimization results not available."
        profiling_results = self.load_profiling_results(dataset_name) if results_check["has_inference"] else "Inference results not available."
        
        # Prepare prompt inputs
        prompt_inputs = {
            "dataset_name": dataset_name,
            "optimization_results": optimization_results,
            "profiling_results": profiling_results
        }
        
        # Generate analysis
        chain = self.analysis_prompt | self.llm
        analysis_response = chain.invoke(prompt_inputs)
        
        # Parse the JSON response
        try:
            analysis_dict = json.loads(analysis_response.content)
            analysis = ComprehensiveAnalysis(**analysis_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing analysis response: {e}")
            print(f"Response content: {analysis_response.content}")
            raise
        
        return analysis
    
    def generate_report(self, dataset_name: str) -> str:
        """Generate a comprehensive report combining optimization and profiling analyses."""
        analysis = self.generate_comprehensive_analysis(dataset_name)
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate report content
        report_content = f"""# Comprehensive Neural Network Analysis Report: {dataset_name}

## Optimization Summary
{analysis.optimization_summary}

## Profiling Summary
{analysis.profiling_summary}

## Deployment Strategy
{analysis.deployment_strategy}

## Key Insights
{chr(10).join(f"- {insight}" for insight in analysis.key_insights)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in analysis.recommendations)}
"""
        
        # Save report
        report_path = f"reports/{dataset_name}_comprehensive_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return report_path

def generate_comprehensive_report(dataset_name):
    """Generate a comprehensive report for a given dataset."""
    analyzer = ComprehensiveAnalyzer()
    return analyzer.generate_report(dataset_name)

def main(dataset_name=None):
    """Main function to generate comprehensive reports."""
    if dataset_name is None:
        raise ValueError("Dataset name must be provided")
    
    report_path = generate_comprehensive_report(dataset_name)
    print(f"Comprehensive report generated at: {report_path}")
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Name of the dataset to analyze")
    args = parser.parse_args()
    main(args.dataset) 