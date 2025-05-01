"""
Script to analyze optimization results including pruning and activation function comparisons.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class OptimizationAnalysis(BaseModel):
    """Structured analysis of model optimization results."""
    dataset_name: str = Field(description="Name of the dataset")
    pruning_analysis: str = Field(description="Analysis of pruning results")
    quantization_analysis: str = Field(description="Analysis of quantization results")
    overall_impact: str = Field(description="Overall impact of optimization")
    recommendations: str = Field(description="Recommendations for further optimization")

class OptimizationAnalyzer:
    """Analyzes neural network optimization results using LangChain."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.results_dir = Path("results") / dataset_name
        
        # Load environment variables
        load_dotenv()
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your-api-key-here":
            raise ValueError("Please set a valid OpenAI API key in the .env file")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=OptimizationAnalysis)
        
        # Define analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization. Analyze the provided metrics 
            for both pruning and quantization, and provide detailed insights about the optimization process.
            
            Your response must be a JSON object with the following structure:
            {{
                "dataset_name": "name of the dataset",
                "pruning_analysis": "detailed analysis of pruning results",
                "quantization_analysis": "detailed analysis of quantization results",
                "overall_impact": "overall impact of optimization",
                "recommendations": "recommendations for further optimization"
            }}
            
            Focus on:
            1. Model size reduction
            2. Accuracy impact
            3. Computational efficiency
            4. Potential trade-offs"""),
            ("human", """Dataset: {dataset_name}
            
            Pruning Results:
            Original Neurons: {original_neurons}
            Pruned Neurons: {pruned_neurons}
            Original Train Accuracy: {original_train_acc}%
            Pruned Train Accuracy: {pruned_train_acc}%
            Original Test Accuracy: {original_test_acc}%
            Pruned Test Accuracy: {pruned_test_acc}%
            Original Model Size: {original_size} bytes
            Pruned Model Size: {pruned_size} bytes
            
            Quantization Results:
            Quantization Bits: {quantization_bits}
            Quantized Model Size: {quantized_size} bytes
            Quantized Pruned Model Size: {quantized_pruned_size} bytes
            
            Please provide a comprehensive analysis of the optimization results in the specified JSON format.""")
        ])
        
        # Define recommendations prompt
        self.recommendations_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the optimization results, provide specific recommendations for further 
            improvements. Consider:
            1. Potential for further pruning
            2. Alternative quantization approaches
            3. Architecture modifications
            4. Training strategy adjustments
            
            Your response should be a JSON object with a single field:
            {{
                "recommendations": "detailed recommendations for further optimization"
            }}"""),
            ("human", """Dataset: {dataset_name}
            
            Current Results:
            {analysis}
            
            Please provide specific recommendations for further optimization in the specified JSON format.""")
        ])
    
    def load_metrics(self) -> Tuple[Dict, Dict]:
        """Load metrics from original and pruned models."""
        original_metrics = {}
        pruned_metrics = {}
        
        # Load original model metrics
        original_path = self.results_dir / f"{self.dataset_name}_original_metrics.json"
        if original_path.exists():
            with open(original_path) as f:
                original_metrics = json.load(f)
                
        # Load pruned model metrics
        pruned_path = self.results_dir / f"{self.dataset_name}_pruned_metrics.json"
        if pruned_path.exists():
            with open(pruned_path) as f:
                pruned_metrics = json.load(f)
                
        return original_metrics, pruned_metrics
    
    def analyze_pruning(self) -> Dict:
        """Analyze pruning results."""
        original_metrics, pruned_metrics = self.load_metrics()
        
        if not original_metrics or not pruned_metrics:
            return {
                "pruning_analysis": "No pruning metrics found",
                "pruning_rate": None,
                "accuracy_impact": None
            }
            
        analysis = {
            "pruning_analysis": "",
            "pruning_rate": pruned_metrics.get("pruning_rate", 0),
            "accuracy_impact": original_metrics.get("test_accuracy", 0) - pruned_metrics.get("test_accuracy", 0)
        }
        
        # Generate pruning analysis text
        if analysis["pruning_rate"] is not None and analysis["accuracy_impact"] is not None:
            analysis["pruning_analysis"] = (
                f"Model achieved {analysis['pruning_rate']:.1%} pruning rate with "
                f"{abs(analysis['accuracy_impact']):.1%} {'decrease' if analysis['accuracy_impact'] > 0 else 'increase'} "
                f"in accuracy"
            )
            
        return analysis
    
    def analyze_activation(self) -> Dict:
        """Analyze activation function comparisons."""
        comparison_path = self.results_dir / f"{self.dataset_name}_activation_comparison.csv"
        
        if not comparison_path.exists():
            return {
                "activation_analysis": "No activation comparison data found",
                "best_activation": None,
                "activation_rankings": []
            }
            
        df = pd.read_csv(comparison_path)
        
        if df.empty:
            return {
                "activation_analysis": "Empty activation comparison data",
                "best_activation": None,
                "activation_rankings": []
            }
            
        # Find best activation
        best_row = df.loc[df["accuracy"].idxmax()]
        
        # Rank activations by accuracy
        rankings = df.sort_values("accuracy", ascending=False)
        activation_rankings = rankings["activation"].tolist()
        
        analysis = {
            "activation_analysis": (
                f"Best activation function was {best_row['activation']} "
                f"with {best_row['accuracy']:.1%} accuracy"
            ),
            "best_activation": best_row["activation"],
            "activation_rankings": activation_rankings
        }
        
        return analysis

def analyze_optimization_results(dataset_name: str) -> Dict:
    """Main function to analyze optimization results."""
    analyzer = OptimizationAnalyzer(dataset_name)
    
    pruning_results = analyzer.analyze_pruning()
    activation_results = analyzer.analyze_activation()
    
    return {
        "pruning_results": pruning_results,
        "activation_results": activation_results
    }

def main():
    """Generate optimization reports for all datasets."""
    # List of datasets to analyze
    datasets = ["iris", "breast_cancer", "raisin", "yeast"]
    
    # Generate reports for each dataset
    for dataset in datasets:
        print(f"\nAnalyzing optimization results for {dataset} dataset...")
        try:
            results = analyze_optimization_results(dataset)
            print(f"Pruning Analysis: {results['pruning_results']}")
            print(f"Activation Analysis: {results['activation_results']}")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error analyzing {dataset}: {str(e)}")
            print("Please check if all required files are present and in the correct format")

if __name__ == "__main__":
    main() 