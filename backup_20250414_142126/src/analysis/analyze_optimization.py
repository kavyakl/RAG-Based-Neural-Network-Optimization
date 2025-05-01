"""
Script to analyze neural network optimization results using LangChain.
Analyzes both pruning and quantization results across all datasets.
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

class OptimizationAnalysis(BaseModel):
    """Structured analysis of model optimization results."""
    dataset_name: str = Field(description="Name of the dataset")
    pruning_analysis: str = Field(description="Analysis of pruning results")
    quantization_analysis: str = Field(description="Analysis of quantization results")
    overall_impact: str = Field(description="Overall impact of optimization")
    recommendations: str = Field(description="Recommendations for further optimization")

class OptimizationAnalyzer:
    """Analyzes neural network optimization results using LangChain."""
    
    def __init__(self):
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
    
    def load_metrics(self, dataset_name: str) -> Dict:
        """Load metrics for both pruning and quantization."""
        # Load pruning metrics
        results_dir = Path("results") / dataset_name
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found for dataset: {dataset_name}")
        
        # Find latest metrics files
        original_metrics_files = list(results_dir.glob(f"{dataset_name}_original_metrics_*.json"))
        pruned_metrics_files = list(results_dir.glob(f"{dataset_name}_pruned_metrics_*.json"))
        
        if not original_metrics_files or not pruned_metrics_files:
            raise FileNotFoundError(f"Metrics files not found for dataset: {dataset_name}")
        
        # Get latest files
        latest_original = max(original_metrics_files, key=os.path.getmtime)
        latest_pruned = max(pruned_metrics_files, key=os.path.getmtime)
        
        # Load metrics
        with open(latest_original) as f:
            original_metrics = json.load(f)
        
        with open(latest_pruned) as f:
            pruned_metrics = json.load(f)
        
        # Load quantization metrics
        quantized_dir = Path("results/quantized/original") / dataset_name
        quantized_pruned_dir = Path("results/quantized/pruned") / dataset_name
        
        if not quantized_dir.exists() or not quantized_pruned_dir.exists():
            raise FileNotFoundError(f"Quantization results not found for dataset: {dataset_name}")
        
        quantized_info = json.loads((quantized_dir / f"quantized_{dataset_name}_model_info.json").read_text())
        quantized_pruned_info = json.loads((quantized_pruned_dir / f"quantized_{dataset_name}_pruned_model_info.json").read_text())
        
        return {
            "original": original_metrics,
            "pruned": pruned_metrics,
            "quantized": quantized_info,
            "quantized_pruned": quantized_pruned_info
        }
    
    def analyze_optimization(self, dataset_name: str) -> OptimizationAnalysis:
        """Generate comprehensive analysis of optimization results."""
        metrics = self.load_metrics(dataset_name)
        
        # Calculate size reductions
        pruning_size_reduction = ((metrics["original"]["model_size"] - metrics["pruned"]["model_size"]) 
                                / metrics["original"]["model_size"]) * 100
        quantization_size_reduction = ((metrics["pruned"]["model_size"] - metrics["quantized_pruned"]["model_size"]) 
                                     / metrics["pruned"]["model_size"]) * 100
        total_size_reduction = ((metrics["original"]["model_size"] - metrics["quantized_pruned"]["model_size"]) 
                              / metrics["original"]["model_size"]) * 100
        
        # Prepare prompt inputs
        prompt_inputs = {
            "dataset_name": dataset_name,
            "original_neurons": metrics["original"]["hidden_neurons"],
            "pruned_neurons": metrics["pruned"]["pruned_hidden_neurons"],
            "original_train_acc": metrics["original"]["train_accuracy"],
            "pruned_train_acc": metrics["pruned"]["pruned_train_accuracy"],
            "original_test_acc": metrics["original"]["test_accuracy"],
            "pruned_test_acc": metrics["pruned"]["pruned_test_accuracy"],
            "original_size": metrics["original"]["model_size"],
            "pruned_size": metrics["pruned"]["model_size"],
            "quantization_bits": metrics["quantized"]["quantization_bits"],
            "quantized_size": metrics["quantized"]["model_size"],
            "quantized_pruned_size": metrics["quantized_pruned"]["model_size"]
        }
        
        # Generate analysis
        chain = self.analysis_prompt | self.llm
        analysis_response = chain.invoke(prompt_inputs)
        
        # Parse the JSON response
        try:
            analysis_dict = json.loads(analysis_response.content)
            analysis = OptimizationAnalysis(**analysis_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing analysis response: {e}")
            print(f"Response content: {analysis_response.content}")
            raise
        
        # Generate recommendations
        recommendations_chain = self.recommendations_prompt | self.llm
        recommendations_response = recommendations_chain.invoke({
            "dataset_name": dataset_name,
            "analysis": analysis.json()
        })
        
        # Parse the recommendations JSON response
        try:
            recommendations_dict = json.loads(recommendations_response.content)
            analysis.recommendations = recommendations_dict["recommendations"]
        except json.JSONDecodeError as e:
            print(f"Error parsing recommendations response: {e}")
            print(f"Response content: {recommendations_response.content}")
            raise
        
        return analysis
    
    def generate_report(self, dataset_name: str) -> str:
        """Generate a comprehensive optimization report."""
        analysis = self.analyze_optimization(dataset_name)
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate report content
        report_content = f"""# Neural Network Optimization Report: {dataset_name}

## Pruning Analysis
{analysis.pruning_analysis}

## Quantization Analysis
{analysis.quantization_analysis}

## Overall Impact
{analysis.overall_impact}

## Recommendations
{analysis.recommendations}
"""
        
        # Save report
        report_path = f"reports/{dataset_name}_optimization_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return report_path

def main():
    """Generate optimization reports for all datasets."""
    # Initialize analyzer
    try:
        analyzer = OptimizationAnalyzer()
    except ValueError as e:
        print(f"Error initializing analyzer: {str(e)}")
        return
    
    # List of datasets to analyze
    datasets = ["iris", "breast_cancer", "raisin", "yeast"]
    
    # Generate reports for each dataset
    for dataset in datasets:
        print(f"\nAnalyzing optimization results for {dataset} dataset...")
        try:
            report_path = analyzer.generate_report(dataset)
            print(f"Report generated: {report_path}")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error analyzing {dataset}: {str(e)}")
            print("Please check if all required files are present and in the correct format")

if __name__ == "__main__":
    main() 