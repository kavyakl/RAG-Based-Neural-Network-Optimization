"""
Script to analyze profiling results from ONNX models.
Analyzes performance metrics across different devices and provides insights.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import argparse

class ProfilingAnalysis(BaseModel):
    """Structured analysis of model profiling results."""
    dataset_name: str = Field(description="Name of the dataset")
    original_model_analysis: str = Field(description="Analysis of original model profiling results")
    pruned_model_analysis: str = Field(description="Analysis of pruned model profiling results")
    device_comparison: str = Field(description="Comparison of performance across devices")
    deployment_recommendations: str = Field(description="Recommendations for model deployment")

class ProfilingAnalyzer:
    """Analyzes profiling results from ONNX models using LangChain."""
    
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
        self.parser = PydanticOutputParser(pydantic_object=ProfilingAnalysis)
        
        # Define analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization and deployment. Analyze the provided 
            profiling results for ONNX models on different devices, and provide detailed insights about the 
            performance characteristics.
            
            Your response must be a JSON object with the following structure:
            {{
                "dataset_name": "name of the dataset",
                "original_model_analysis": "detailed analysis of original model profiling results",
                "pruned_model_analysis": "detailed analysis of pruned model profiling results",
                "device_comparison": "comparison of performance across devices",
                "deployment_recommendations": "recommendations for model deployment"
            }}
            
            Focus on:
            1. Inference time across devices
            2. Memory usage (ROM and RAM)
            3. MCU compatibility
            4. Performance improvements from pruning
            5. Device-specific optimizations"""),
            ("human", """Dataset: {dataset_name}
            
            Original Model Profiling:
            Raspberry Pi 4:
            - Inference Time: {original_rpi4_inference_time} ms
            - ROM Size: {original_rpi4_rom_size} bytes
            - RAM Usage: {original_rpi4_ram_usage} bytes
            - MCU Support: {original_rpi4_mcu_support}
            
            Raspberry Pi RP2040:
            - Inference Time: {original_rp2040_inference_time} ms
            - ROM Size: {original_rp2040_rom_size} bytes
            - RAM Usage: {original_rp2040_ram_usage} bytes
            - MCU Support: {original_rp2040_mcu_support}
            
            Pruned Model Profiling:
            Raspberry Pi 4:
            - Inference Time: {pruned_rpi4_inference_time} ms
            - ROM Size: {pruned_rpi4_rom_size} bytes
            - RAM Usage: {pruned_rpi4_ram_usage} bytes
            - MCU Support: {pruned_rpi4_mcu_support}
            
            Raspberry Pi RP2040:
            - Inference Time: {pruned_rp2040_inference_time} ms
            - ROM Size: {pruned_rp2040_rom_size} bytes
            - RAM Usage: {pruned_rp2040_ram_usage} bytes
            - MCU Support: {pruned_rp2040_mcu_support}
            
            Please provide a comprehensive analysis of the profiling results in the specified JSON format.""")
        ])
    
    def load_profiling_results(self, dataset_name: str) -> Dict:
        """Load profiling results for both original and pruned models."""
        # Path to profiling results
        profiling_dir = Path("results/profiling")
        if not profiling_dir.exists():
            raise FileNotFoundError(f"Profiling results directory not found")
        
        # Find latest profiling results file
        profiling_files = list(profiling_dir.glob(f"{dataset_name}_profiling_results_*.json"))
        if not profiling_files:
            raise FileNotFoundError(f"Profiling results not found for dataset: {dataset_name}")
        
        # Get latest file
        latest_profiling = max(profiling_files, key=os.path.getmtime)
        
        # Load profiling results
        with open(latest_profiling) as f:
            profiling_results = json.load(f)
        
        return profiling_results
    
    def analyze_profiling(self, dataset_name: str) -> ProfilingAnalysis:
        """Generate comprehensive analysis of profiling results."""
        profiling_results = self.load_profiling_results(dataset_name)
        
        # Extract metrics for original model
        original_rpi4 = profiling_results["original"]["raspberry-pi-4"]
        original_rp2040 = profiling_results["original"]["raspberry-pi-rp2040"]
        
        # Extract metrics for pruned model
        pruned_rpi4 = profiling_results["pruned"]["raspberry-pi-4"]
        pruned_rp2040 = profiling_results["pruned"]["raspberry-pi-rp2040"]
        
        # Prepare prompt inputs
        prompt_inputs = {
            "dataset_name": dataset_name,
            "original_rpi4_inference_time": original_rpi4["inference_time_ms"],
            "original_rpi4_rom_size": original_rpi4["rom_size_bytes"],
            "original_rpi4_ram_usage": original_rpi4["ram_usage_bytes"],
            "original_rpi4_mcu_support": "Yes" if original_rpi4["mcu_supported"] else "No",
            "original_rp2040_inference_time": original_rp2040["inference_time_ms"],
            "original_rp2040_rom_size": original_rp2040["rom_size_bytes"],
            "original_rp2040_ram_usage": original_rp2040["ram_usage_bytes"],
            "original_rp2040_mcu_support": "Yes" if original_rp2040["mcu_supported"] else "No",
            "pruned_rpi4_inference_time": pruned_rpi4["inference_time_ms"],
            "pruned_rpi4_rom_size": pruned_rpi4["rom_size_bytes"],
            "pruned_rpi4_ram_usage": pruned_rpi4["ram_usage_bytes"],
            "pruned_rpi4_mcu_support": "Yes" if pruned_rpi4["mcu_supported"] else "No",
            "pruned_rp2040_inference_time": pruned_rp2040["inference_time_ms"],
            "pruned_rp2040_rom_size": pruned_rp2040["rom_size_bytes"],
            "pruned_rp2040_ram_usage": pruned_rp2040["ram_usage_bytes"],
            "pruned_rp2040_mcu_support": "Yes" if pruned_rp2040["mcu_supported"] else "No"
        }
        
        # Generate analysis
        chain = self.analysis_prompt | self.llm
        analysis_response = chain.invoke(prompt_inputs)
        
        # Parse the JSON response
        try:
            analysis_dict = json.loads(analysis_response.content)
            analysis = ProfilingAnalysis(**analysis_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing analysis response: {e}")
            print(f"Response content: {analysis_response.content}")
            raise
        
        return analysis
    
    def generate_report(self, dataset_name: str) -> str:
        """Generate a comprehensive profiling report."""
        analysis = self.analyze_profiling(dataset_name)
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate report content
        report_content = f"""# ONNX Model Profiling Report: {dataset_name}

## Original Model Analysis
{analysis.original_model_analysis}

## Pruned Model Analysis
{analysis.pruned_model_analysis}

## Device Comparison
{analysis.device_comparison}

## Deployment Recommendations
{analysis.deployment_recommendations}
"""
        
        # Save report
        report_path = f"reports/{dataset_name}_profiling_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return report_path

def main():
    """Generate profiling reports for specified dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze profiling results for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to analyze")
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = ProfilingAnalyzer()
    except ValueError as e:
        print(f"Error initializing analyzer: {str(e)}")
        return
    
    # Generate report for the specified dataset
    print(f"\nAnalyzing profiling results for {args.dataset} dataset...")
    try:
        report_path = analyzer.generate_report(args.dataset)
        print(f"Profiling report generated: {report_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error analyzing {args.dataset}: {str(e)}")

if __name__ == "__main__":
    main() 