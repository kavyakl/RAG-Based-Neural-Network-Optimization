"""
LangChain-based analysis module for neural network optimization results.
Provides functionality to generate insights and answer queries about model performance.
"""

import os
import json
import glob
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pathlib import Path

class ModelAnalysis(BaseModel):
    """Structured output for model analysis."""
    dataset_name: str = Field(description="Name of the dataset analyzed")
    original_accuracy: float = Field(description="Accuracy of the original model")
    pruned_accuracy: float = Field(description="Accuracy of the pruned model")
    size_reduction: float = Field(description="Percentage reduction in model size")
    key_insights: List[str] = Field(description="Key insights about model performance")
    recommendations: List[str] = Field(description="Recommendations for improvement")

class LLMAnalyzer:
    """Analyzes neural network optimization results using LangChain."""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your-api-key-here":
            raise ValueError("Please set a valid OpenAI API key in the .env file")
            
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ModelAnalysis)
        
        # Define analysis prompt template
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization and analysis.
            Analyze the provided model metrics and generate insights about the optimization process.
            Focus on accuracy preservation, size reduction, and potential improvements."""),
            ("human", """Please analyze the following model metrics:
            
            Dataset: {dataset_name}
            Original Model:
            - Hidden Neurons: {original_neurons}
            - Training Accuracy: {original_train_acc}
            - Test Accuracy: {original_test_acc}
            
            Pruned Model:
            - Hidden Neurons: {pruned_neurons}
            - Training Accuracy: {pruned_train_acc}
            - Test Accuracy: {pruned_test_acc}
            
            {format_instructions}
            """)
        ])
        
        # Define query prompt template
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization.
            Answer questions about the model optimization process and results.
            Provide clear, concise, and accurate responses."""),
            ("human", """Context about the model:
            
            Dataset: {dataset_name}
            Original Model:
            - Hidden Neurons: {original_neurons}
            - Training Accuracy: {original_train_acc}
            - Test Accuracy: {original_test_acc}
            
            Pruned Model:
            - Hidden Neurons: {pruned_neurons}
            - Training Accuracy: {pruned_train_acc}
            - Test Accuracy: {pruned_test_acc}
            
            Question: {question}
            """)
        ])

    def load_metrics(self, dataset_name: str) -> Dict:
        """Load metrics for a specific dataset."""
        results_dir = Path("results") / dataset_name
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found for dataset: {dataset_name}")
        
        # Find the latest metrics files using glob
        original_metrics_files = glob.glob(str(results_dir / f"{dataset_name}_original_metrics_*.json"))
        pruned_metrics_files = glob.glob(str(results_dir / f"{dataset_name}_pruned_metrics_*.json"))
        
        if not original_metrics_files or not pruned_metrics_files:
            raise FileNotFoundError(f"Metrics files not found for dataset: {dataset_name}")
        
        # Sort by modification time and get the latest
        latest_original = max(original_metrics_files, key=os.path.getmtime)
        latest_pruned = max(pruned_metrics_files, key=os.path.getmtime)
        
        # Load metrics from files
        with open(latest_original) as f:
            original_metrics = json.load(f)
        
        with open(latest_pruned) as f:
            pruned_metrics = json.load(f)
        
        return {
            "original": original_metrics,
            "pruned": pruned_metrics
        }

    def analyze_model(self, dataset_name: str) -> ModelAnalysis:
        """Generate comprehensive analysis for a model."""
        metrics = self.load_metrics(dataset_name)
        
        # Calculate size reduction percentage
        original_size = metrics["original"]["model_size"]
        pruned_size = metrics["pruned"]["model_size"]
        size_reduction = ((original_size - pruned_size) / original_size) * 100
        
        # Get metrics from original model file
        original_train_acc = metrics["original"]["train_accuracy"]
        original_test_acc = metrics["original"]["test_accuracy"]
        original_neurons = metrics["original"]["hidden_neurons"]
        
        # Get metrics from pruned model file (which contains both original and pruned metrics)
        pruned_train_acc = metrics["pruned"]["pruned_train_accuracy"]
        pruned_test_acc = metrics["pruned"]["pruned_test_accuracy"]
        pruned_neurons = metrics["pruned"]["pruned_hidden_neurons"]
        
        # Prepare prompt inputs
        prompt_inputs = {
            "dataset_name": dataset_name,
            "original_neurons": original_neurons,
            "original_train_acc": original_train_acc,
            "original_test_acc": original_test_acc,
            "pruned_neurons": pruned_neurons,
            "pruned_train_acc": pruned_train_acc,
            "pruned_test_acc": pruned_test_acc,
            "format_instructions": self.parser.get_format_instructions()
        }
        
        # Generate analysis
        chain = self.analysis_prompt | self.llm | self.parser
        analysis = chain.invoke(prompt_inputs)
        
        return analysis

    def answer_query(self, dataset_name: str, question: str) -> str:
        """Answer specific questions about model optimization."""
        metrics = self.load_metrics(dataset_name)
        
        # Get metrics from original model file
        original_train_acc = metrics["original"]["train_accuracy"]
        original_test_acc = metrics["original"]["test_accuracy"]
        original_neurons = metrics["original"]["hidden_neurons"]
        
        # Get metrics from pruned model file
        pruned_train_acc = metrics["pruned"]["pruned_train_accuracy"]
        pruned_test_acc = metrics["pruned"]["pruned_test_accuracy"]
        pruned_neurons = metrics["pruned"]["pruned_hidden_neurons"]
        
        # Prepare prompt inputs
        prompt_inputs = {
            "dataset_name": dataset_name,
            "original_neurons": original_neurons,
            "original_train_acc": original_train_acc,
            "original_test_acc": original_test_acc,
            "pruned_neurons": pruned_neurons,
            "pruned_train_acc": pruned_train_acc,
            "pruned_test_acc": pruned_test_acc,
            "question": question
        }
        
        # Generate response
        chain = self.query_prompt | self.llm
        response = chain.invoke(prompt_inputs)
        
        return response.content

    def generate_report(self, dataset_name: str, output_dir: str = "reports") -> str:
        """Generate a comprehensive analysis report."""
        analysis = self.analyze_model(dataset_name)
        
        # Create reports directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report content
        report_content = f"""# Neural Network Optimization Analysis Report
## Dataset: {analysis.dataset_name}

### Model Performance
- Original Model Accuracy: {analysis.original_accuracy:.2%}
- Pruned Model Accuracy: {analysis.pruned_accuracy:.2%}
- Size Reduction: {analysis.size_reduction:.1f}%

### Key Insights
{chr(10).join(f"- {insight}" for insight in analysis.key_insights)}

### Recommendations
{chr(10).join(f"- {rec}" for rec in analysis.recommendations)}
"""
        
        # Save report
        report_path = os.path.join(output_dir, f"{dataset_name}_analysis_report.md")
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return report_path 