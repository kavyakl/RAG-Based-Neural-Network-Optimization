"""
Smart RAG-based analysis layer for neural network optimization results.
Provides intelligent, actionable insights and professor-friendly analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from dataclasses import dataclass

class ActivationAnalysis(BaseModel):
    """Structured output for activation function analysis."""
    dataset_name: str = Field(description="Name of the dataset analyzed")
    best_accuracy_activation: str = Field(description="Activation function with best post-pruning accuracy")
    best_size_reduction_activation: str = Field(description="Activation function with best model size reduction")
    best_speedup_activation: str = Field(description="Activation function with best inference speedup")
    activation_rankings: Dict[str, float] = Field(description="Rankings of activations based on custom weights")
    key_insights: List[str] = Field(description="Key insights about activation function performance")

class ConstraintRecommendation(BaseModel):
    """Structured output for constraint-based recommendations."""
    dataset_name: str = Field(description="Name of the dataset analyzed")
    device: str = Field(description="Target deployment device")
    recommended_activation: str = Field(description="Recommended activation function")
    accuracy_drop: float = Field(description="Expected accuracy drop")
    inference_speedup: float = Field(description="Expected inference speedup")
    size_reduction: float = Field(description="Expected size reduction")
    rationale: str = Field(description="Explanation for the recommendation")

class PruningInsight(BaseModel):
    """Structured output for neuronal pruning insights."""
    dataset_name: str = Field(description="Name of the dataset analyzed")
    original_neurons: int = Field(description="Original number of hidden neurons")
    pruned_neurons: int = Field(description="Number of neurons after pruning")
    pruning_rate: float = Field(description="Percentage of neurons pruned")
    accuracy_drop: float = Field(description="Drop in accuracy after pruning")
    neuron_correlation_insights: List[str] = Field(description="Insights about neuron correlations")
    efficiency_metrics: Dict[str, float] = Field(description="Efficiency metrics like neurons per accuracy point")

@dataclass
class OptimizationRecommendation:
    device: str
    recommended_activation: str
    expected_accuracy_drop: float
    expected_speedup: float
    expected_size_reduction: float
    rationale: str

class SmartRAGAnalyzer:
    """Analyzes neural network optimization results using RAG-based approach."""
    
    def __init__(self, activation_data_path: str = "results/activation_comparison.csv"):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Initialize parsers
        self.activation_parser = PydanticOutputParser(pydantic_object=ActivationAnalysis)
        self.constraint_parser = PydanticOutputParser(pydantic_object=ConstraintRecommendation)
        self.pruning_parser = PydanticOutputParser(pydantic_object=PruningInsight)
        
        # Load activation comparison data
        self.activation_data = pd.read_csv(activation_data_path) if os.path.exists(activation_data_path) else pd.DataFrame()
        
        # Device-specific constraints (example values, should be calibrated)
        self.device_constraints = {
            "raspberry-pi-4": {
                "max_model_size_mb": 50,
                "max_inference_time_ms": 100,
                "min_accuracy": 0.85
            },
            "stm32": {
                "max_model_size_mb": 10,
                "max_inference_time_ms": 50,
                "min_accuracy": 0.80
            },
            "rp2040": {
                "max_model_size_mb": 5,
                "max_inference_time_ms": 30,
                "min_accuracy": 0.75
            }
        }
        
        # Define analysis prompts
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup LangChain prompts for different analyses."""
        # Activation analysis prompt
        self.activation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network optimization and activation functions.
            Analyze the performance of different activation functions and provide insights.
            Focus on accuracy preservation, size reduction, and inference speed."""),
            ("human", """Analyze the following activation function metrics:
            
            Dataset: {dataset_name}
            Metrics:
            {metrics_data}
            
            {format_instructions}
            """)
        ])
        
        # Constraint-based recommendation prompt
        self.constraint_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network deployment optimization.
            Provide recommendations based on specific constraints and requirements.
            Consider accuracy, size, and speed trade-offs."""),
            ("human", """Given the following constraints:
            
            Dataset: {dataset_name}
            Device: {device}
            Constraints: {constraints}
            
            Available metrics:
            {metrics_data}
            
            {format_instructions}
            """)
        ])
        
        # Pruning insight prompt
        self.pruning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in neural network pruning and optimization.
            Analyze pruning results and provide insights about neuron removal and efficiency."""),
            ("human", """Analyze the following pruning metrics:
            
            Dataset: {dataset_name}
            Metrics:
            {metrics_data}
            
            {format_instructions}
            """)
        ])
    
    def analyze_activation_tradeoffs(self, dataset_name: str) -> ActivationAnalysis:
        """Analyze trade-offs between different activation functions."""
        # Filter data for the dataset
        dataset_data = self.activation_data[self.activation_data['dataset'] == dataset_name]
        
        if dataset_data.empty:
            raise ValueError(f"No data found for dataset: {dataset_name}")
        
        # Prepare metrics data for the prompt
        metrics_data = dataset_data.to_string()
        
        # Generate analysis
        chain = self.activation_prompt | self.llm | self.activation_parser
        analysis = chain.invoke({
            "dataset_name": dataset_name,
            "metrics_data": metrics_data,
            "format_instructions": self.activation_parser.get_format_instructions()
        })
        
        return analysis
    
    def get_constraint_based_recommendations(
        self,
        dataset_name: str,
        device: str,
        custom_constraints: Optional[Dict[str, float]] = None
    ) -> List[OptimizationRecommendation]:
        """Get recommendations based on device constraints."""
        if device not in self.device_constraints:
            raise ValueError(f"Unknown device: {device}")
        
        constraints = self.device_constraints[device].copy()
        if custom_constraints:
            constraints.update(custom_constraints)
        
        analysis = self.analyze_dataset(dataset_name)
        if not analysis:
            return []
        
        recommendations = []
        baseline = analysis['baseline_accuracy']
        
        for activation, metrics in analysis['optimization_potential'].items():
            # Check if this activation meets constraints
            if (metrics['accuracy_drop'] <= (1 - constraints['min_accuracy']) * 100 and
                metrics['size_reduction'] >= 0 and
                metrics['speedup'] >= 1):
                
                rationale = (
                    f"Activation function {activation} meets device constraints: "
                    f"{metrics['accuracy_drop']:.1f}% accuracy drop, "
                    f"{metrics['size_reduction']:.1f}% size reduction, "
                    f"{metrics['speedup']:.1f}x speedup"
                )
                
                recommendations.append(OptimizationRecommendation(
                    device=device,
                    recommended_activation=activation,
                    expected_accuracy_drop=metrics['accuracy_drop'],
                    expected_speedup=metrics['speedup'],
                    expected_size_reduction=metrics['size_reduction'],
                    rationale=rationale
                ))
        
        # Sort by best balance of metrics
        recommendations.sort(key=lambda x: (
            x.expected_speedup * 0.4 +
            x.expected_size_reduction * 0.4 -
            x.expected_accuracy_drop * 0.2
        ), reverse=True)
        
        return recommendations
    
    def analyze_pruning_insights(self, dataset_name: str) -> PruningInsight:
        """Analyze neuronal pruning results and generate insights."""
        # Filter data for the dataset
        dataset_data = self.activation_data[self.activation_data['dataset'] == dataset_name]
        
        if dataset_data.empty:
            raise ValueError(f"No data found for dataset: {dataset_name}")
        
        # Prepare metrics data for the prompt
        metrics_data = dataset_data.to_string()
        
        # Generate insights
        chain = self.pruning_prompt | self.llm | self.pruning_parser
        insight = chain.invoke({
            "dataset_name": dataset_name,
            "metrics_data": metrics_data,
            "format_instructions": self.pruning_parser.get_format_instructions()
        })
        
        return insight
    
    def analyze_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Analyze a specific dataset's results."""
        dataset_data = self.activation_data[self.activation_data['dataset'] == dataset_name]
        if dataset_data.empty:
            return {}
        
        analysis = {
            'dataset': dataset_name,
            'best_activation': None,
            'baseline_accuracy': None,
            'optimization_potential': {},
            'recommendations': []
        }
        
        # Find baseline (usually ReLU)
        baseline = dataset_data[dataset_data['activation'] == 'relu']
        if not baseline.empty:
            # Normalize accuracy to percentage if needed
            orig_acc = baseline['original_accuracy'].iloc[0]
            analysis['baseline_accuracy'] = orig_acc / 100 if orig_acc > 100 else orig_acc
            analysis['best_activation'] = 'relu'
        
        # Analyze each activation function
        for _, row in dataset_data.iterrows():
            activation = row['activation']
            # Normalize accuracy to percentage if needed
            orig_acc = row['original_accuracy']
            accuracy = orig_acc / 100 if orig_acc > 100 else orig_acc
            size = row['original_rom_bytes'] if 'original_rom_bytes' in row else 0
            inference_time = row['original_inference_ms'] if 'original_inference_ms' in row else 0
            
            if accuracy > analysis.get('baseline_accuracy', 0):
                analysis['best_activation'] = activation
                analysis['baseline_accuracy'] = accuracy
            
            # Calculate optimization metrics
            if not baseline.empty:
                # Get pruned metrics if available
                pruned_size = row['pruned_rom_bytes'] if 'pruned_rom_bytes' in row else size
                pruned_inference_time = row['pruned_inference_ms'] if 'pruned_inference_ms' in row else inference_time
                pruned_acc = row['pruned_accuracy'] if 'pruned_accuracy' in row else accuracy
                pruned_accuracy = pruned_acc / 100 if pruned_acc > 100 else pruned_acc
                
                # Calculate metrics
                size_reduction = ((size - pruned_size) / size * 100) if size > 0 else 0
                speedup = (inference_time / pruned_inference_time) if pruned_inference_time > 0 else 0
                accuracy_drop = ((accuracy - pruned_accuracy) / accuracy * 100) if accuracy > 0 else 0
                
                analysis['optimization_potential'][activation] = {
                    'size_reduction': size_reduction,
                    'speedup': speedup,
                    'accuracy_drop': accuracy_drop
                }
        
        return analysis
    
    def generate_comprehensive_report(self, dataset_name: str) -> str:
        """Generate a comprehensive analysis report for a dataset."""
        analysis = self.analyze_dataset(dataset_name)
        if not analysis:
            return f"No analysis available for dataset: {dataset_name}"
        
        report = [
            f"# Comprehensive Analysis Report for {dataset_name}",
            "",
            "## Overview",
            f"- Best performing activation function: {analysis['best_activation']}",
            f"- Baseline accuracy: {analysis['baseline_accuracy']:.2%}",
            "",
            "## Optimization Potential by Activation Function",
            ""
        ]
        
        # Add optimization metrics table
        report.extend([
            "| Activation | Accuracy Drop | Size Reduction | Speedup |",
            "|------------|---------------|----------------|---------|"
        ])
        
        for activation, metrics in analysis['optimization_potential'].items():
            report.append(
                f"| {activation} | {metrics['accuracy_drop']:.1f}% | "
                f"{metrics['size_reduction']:.1f}% | {metrics['speedup']:.1f}x |"
            )
        
        report.extend([
            "",
            "## Device-Specific Recommendations",
            ""
        ])
        
        # Add recommendations for each device
        for device in self.device_constraints:
            recommendations = self.get_constraint_based_recommendations(dataset_name, device)
            report.append(f"### {device.upper()}")
            if recommendations:
                for rec in recommendations:
                    report.extend([
                        f"- **Recommended Activation**: {rec.recommended_activation}",
                        f"  - Expected accuracy drop: {rec.expected_accuracy_drop:.1f}%",
                        f"  - Expected speedup: {rec.expected_speedup:.1f}x",
                        f"  - Expected size reduction: {rec.expected_size_reduction:.1f}%",
                        f"  - Rationale: {rec.rationale}",
                        ""
                    ])
            else:
                report.append("No activation functions meet the device constraints.")
                report.append("")
        
        return "\n".join(report)
    
    def _format_rankings(self, rankings: Dict[str, float]) -> str:
        """Format activation rankings as a markdown table."""
        return "| Activation | Score |\n|------------|-------|\n" + \
               "\n".join(f"| {k} | {v:.2f} |" for k, v in rankings.items())
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as markdown bullet points."""
        return "\n".join(f"- {item}" for item in items)
    
    def _format_dict(self, d: Dict[str, float]) -> str:
        """Format a dictionary as a markdown table."""
        return "| Metric | Value |\n|--------|-------|\n" + \
               "\n".join(f"| {k} | {v:.2f} |" for k, v in d.items()) 