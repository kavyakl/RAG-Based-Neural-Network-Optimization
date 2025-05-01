"""
LangChain-based analysis module for neural network optimization results.
Provides functionality to generate insights and answer queries about model performance.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pathlib import Path

class ActivationScore(BaseModel):
    """Structured output for activation function scoring."""
    best_activation: str = Field(description="Best performing activation function", pattern="^(relu|sigmoid|tanh)$")
    scores: Dict[str, float] = Field(description="Scores for each activation function between 0 and 100")
    reasoning: str = Field(description="Explanation for the scores")

    class Config:
        json_schema_extra = {
            "example": {
                "best_activation": "relu",
                "scores": {"relu": 85.5, "sigmoid": 70.2, "tanh": 75.8},
                "reasoning": "ReLU shows best balance of accuracy and efficiency"
            }
        }

class NeuronSensitivity(BaseModel):
    """Structured output for neuron sensitivity analysis."""
    core_pruned_neurons: List[int] = Field(description="Neurons pruned across all activations")
    activation_specific: Dict[str, List[int]] = Field(description="Neurons specific to each activation")
    insights: str = Field(description="Key insights about pruning patterns")

class ConstraintStrategy(BaseModel):
    """Structured output for constraint-based strategy recommendation."""
    recommended_strategy: str = Field(description="Recommended activation and pruning strategy")
    meets_constraints: bool = Field(description="Whether strategy meets all constraints")
    expected_accuracy: float = Field(description="Expected accuracy with this strategy")
    warnings: List[str] = Field(description="Any warnings about constraints")
    justification: str = Field(description="Explanation for the recommendation")

class CrossDatasetStability(BaseModel):
    """Structured output for cross-dataset stability analysis."""
    most_stable_activation: str = Field(description="Most stable activation across datasets")
    datasets_best: List[str] = Field(description="Datasets where activation performs best")
    datasets_worst: List[str] = Field(description="Datasets where activation performs worst")
    comment: str = Field(description="Overall stability assessment")

class FailurePattern(BaseModel):
    """Structured output for failure pattern analysis."""
    dataset: str = Field(description="Dataset name")
    issues: List[str] = Field(description="List of identified issues")
    suggestion: str = Field(description="Suggested remediation")

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
        
        # Initialize parsers
        self.activation_parser = PydanticOutputParser(pydantic_object=ActivationScore)
        self.neuron_parser = PydanticOutputParser(pydantic_object=NeuronSensitivity)
        self.constraint_parser = PydanticOutputParser(pydantic_object=ConstraintStrategy)
        self.stability_parser = PydanticOutputParser(pydantic_object=CrossDatasetStability)
        self.failure_parser = PydanticOutputParser(pydantic_object=FailurePattern)
        
        # Initialize prompts
        self._init_prompts()

    def _init_prompts(self):
        """Initialize analysis prompts."""
        self.activation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in model compression and deployment tradeoffs.
            You MUST respond with a valid JSON object containing exactly these fields:
            {{
                "best_activation": "name of best activation",
                "scores": {{"relu": score, "sigmoid": score, "tanh": score}},
                "reasoning": "explanation string"
            }}"""),
            ("human", """Score each activation function (relu, tanh, sigmoid) based on:
            - Accuracy retention after pruning
            - Size reduction (%)
            - Inference time improvement (%)
            
            Dataset: {dataset_name}
            
            Activation Comparison Table:
            {activation_comparison_table}
            
            {format_instructions}""")
        ])
        
        self.neuron_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are analyzing pruning patterns across different activation functions."),
            ("human", """Given this list of pruned neurons for each activation, identify patterns:
            
            Dataset: {dataset_name}
            Neuron Pruning Patterns:
            {pruning_patterns}
            
            {format_instructions}""")
        ])
        
        self.constraint_prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an embedded AI deployment expert."),
            ("human", """Given performance data and constraints, recommend the best configuration:
            
            Dataset: {dataset_name}
            User Constraint: {constraints}
            
            Available Configurations:
            {configurations}
            
            {format_instructions}""")
        ])
        
        self.stability_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are analyzing activation stability across multiple datasets."),
            ("human", """Analyze activation function stability:
            
            Dataset Activation Summary:
            {cross_dataset_summary}
            
            {format_instructions}""")
        ])
        
        self.failure_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are tasked with identifying failure patterns in pruning outcomes."),
            ("human", """Flag problematic pruning outcomes:
            
            Dataset Summary:
            {dataset_summary}
            
            {format_instructions}""")
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

    def analyze_activation_scores(self, dataset_name: str, comparison_data: pd.DataFrame) -> ActivationScore:
        """Score activation functions based on multiple metrics."""
        prompt_inputs = {
            "dataset_name": dataset_name,
            "activation_comparison_table": comparison_data.to_string(),
            "format_instructions": self.activation_parser.get_format_instructions()
        }
        chain = self.activation_prompt | self.llm | self.activation_parser
        return chain.invoke(prompt_inputs)

    def analyze_neuron_sensitivity(self, dataset_name: str, pruning_data: Dict[str, List[int]]) -> NeuronSensitivity:
        """Analyze neuron pruning patterns."""
        prompt_inputs = {
            "dataset_name": dataset_name,
            "pruning_patterns": json.dumps(pruning_data, indent=2),
            "format_instructions": self.neuron_parser.get_format_instructions()
        }
        chain = self.neuron_prompt | self.llm | self.neuron_parser
        return chain.invoke(prompt_inputs)

    def recommend_strategy(self, dataset_name: str, constraints: str, configurations: pd.DataFrame) -> ConstraintStrategy:
        """Recommend strategy based on constraints."""
        prompt_inputs = {
            "dataset_name": dataset_name,
            "constraints": constraints,
            "configurations": configurations.to_string(),
            "format_instructions": self.constraint_parser.get_format_instructions()
        }
        chain = self.constraint_prompt | self.llm | self.constraint_parser
        return chain.invoke(prompt_inputs)

    def analyze_cross_dataset_stability(self, summary_data: pd.DataFrame) -> CrossDatasetStability:
        """Analyze activation stability across datasets."""
        prompt_inputs = {
            "cross_dataset_summary": summary_data.to_string(),
            "format_instructions": self.stability_parser.get_format_instructions()
        }
        chain = self.stability_prompt | self.llm | self.stability_parser
        return chain.invoke(prompt_inputs)

    def detect_failure_patterns(self, dataset_summary: pd.DataFrame) -> List[FailurePattern]:
        """Detect and analyze failure patterns."""
        prompt_inputs = {
            "dataset_summary": dataset_summary.to_string(),
            "format_instructions": self.failure_parser.get_format_instructions()
        }
        chain = self.failure_prompt | self.llm | self.failure_parser
        return chain.invoke(prompt_inputs)

    def generate_comprehensive_report(self, dataset_name: str, metrics_data: dict) -> str:
        """Generate a comprehensive analysis report combining all insights."""
        # Load and prepare data
        comparison_data = pd.DataFrame(metrics_data.get('activation_comparison', []))
        pruning_data = metrics_data.get('pruning_patterns', {})
        configurations = pd.DataFrame(metrics_data.get('configurations', []))
        
        # Run all analyses
        activation_scores = self.analyze_activation_scores(dataset_name, comparison_data)
        neuron_insights = self.analyze_neuron_sensitivity(dataset_name, pruning_data)
        strategy_rec = self.recommend_strategy(dataset_name, "inference time < 5ms and size < 50KB", configurations)
        failure_patterns = self.detect_failure_patterns(comparison_data)
        
        # Generate report
        report = f"""# Comprehensive Neural Network Analysis for {dataset_name}

## 1. Activation Function Analysis
Best Activation: {activation_scores.best_activation}
Reasoning: {activation_scores.reasoning}

Scores:
{json.dumps(activation_scores.scores, indent=2)}

## 2. Neuron Sensitivity Analysis
Core Pruned Neurons: {neuron_insights.core_pruned_neurons}
Key Insights: {neuron_insights.insights}

## 3. Deployment Strategy
Recommended: {strategy_rec.recommended_strategy}
Expected Accuracy: {strategy_rec.expected_accuracy}%
{strategy_rec.justification}

## 4. Failure Analysis
"""
        for pattern in failure_patterns:
            report += f"\nIssues in {pattern.dataset}:\n"
            report += "\n".join(f"- {issue}" for issue in pattern.issues)
            report += f"\nSuggestion: {pattern.suggestion}\n"
        
        return report

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