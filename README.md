# Neural Network Optimization Pipeline

This project implements a comprehensive neural network optimization pipeline for model pruning, quantization, and deployment to edge devices, with advanced analysis capabilities.

## Overview

The pipeline includes:
- Model training with various activation functions (ReLU, Sigmoid, Tanh)
- Structural pruning with weight adjustment using linear regression
- ONNX export for deployment
- Edge Impulse compatibility for microcontroller deployment
- Model profiling and comparison
- Advanced analysis tools:
  - Optimization analysis
  - Performance profiling
  - LLM-powered insights
  - Smart RAG analysis
  - Visualization generation

## Project Structure

- `configs/`: Configuration files for different datasets
- `src/`: Source code for the pipeline components
  - `models/`: Neural network model definitions
  - `pruning/`: Pruning algorithms and implementations
  - `training/`: Training utilities
  - `export/`: ONNX export and Edge Impulse compatibility
  - `inference/`: Inference and model comparison
  - `analysis/`: Analysis tools and components
    - `analyze_optimization.py`: Optimization analysis
    - `analyze_profiling.py`: Performance profiling
    - `generate_insights.py`: Insights generation
    - `llm_analysis.py`: LLM-powered analysis
    - `smart_rag_analysis.py`: RAG-based analysis
    - `generate_visualizations.py`: Visualization tools
  - `utils/`: Utility functions
- `scripts/`: Pipeline execution scripts
  - `pipeline/`: Pipeline orchestration
  - `unified_results_processor.py`: Results processing
  - `generate_summary_csv.py`: Summary generation
  - `bulk_run_pipeline.py`: Batch processing
- `results/`: Output directories for models and metrics
  - `{dataset_name}/{dataset_name}/`: Dataset-specific results
  - `profiling/`: Performance profiling results
  - `quantized/`: Quantization results
- `reports/`: Analysis reports
  - `{dataset_name}_optimization_report.md`
  - `{dataset_name}_profiling_report.md`
  - `{dataset_name}_comprehensive_report.md`
  - `{dataset_name}_llm_analysis.md`
  - `{dataset_name}_smart_rag_analysis.md`

## Usage

1. Configure your dataset in the `configs/` directory
2. Run the complete pipeline:
   ```bash
   python scripts/bulk_run_pipeline.py
   ```
   
   Or for a single dataset:
   ```bash
   python scripts/pipeline/pipeline_export_onnx_for_ei.py --config configs/config_your_dataset.yaml
   ```

3. Run analysis:
   ```bash
   python src/analysis/run_all_analysis.py --dataset your_dataset
   ```

4. Process results:
   ```bash
   python scripts/unified_results_processor.py
   python scripts/generate_summary_csv.py
   ```

## Dependencies

- PyTorch
- ONNX
- ONNX Runtime
- scikit-learn
- NumPy
- YAML
- OpenAI (for LLM analysis)
- LangChain (for RAG analysis)
- Pandas
- Matplotlib/Seaborn (for visualizations)

## Configuration

Key configuration options in `configs/config_your_dataset.yaml`:
```yaml
model:
  hidden_neurons: 10
  activation: sigmoid  # Options: relu, sigmoid, tanh
  learning_rate: 0.001
  batch_size: 32
  epochs: 1000

pruning:
  enabled: true
  correlation_threshold: 0.8
  sensitivity_threshold: 0.1
  activity_threshold: 0.1

analysis:
  enabled: true
  optimization: true
  profiling: true
  comprehensive: true
  llm: true
  smart_rag: true
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon research in neural network optimization and model compression techniques. 