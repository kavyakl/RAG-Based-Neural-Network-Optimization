# Neural Network Optimization Pipeline

This project implements a comprehensive neural network optimization pipeline for model pruning, quantization, and deployment to edge devices.

## Overview

The pipeline includes:
- Model training with various activation functions
- Structural pruning with weight adjustment using linear regression
- ONNX export for deployment
- Edge Impulse compatibility for microcontroller deployment
- Model profiling and comparison

## Project Structure

- `configs/`: Configuration files for different datasets
- `src/`: Source code for the pipeline components
  - `models/`: Neural network model definitions
  - `pruning/`: Pruning algorithms and implementations
  - `training/`: Training utilities
  - `export/`: ONNX export and Edge Impulse compatibility
  - `inference/`: Inference and model comparison
  - `deployment/`: Edge device deployment utilities
  - `tagging/`: Data tagging and annotation utilities
  - `data/`: Data processing and management utilities
  - `analysis/`: Model analysis and performance evaluation tools
  - `utils/`: Utility functions
- `scripts/`: Pipeline execution scripts
- `results/`: Output directories for models and metrics

## Usage

1. Configure your dataset in the `configs/` directory
2. Run the pipeline:
   ```
   python scripts/pipeline/pipeline_export_onnx_for_ei.py --config configs/config_your_dataset.yaml
   ```

## Dependencies

- PyTorch
- ONNX
- ONNX Runtime
- scikit-learn
- NumPy
- YAML

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon research in neural network optimization and model compression techniques. 