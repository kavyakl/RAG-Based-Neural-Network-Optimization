# Neural Network Optimization Recipe Book

This document provides detailed instructions for running different components of the neural network optimization pipeline.

## Table of Contents

1. [Training Models](#training-models)
2. [Pruning Models](#pruning-models)
3. [Exporting to ONNX](#exporting-to-onnx)
4. [Edge Impulse Profiling](#edge-impulse-profiling)
5. [Running Inference](#running-inference)
6. [Model Tagging](#model-tagging)
7. [Running Complete Pipeline](#running-complete-pipeline)
8. [Experimenting with Different Configurations](#experimenting-with-different-configurations)
9. [Managing Results](#managing-results)
10. [Fixing Configuration Files](#fixing-configuration-files)
11. [Running Multiple Configurations](#running-multiple-configurations)

## Training Models

### Single Dataset Training

```bash
python src/training/train_model.py --config config/config_iris.yaml
```

This will:
- Load and preprocess the dataset
- Train the model
- Save the model to `models/original/`
- Save metrics to `results/`

### Training All Datasets

```bash
./scripts/run_all_datasets.sh
```

This script:
- Iterates through all dataset configs
- Runs training for each dataset
- Saves models and metrics

## Pruning Models

### Single Dataset Pruning

```bash
python src/pruning/prune_model.py --config config/config_iris.yaml
```

This will:
- Load the trained model
- Perform structural pruning
- Save pruned model to `models/pruned/`
- Save pruning metrics to `results/`

### Pruning Parameters

Key parameters in config file:
```yaml
pruning:
  enabled: true
  correlation_threshold: 0.8
  sensitivity_threshold: 0.1
  activity_threshold: 0.1
  validation:
    max_accuracy_drop: 0.03
    min_accuracy: 0.9
```

## Exporting to ONNX

### Single Model Export

```bash
python src/export/onnx_converter.py --config config/config_iris.yaml
```

This will:
- Convert PyTorch model to ONNX
- Save to `models/onnx/`
- Validate accuracy on test set

### Export Settings

Key parameters in config:
```yaml
export:
  enabled: true
  save_dir: models/onnx
```

## Edge Impulse Profiling

### Profile Single Model

```bash
python src/export/profile_models.py --config config/config_iris.yaml
```

This will:
- Profile model on target devices
- Save results to `results/profiling/`
- Generate compatibility report

### Profiling Settings

Key parameters in config:
```yaml
export:
  edge_impulse:
    enabled: true
    target_devices:
      - raspberry-pi-4
      - raspberry-pi-rp2040
```

## Running Inference

### Single Model Inference

```bash
python src/inference/run_inference.py --config config/config_iris.yaml
```

This will:
- Load the model
- Run inference on test set
- Save results to `results/inference/`

### Inference Settings

Key parameters in config:
```yaml
inference:
  enabled: true
  output_dir: results/inference
```

## Model Tagging

### Tag Models

```bash
python src/export/tag_models.py --config config/config_iris.yaml
```

This will:
- Tag models with metadata
- Create similarity index
- Save index to `results/faiss_index/`

## Running Complete Pipeline

### Single Dataset Pipeline

```bash
python scripts/pipeline/pipeline_export_onnx_for_ei.py --config config/config_iris.yaml
```

This will run the complete pipeline:
1. Training
2. Pruning
3. ONNX Export
4. Edge Impulse Profiling
5. Model Tagging

### All Datasets Pipeline

```bash
./scripts/run_all_datasets.sh
```

## Experimenting with Different Configurations

### Update Activation Functions

```bash
./scripts/update_configs.sh
```

This script:
1. Updates activation functions in configs
2. Runs pipeline for all datasets
3. Compares results

### Configuration Parameters

Key parameters to experiment with:

1. Model Architecture:
```yaml
model:
  hidden_neurons: 10
  activation: sigmoid  # Options: relu, sigmoid, tanh
  learning_rate: 0.001
  batch_size: 32
  epochs: 1000
```

2. Pruning Strategy:
```yaml
pruning:
  correlation_threshold: 0.8
  sensitivity_threshold: 0.1
  activity_threshold: 0.1
```

3. Validation Criteria:
```yaml
pruning:
  validation:
    max_accuracy_drop: 0.03
    min_accuracy: 0.9
    validation_strategy: single  # Options: single, kfold
```

## Managing Results

### Cleaning Up Results

To clean up all existing results and start fresh:

```bash
./scripts/cleanup_results.sh --cleanup
```

This will:
- Remove all models from `models/original/`, `models/pruned/`, and `models/onnx/`
- Clear all results from `results/inference/`, `results/profiling/`, and `results/faiss_index/`
- Delete all log files from `results/`

### Backing Up Results

To create a backup of all results before cleaning:

```bash
./scripts/cleanup_results.sh --backup
```

This will:
- Create a timestamped backup directory (e.g., `results_backup_20230410_123045`)
- Copy all models and results to the backup directory
- Preserve all log files

### Backup and Cleanup

To backup all results and then clean up:

```bash
./scripts/cleanup_results.sh --backup --cleanup
```

### Using Cleanup with Configuration Updates

When updating configurations, you can include cleanup options:

```bash
./scripts/update_configs.sh --cleanup --backup
```

This will:
1. Create a backup of all existing results
2. Clean up all existing results
3. Update configurations with new settings
4. Run the pipeline with the new configurations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check if model files exist in correct directories
   - Verify model info JSON files are present
   - Ensure paths in config match actual file structure

2. **Pruning Failures**
   - Check if accuracy drop exceeds threshold
   - Verify minimum neuron requirements
   - Ensure validation data is properly loaded

3. **ONNX Export Issues**
   - Verify model architecture compatibility
   - Check input/output tensor shapes
   - Ensure all operations are supported

4. **Profiling Errors**
   - Verify Edge Impulse SDK installation
   - Check device compatibility
   - Ensure model size within limits

### Debugging Tips

1. Enable verbose logging:
```yaml
logging:
  level: DEBUG
```

2. Check intermediate results:
- Look in `results/` directory
- Examine JSON files for metrics
- Review log files

3. Test individual components:
- Run each step separately
- Verify inputs/outputs
- Check file paths and permissions

## Fixing Configuration Files

### Common YAML Issues

If you encounter YAML parsing errors, check for these common issues:

1. **Malformed Data Paths**
   ```yaml
   # INCORRECT:
   dataset:
     name: balance
   data/balance.csv    # Loose path without key

   # CORRECT:
   dataset:
     name: balance
     data_path: data/balance.csv
   ```

2. **Duplicate/Incorrect Values**
   ```yaml
   # INCORRECT:
   model:
     hidden_neurons: 24
     24              # Duplicate value
   23                # Loose value

   # CORRECT:
   model:
     hidden_neurons: 24
     input_features: 23
   ```

3. **Missing Indentation**
   ```yaml
   # INCORRECT:
   dataset:
     name: iris
   data_path: data/iris.csv    # Should be indented

   # CORRECT:
   dataset:
     name: iris
     data_path: data/iris.csv
   ```

### Automated Config Fixes

Use the `update_config.py` script to fix and update configurations:

```bash
python update_config.py --config config/config_balance.yaml --activation relu --enhanced-pruning true --edge-impulse false
```

Parameters:
- `--config`: Path to the configuration file
- `--activation`: Activation function (relu, sigmoid, tanh)
- `--enhanced-pruning`: Enable enhanced pruning (true/false)
- `--edge-impulse`: Enable Edge Impulse export (true/false)
- `--run-pipeline`: Optional flag to run the pipeline after updating config

## Running Multiple Configurations

### Using run_all_datasets.sh

The `run_all_datasets.sh` script automates running multiple configurations:

```bash
./run_all_datasets.sh
```

This script will:
1. Process all datasets in the configured list
2. Test each dataset with multiple activation functions:
   - ReLU
   - Sigmoid
   - Tanh
3. For each combination, run three variants:
   - Enhanced pruning enabled
   - Enhanced pruning disabled
   - Edge Impulse enabled
4. Log all output to timestamped files in `logs/`

### Dataset List Configuration

The script includes these datasets by default:
```bash
datasets=(
    "balance"
    "breast_cancer"
    "creditcard"
    "customers"
    "digits_binary_digit0"
    "fetalhealth"
    "heart"
    "iris"
    "liverdiagnostic"
    "raisin"
    "yeast"
)
```

### Activation Functions

Available activation functions:
```bash
activations=("relu" "sigmoid" "tanh")
```

### Logging

All runs are logged to timestamped files:
```bash
logs/run_all_YYYYMMDD_HHMMSS.log
```

### Running Individual Configurations

For testing or debugging, you can run specific configurations:

```bash
# Run single dataset with specific settings
python update_config.py --config config/config_iris.yaml --activation relu --enhanced-pruning true --edge-impulse false --run-pipeline

# Run single dataset with all activation functions
for activation in relu sigmoid tanh; do
    python update_config.py --config config/config_iris.yaml --activation $activation --enhanced-pruning true --edge-impulse false --run-pipeline
done
``` 