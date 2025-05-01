#!/bin/bash

# Function to fix a configuration file
fix_config() {
    local config_file=$1
    
    # Skip if file doesn't exist
    if [ ! -f "$config_file" ]; then
        echo "Configuration file not found: $config_file"
        return 1
    fi
    
    # Create a temporary file
    local temp_file=$(mktemp)
    
    # Extract dataset-specific values
    local input_features=$(grep "input_features:" "$config_file" | awk '{print $2}')
    local num_classes=$(grep "num_classes:" "$config_file" | awk '{print $2}')
    local hidden_neurons=$(grep "hidden_neurons:" "$config_file" | awk '{print $2}')
    local dataset_name=$(grep "name:" "$config_file" | awk '{print $2}')
    local dataset_path=$(grep "path:" "$config_file" | awk '{print $2}')
    
    # Extract model_path if it exists, otherwise use default
    local model_path=$(grep "model_path:" "$config_file" | awk '{print $2}')
    if [ -z "$model_path" ]; then
        model_path="models/original/relu/$dataset_name"
    fi
    
    # Extract activation function
    local activation=$(grep "activation:" "$config_file" | awk '{print $2}')
    if [ -z "$activation" ]; then
        activation="relu"
    fi
    
    # Create new configuration with preserved values
    cat > "$temp_file" << EOF
dataset:
  input_features: $input_features
  model_path: $model_path
  name: $dataset_name
  num_classes: $num_classes
  path: $dataset_path

export:
  edge_impulse:
    enabled: true
    project_id: 669207
  enabled: true
  onnx_dir: models/onnx
  original_model_dir: models/original
  pruned_model_dir: models/pruned
  save_dir: models/pruned/relu

inference:
  batch_size: 32
  enabled: true
  output_dir: results/inference/relu

model:
  activation: $activation
  hidden_neurons: $hidden_neurons
  model_save_path: models/$dataset_name

output:
  results_dir: results
  save_dir: models/pruned/relu

pruning:
  accuracy_drop: 0.05
  activity_threshold: 0.1
  correlation_threshold: 0.8
  enabled: true
  pruned: false
  save_dir: models/pruned/relu
  sensitivity_threshold: 0.01
  validation:
    accuracy_weight: 1.0
    batch_size: 3
    max_accuracy_drop: 0.05
    max_neurons_pruned: 8
    min_accuracy: 0.85
    require_min_neurons: 6
    size_weight: 0.0
    target_pruning_rate: 0.6
    validation_strategy: batch

training:
  batch_size: 32
  enabled: true
  epochs: 1000
  learning_rate: 0.001
  save_dir: models/original/relu
EOF

    # Replace the original file with the new one
    mv "$temp_file" "$config_file"
    echo "Fixed configuration: $config_file"
}

# Main execution
for config_file in config/config_*.yaml; do
    if [ -f "$config_file" ]; then
        fix_config "$config_file"
    fi
done

echo "All configuration files have been fixed." 