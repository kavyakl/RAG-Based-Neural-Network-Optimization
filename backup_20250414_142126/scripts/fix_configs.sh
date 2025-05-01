#!/bin/bash

# Function to update a config file
update_config() {
    local config_file="$1"
    local dataset_name=$(basename "$config_file" .yaml | sed 's/config_//')
    local activation=$(grep "activation:" "$config_file" | awk '{print $2}')

    # Create temporary file
    local temp_file=$(mktemp)

    # Update the config file structure
    cat > "$temp_file" << EOL
dataset:
  input_features: $(grep "input_features:" "$config_file" | awk '{print $2}')
  name: $dataset_name
  num_classes: $(grep "num_classes:" "$config_file" | awk '{print $2}')
  data_path: data/${dataset_name}.csv

model:
  activation: $activation
  hidden_neurons: $(grep "hidden_neurons:" "$config_file" | awk '{print $2}')
  base_dir: models/${dataset_name}/${activation}
  
training:
  batch_size: 32
  enabled: true
  epochs: 1000
  learning_rate: 0.001
  save_dir: models/${dataset_name}/${activation}/original

inference:
  batch_size: 32
  enabled: true
  output_dir: results/${dataset_name}/${activation}/inference

pruning:
  enabled: true
  pruned: false
  accuracy_drop: 0.05
  activity_threshold: 0.1
  correlation_threshold: 0.8
  sensitivity_threshold: 0.01
  save_dir: models/${dataset_name}/${activation}/pruned
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

export:
  enabled: true
  edge_impulse:
    enabled: true
    project_id: 669207
  onnx_dir: models/${dataset_name}/${activation}/onnx

output:
  results_dir: results/${dataset_name}/${activation}
EOL

    # Replace the original file
    mv "$temp_file" "$config_file"
    echo "Updated $config_file"
}

# Update all config files
for config_file in config/config_*.yaml; do
    if [[ "$config_file" != "config/config_template.yaml" ]]; then
        update_config "$config_file"
    fi
done

echo "All config files have been updated!" 