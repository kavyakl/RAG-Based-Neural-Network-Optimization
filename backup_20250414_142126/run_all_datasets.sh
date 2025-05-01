#!/bin/bash

# Create log directory
mkdir -p logs
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs/run_all_${timestamp}.log"

# List of datasets (only including those with config files)
datasets=(
    "balance"
    "fetalhealth"
)

# List of activation functions
activations=("relu" "sigmoid" "tanh")

# Start logging
{
    echo "Starting pipeline runs at $(date)"
    echo "----------------------------------------"

    # Process each dataset
    for dataset in "${datasets[@]}"; do
        echo "Processing dataset: $dataset"
        config_file="configs/config_${dataset}.yaml"
        
        if [ ! -f "$config_file" ]; then
            echo "Config file not found: $config_file"
            continue
        fi
        
        # Test different activation functions
        for activation in "${activations[@]}"; do
            echo "Testing activation: $activation"
            
            # Run with enhanced pruning enabled
            echo "Running with enhanced pruning enabled"
            python src/training/train_model.py --config "$config_file" --activation "$activation" --enhanced-pruning "true" --edge-impulse "false" || {
                echo "Error running pipeline for $dataset with $activation (enhanced pruning)"
                continue
            }
            
            # Run with enhanced pruning disabled
            echo "Running with enhanced pruning disabled"
            python src/training/train_model.py --config "$config_file" --activation "$activation" --enhanced-pruning "false" --edge-impulse "false" || {
                echo "Error running pipeline for $dataset with $activation (no enhanced pruning)"
                continue
            }
            
            echo "Completed runs for $dataset with $activation"
            echo "----------------------------------------"
        done
    done
    
    echo "All pipeline runs completed at $(date)"
} 2>&1 | tee "$log_file"

echo "Log file saved to: $log_file" 