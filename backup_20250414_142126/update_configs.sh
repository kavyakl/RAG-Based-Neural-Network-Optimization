#!/bin/bash

# Script to update configuration files with different activation functions
# Usage: ./update_configs.sh [--cleanup] [--enhanced-pruning] [--no-edge-impulse]

set -e

# Create log directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs/update_configs_${timestamp}.log"

# Default values
CLEANUP=false
ENHANCED_PRUNING=true  # Set to true by default for this run
EDGE_IMPULSE=false     # Set to false by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --enhanced-pruning)
            ENHANCED_PRUNING=true
            shift
            ;;
        --no-edge-impulse)
            EDGE_IMPULSE=false
            shift
            ;;
        --edge-impulse)
            EDGE_IMPULSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log current settings
echo "Current settings:"
echo "Cleanup: $CLEANUP"
echo "Enhanced Pruning: $ENHANCED_PRUNING"
echo "Edge Impulse Deployment: $EDGE_IMPULSE"

# Function to create backup
create_backup() {
    echo "Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/backup_${timestamp}"
    mkdir -p "$backup_dir"
    
    # Backup models and results
    if [ -d "models" ]; then
        cp -r models "$backup_dir/"
    fi
    
    if [ -d "results" ]; then
        cp -r results "$backup_dir/"
    fi
    
    if [ -d "onnx" ]; then
        cp -r onnx "$backup_dir/"
    fi
    
    echo "Backup created in $backup_dir"
}

# Function to clean up results and models
cleanup() {
    echo "Cleaning up results and models..."
    
    # Remove results directory
    if [ -d "results" ]; then
        rm -rf results
    fi
    
    # Remove models directory
    if [ -d "models" ]; then
        rm -rf models
    fi
    
    # Remove onnx directory
    if [ -d "onnx" ]; then
        rm -rf onnx
    fi
    
    echo "Cleanup completed"
}

# Function to run pipeline with a specific config
run_pipeline() {
    local config_file=$1
    echo "Running pipeline with config: $config_file"
    python scripts/pipeline/pipeline_export_onnx_for_ei.py --config "$config_file"
}

# Start logging
{
    echo "Starting update_configs.sh at $(date)"
    echo "Log file: $log_file"
    echo "----------------------------------------"
    echo "Enhanced pruning: $ENHANCED_PRUNING"
    echo "Edge Impulse Deployment: $EDGE_IMPULSE"

    # Check if cleanup flag is provided
    if [ "$CLEANUP" = true ]; then
        create_backup
        cleanup
    fi

    # Create necessary directories
    mkdir -p models/original
    mkdir -p models/pruned
    mkdir -p results/tagging
    mkdir -p results/tagging_index
    mkdir -p results/profiling
    mkdir -p results/inference

    # Get all dataset config files
    config_files=(config/config_*.yaml)

    # Process each dataset
    for config_file in "${config_files[@]}"; do
        dataset_name=$(basename "$config_file" .yaml | sed 's/config_//')
        echo "Processing dataset: $dataset_name"
        
        # Test different activation functions
        for activation in "relu" "sigmoid" "tanh"; do
            echo "Testing activation: $activation"
            
            # Test with enhanced pruning enabled
            echo "Running with enhanced pruning enabled"
            python update_config.py --config "$config_file" --activation "$activation" --enhanced-pruning "true" --edge-impulse "false" --run-pipeline
            
            # Test with enhanced pruning disabled
            echo "Running with enhanced pruning disabled"
            python update_config.py --config "$config_file" --activation "$activation" --enhanced-pruning "false" --edge-impulse "false" --run-pipeline
            
            # Test with Edge Impulse deployment enabled
            echo "Running with Edge Impulse deployment enabled"
            python update_config.py --config "$config_file" --activation "$activation" --enhanced-pruning "false" --edge-impulse "true" --run-pipeline
        done
    done

    echo "All configurations have been processed"
    echo "----------------------------------------"
    echo "Script completed at $(date)"
} 2>&1 | tee "$log_file"

echo "Log file saved to: $log_file" 