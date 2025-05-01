#!/bin/bash

# Add cleanup and backup flags
CLEANUP=false
BACKUP=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cleanup) CLEANUP=true ;;
        --backup) BACKUP=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to update activation function and control inference/export in all config files
update_config() {
    local activation=$1
    local enable_inference=$2
    local enable_export=$3
    local enable_edge_impulse=$4
    local enable_training=$5
    local enable_pruning=$6
    
    echo "======================================================================================"
    echo "🔄 Updating configuration:"
    echo "   - Activation function: ${activation}"
    echo "   - Inference: ${enable_inference}"
    echo "   - Export: ${enable_export}"
    echo "   - Edge Impulse: ${enable_edge_impulse}"
    echo "   - Training: ${enable_training}"
    echo "   - Pruning: ${enable_pruning}"
    echo "======================================================================================"
    
    # Update all config files
    for config_file in config/config_*.yaml; do
        if [[ "$config_file" == "config/config_template.yaml" ]]; then
            continue
        fi
        
        echo "📝 Updating $config_file..."
        
        # First, read all dataset-specific values
        input_features=$(grep "input_features:" "$config_file" | awk '{print $2}')
        num_classes=$(grep "num_classes:" "$config_file" | awk '{print $2}')
        hidden_neurons=$(grep "hidden_neurons:" "$config_file" | awk '{print $2}')
        dataset_name=$(grep "name:" "$config_file" | awk '{print $2}')
        dataset_path=$(grep "path:" "$config_file" | awk '{print $2}')
        save_dir=$(grep "save_dir:" "$config_file" | awk '{print $2}')
        results_dir=$(grep "results_dir:" "$config_file" | awk '{print $2}')
        log_file=$(grep "file:" "$config_file" | awk '{print $2}')
        
        # Update the activation function while preserving other values
        sed -i '' "s/activation: [a-zA-Z]*/activation: ${activation}/g" "$config_file"
        
        # Update inference setting
        if [ "$enable_inference" = "true" ]; then
            sed -i '' "/inference:/,/enabled:/ s/enabled:.*/enabled: true/g" "$config_file"
        else
            sed -i '' "/inference:/,/enabled:/ s/enabled:.*/enabled: false/g" "$config_file"
        fi
        
        # Update export setting
        if [ "$enable_export" = "true" ]; then
            sed -i '' "/export:/,/enabled:/ s/enabled:.*/enabled: true/g" "$config_file"
        else
            sed -i '' "/export:/,/enabled:/ s/enabled:.*/enabled: false/g" "$config_file"
        fi
        
        # Update Edge Impulse setting
        if [ "$enable_edge_impulse" = "true" ]; then
            sed -i '' "/edge_impulse:/,/enabled:/ s/enabled:.*/enabled: true/g" "$config_file"
        else
            sed -i '' "/edge_impulse:/,/enabled:/ s/enabled:.*/enabled: false/g" "$config_file"
        fi
        
        # Update training setting
        if [ "$enable_training" = "true" ]; then
            sed -i '' "/training:/,/enabled:/ s/enabled:.*/enabled: true/g" "$config_file"
        else
            sed -i '' "/training:/,/enabled:/ s/enabled:.*/enabled: false/g" "$config_file"
        fi
        
        # Update pruning setting
        if [ "$enable_pruning" = "true" ]; then
            sed -i '' "/pruning:/,/enabled:/ s/enabled:.*/enabled: true/g" "$config_file"
        else
            sed -i '' "/pruning:/,/enabled:/ s/enabled:.*/enabled: false/g" "$config_file"
        fi
        
        # Ensure all dataset-specific values are preserved
        sed -i '' "s/input_features: [0-9]*/input_features: ${input_features}/g" "$config_file"
        sed -i '' "s/num_classes: [0-9]*/num_classes: ${num_classes}/g" "$config_file"
        sed -i '' "s/hidden_neurons: [0-9]*/hidden_neurons: ${hidden_neurons}/g" "$config_file"
        sed -i '' "s/name: [a-zA-Z_]*/name: ${dataset_name}/g" "$config_file"
        sed -i '' "s|path: .*|path: ${dataset_path}|g" "$config_file"
        sed -i '' "s|save_dir: .*|save_dir: ${save_dir}|g" "$config_file"
        sed -i '' "s|results_dir: .*|results_dir: ${results_dir}|g" "$config_file"
        sed -i '' "s|file: .*|file: ${log_file}|g" "$config_file"
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully updated $config_file"
            echo "   Dataset-specific parameters preserved:"
            echo "   - Dataset name: ${dataset_name}"
            echo "   - Dataset path: ${dataset_path}"
            echo "   - Input features: ${input_features}"
            echo "   - Number of classes: ${num_classes}"
            echo "   - Hidden neurons: ${hidden_neurons}"
            echo "   - Save directory: ${save_dir}"
            echo "   - Results directory: ${results_dir}"
            echo "   - Log file: ${log_file}"
            echo "   - New activation: ${activation}"
            echo "   - Inference enabled: ${enable_inference}"
            echo "   - Export enabled: ${enable_export}"
            echo "   - Edge Impulse enabled: ${enable_edge_impulse}"
            echo "   - Training enabled: ${enable_training}"
            echo "   - Pruning enabled: ${enable_pruning}"
        else
            echo "❌ Failed to update $config_file"
        fi
    done
}

# Function to run the pipeline
run_pipeline() {
    echo "======================================================================================"
    echo "🚀 Running pipeline with current configuration"
    echo "======================================================================================"
./run_all_datasets.sh
}

# Main execution
echo "======================================================================================"
echo "🔄 Starting configuration testing pipeline"
echo "======================================================================================"

# If cleanup or backup flags are set, run the cleanup script
if [ "$CLEANUP" = true ] || [ "$BACKUP" = true ]; then
    echo "======================================================================================"
    echo "🧹 Running cleanup/backup operations..."
    echo "======================================================================================"
    
    # Make sure the cleanup script is executable
    chmod +x scripts/cleanup_results.sh
    
    # Run the cleanup script with the appropriate flags
    ./scripts/cleanup_results.sh --cleanup=$CLEANUP --backup=$BACKUP
    
    echo "======================================================================================"
    echo "✅ Cleanup/backup operations completed"
    echo "======================================================================================"
fi

# Default settings
ENABLE_INFERENCE=true
ENABLE_EXPORT=true
ENABLE_EDGE_IMPULSE=false
ENABLE_TRAINING=true
ENABLE_PRUNING=true

# Run through each activation function once
for activation in "relu" "sigmoid" "tanh"; do
    echo "======================================================================================"
    echo "🔄 Testing with activation function: ${activation}"
    echo "======================================================================================"
    
    # Update configuration
    update_config "$activation" "$ENABLE_INFERENCE" "$ENABLE_EXPORT" "$ENABLE_EDGE_IMPULSE" "$ENABLE_TRAINING" "$ENABLE_PRUNING"
    
    # Run the pipeline
    run_pipeline
    
    echo "======================================================================================"
    echo "✅ Completed testing with ${activation} activation function"
    echo "======================================================================================"
    echo ""
done

echo "======================================================================================"
echo "✅ All configurations tested successfully"
echo "======================================================================================"


