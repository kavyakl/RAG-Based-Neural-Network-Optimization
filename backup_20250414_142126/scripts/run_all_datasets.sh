#!/bin/bash

# Run the neural network optimization pipeline for all available datasets
echo "======================================================================================"
echo "Starting neural network optimization pipeline for all available datasets"
echo "======================================================================================"

# Detect OS for sed compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE="sed -i ''"
else
    SED_INPLACE="sed -i"
fi

# Function to enable/disable ONNX export in config files
update_export_settings() {
    dataset=$1
    config_file="config/config_${dataset}.yaml"
    
    echo "Updating export settings for ${dataset}..."
    # Use sed to update export settings
    $SED_INPLACE 's/export:\n  enabled: true/export:\n  enabled: false/' "$config_file"
    $SED_INPLACE 's/edge_impulse:\n    enabled: true/edge_impulse:\n    enabled: false/' "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Export settings updated for ${dataset}"
    else
        echo "❌ Failed to update export settings for ${dataset}"
        return 1
    fi
}

# Create a function to run the pipeline with a specific configuration
run_pipeline() {
    dataset=$1
    config_file="config/config_${dataset}.yaml"
    data_file="data/${dataset}.csv"
    
    echo "======================================================================================"
    echo "Running pipeline for ${dataset} dataset using configuration: ${config_file}"
    echo "======================================================================================"
    
    # Check if data file exists
    if [ ! -f "$data_file" ]; then
        echo "❌ Skipping ${dataset}: Data file ${data_file} not found."
        return 1
    fi
    
    # Run the pipeline with the config file
    python scripts/pipeline/pipeline_export_onnx_for_ei.py --config ${config_file}
    
    # Check if the pipeline ran successfully
    if [ $? -eq 0 ]; then
        echo "✅ Completed pipeline for ${dataset} dataset successfully"
    else
        echo "❌ Pipeline for ${dataset} dataset failed with error code $?"
        return 1
    fi
    echo ""
}

# Find all config files in the config directory
config_files=(config/config_*.yaml)

# Extract dataset names from config files
datasets=()
for config_file in "${config_files[@]}"; do
    # Skip the template file
    if [[ "$config_file" == "config/config_template.yaml" ]]; then
        continue
    fi
    
    # Extract dataset name from filename (remove config/config_ and .yaml)
    dataset_name=$(basename "$config_file" | sed 's/config_//' | sed 's/\.yaml//')
    datasets+=("$dataset_name")
done

# Sort datasets alphabetically
IFS=$'\n' sorted_datasets=($(sort <<<"${datasets[*]}"))
unset IFS

# Display datasets to be processed
echo "Found ${#sorted_datasets[@]} datasets to process:"
for dataset in "${sorted_datasets[@]}"; do
    data_file="data/${dataset}.csv"
    if [ -f "$data_file" ]; then
        echo "  - ${dataset} (data file exists)"
    else
        echo "  - ${dataset} (❌ data file missing)"
    fi
done
echo ""

# Process each dataset
total_datasets=${#sorted_datasets[@]}
current=0
processed=0
skipped=0

for dataset in "${sorted_datasets[@]}"; do
    current=$((current + 1))
    echo "Processing dataset ${current}/${total_datasets}: ${dataset}"
    
    if [ -f "config/config_${dataset}.yaml" ]; then
        # Update export settings before running pipeline
        update_export_settings ${dataset}
        if [ $? -eq 0 ]; then
            run_pipeline ${dataset}
            if [ $? -eq 0 ]; then
                processed=$((processed + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            echo "❌ Skipping ${dataset} due to configuration error"
            skipped=$((skipped + 1))
        fi
    else
        echo "❌ Skipping ${dataset}: Configuration file not found."
        skipped=$((skipped + 1))
    fi
done

echo "======================================================================================"
echo "Pipeline Summary:"
echo "  - Total datasets found: ${total_datasets}"
echo "  - Successfully processed: ${processed}"
echo "  - Skipped/Failed: ${skipped}"
echo "======================================================================================" 
