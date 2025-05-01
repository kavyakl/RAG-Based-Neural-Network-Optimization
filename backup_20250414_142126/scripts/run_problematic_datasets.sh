#!/bin/bash

# Array of problematic datasets
datasets=("creditcard" "customers" "heart")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Run the pipeline for each dataset
    python pipeline_export_onnx_for_ei.py --config "config/config_${dataset}.yaml"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset"
    else
        echo "Error processing $dataset"
    fi
    
    echo "----------------------------------------"
done 