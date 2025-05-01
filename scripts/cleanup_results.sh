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

# Function to create timestamped backup
backup_results() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="results_backup_${timestamp}"
    
    echo "======================================================================================"
    echo "📦 Creating backup in ${backup_dir}..."
    echo "======================================================================================"
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Directories to backup
    directories=(
        "models/original"
        "models/pruned"
        "models/onnx"
        "results/inference"
        "results/faiss_index"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            echo "Backing up $dir..."
            mkdir -p "${backup_dir}/${dir}"
            cp -r "$dir"/* "${backup_dir}/${dir}/" 2>/dev/null || true
            echo "✅ Backed up $dir"
        else
            echo "⚠️  Directory $dir not found, skipping..."
        fi
    done
    
    # Backup dataset-specific directories in results
    if [ -d "results" ]; then
        echo "Backing up dataset-specific directories in results..."
        mkdir -p "${backup_dir}/results"
        
        # Find all dataset directories in results
        for dataset_dir in results/*/; do
            if [ -d "$dataset_dir" ] && [ "$dataset_dir" != "results/inference/" ] && [ "$dataset_dir" != "results/faiss_index/" ]; then
                dataset_name=$(basename "$dataset_dir")
                echo "Backing up dataset directory: $dataset_name"
                mkdir -p "${backup_dir}/results/${dataset_name}"
                cp -r "$dataset_dir"/* "${backup_dir}/results/${dataset_name}/" 2>/dev/null || true
                echo "✅ Backed up results/${dataset_name}"
            fi
        done
    fi
    
    echo "======================================================================================"
    echo "✅ Backup complete: ${backup_dir}"
    echo "======================================================================================"
}

# Function to clean up results
cleanup_results() {
    echo "======================================================================================"
    echo "🗑️  Cleaning up existing results..."
    echo "======================================================================================"
    
    # Directories to clean
    directories=(
        "models/original"
        "models/pruned"
        "models/onnx"
        "results/inference"
        "results/faiss_index"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            echo "Cleaning $dir..."
            rm -rf "$dir"/*
            echo "✅ Cleaned $dir"
        else
            echo "⚠️  Directory $dir not found, skipping..."
        fi
    done
    
    # Clean dataset-specific directories in results
    if [ -d "results" ]; then
        echo "Cleaning dataset-specific directories in results..."
        
        # Find all dataset directories in results
        for dataset_dir in results/*/; do
            if [ -d "$dataset_dir" ] && [ "$dataset_dir" != "results/inference/" ] && [ "$dataset_dir" != "results/faiss_index/" ]; then
                dataset_name=$(basename "$dataset_dir")
                echo "Cleaning dataset directory: $dataset_name"
                rm -rf "$dataset_dir"/*
                echo "✅ Cleaned results/${dataset_name}"
            fi
        done
    fi
    
    echo "======================================================================================"
    echo "✅ All existing results cleaned"
    echo "======================================================================================"
}

# Main execution
if [ "$BACKUP" = true ] || [ "$CLEANUP" = true ]; then
    if [ "$BACKUP" = true ]; then
        backup_results
    fi
    
    if [ "$CLEANUP" = true ]; then
        cleanup_results
    fi
else
    echo "Usage: $0 [--cleanup] [--backup]"
    echo "  --cleanup  : Remove all existing results"
    echo "  --backup   : Create a backup before cleaning"
    exit 1
fi 