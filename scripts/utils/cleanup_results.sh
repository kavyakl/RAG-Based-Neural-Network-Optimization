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
        "results/profiling"
        "results/faiss_index"
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
    
    # Backup log files
    if [ -d "results" ]; then
        echo "Backing up log files..."
        mkdir -p "${backup_dir}/results"
        cp results/*.log "${backup_dir}/results/" 2>/dev/null || true
        echo "✅ Backed up log files"
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
        "results/profiling"
        "results/faiss_index"
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
    
    # Clean log files
    if [ -d "results" ]; then
        echo "Cleaning log files..."
        rm -f results/*.log
        echo "✅ Cleaned log files"
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