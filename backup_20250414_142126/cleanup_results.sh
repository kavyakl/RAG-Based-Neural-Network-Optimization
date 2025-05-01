#!/bin/bash

# Parse command line arguments
CLEANUP=false
BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to backup results
backup_results() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="backups/backup_${timestamp}"
    
    echo "Creating backup in ${backup_dir}..."
    mkdir -p "${backup_dir}"
    
    # Backup directories
    for dir in models/original models/pruned models/onnx results/inference results/faiss_index logs; do
        if [ -d "$dir" ]; then
            echo "Backing up $dir..."
            cp -r "$dir" "${backup_dir}/"
        fi
    done
    
    # Backup dataset-specific directories in results
    for dir in results/*/; do
        if [ -d "$dir" ] && [ "$dir" != "results/inference/" ] && [ "$dir" != "results/faiss_index/" ]; then
            echo "Backing up $dir..."
            cp -r "$dir" "${backup_dir}/"
        fi
    done
    
    echo "Backup completed in ${backup_dir}"
}

# Function to cleanup results
cleanup_results() {
    echo "Cleaning up results..."
    
    # Clean directories
    for dir in models/original models/pruned models/onnx results/inference results/faiss_index logs; do
        if [ -d "$dir" ]; then
            echo "Cleaning $dir..."
            rm -rf "${dir:?}"/*
        fi
    done
    
    # Clean dataset-specific directories in results
    for dir in results/*/; do
        if [ -d "$dir" ] && [ "$dir" != "results/inference/" ] && [ "$dir" != "results/faiss_index/" ]; then
            echo "Cleaning $dir..."
            rm -rf "${dir:?}"/*
        fi
    done
    
    echo "Cleanup completed"
}

# Main execution
if [ "$BACKUP" = true ]; then
    backup_results
fi

if [ "$CLEANUP" = true ]; then
    cleanup_results
fi 