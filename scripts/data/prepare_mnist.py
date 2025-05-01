import torch
import torchvision
import numpy as np
import pandas as pd
from pathlib import Path

def format_mnist():
    """Format existing MNIST dataset into a single CSV file"""
    print("\nFormatting MNIST dataset...")
    data_dir = Path("data/mnist")
    
    # Load existing MNIST data
    trainset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=False  # Don't download, just load
    )
    
    testset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=False  # Don't download, just load
    )
    
    print("Converting MNIST to single CSV format...")
    
    # Combine training and test data
    X = np.vstack([
        trainset.data.numpy().reshape(-1, 784),
        testset.data.numpy().reshape(-1, 784)
    ])
    y = np.hstack([
        trainset.targets.numpy(),
        testset.targets.numpy()
    ])
    
    # Scale pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Create combined DataFrame
    df = pd.DataFrame(
        np.column_stack((X, y)),
        columns=[f'pixel_{i}' for i in range(784)] + ['target']
    )
    
    # Save to single CSV file
    output_path = Path("data") / 'mnist.csv'
    df.to_csv(output_path, index=False)
    
    print(f"MNIST dataset formatted successfully! Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of classes: {len(df['target'].unique())}")

if __name__ == "__main__":
    format_mnist() 