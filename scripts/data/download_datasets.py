import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def create_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_mnist():
    """Download and prepare MNIST dataset"""
    print("\nDownloading MNIST dataset...")
    data_dir = Path("data/mnist")
    create_directory(data_dir)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training set
    trainset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    print("Converting MNIST to CSV format...")
    
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
    df.to_csv(Path("data") / 'mnist.csv', index=False)
    
    print("MNIST dataset prepared successfully!")

def download_cifar10():
    """Download and prepare CIFAR-10 dataset"""
    print("\nDownloading CIFAR-10 dataset...")
    data_dir = Path("data/cifar10")
    create_directory(data_dir)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download training set
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    # Convert to CSV format
    print("Converting CIFAR-10 to CSV format...")
    
    # Training data
    X_train = trainset.data.reshape(-1, 3072)  # 32x32x3 = 3072
    y_train = np.array(trainset.targets)
    train_df = pd.DataFrame(
        np.column_stack((X_train, y_train)),
        columns=[f'pixel_{i}' for i in range(3072)] + ['target']
    )
    train_df.to_csv(data_dir / 'cifar10_train.csv', index=False)
    
    # Test data
    X_test = testset.data.reshape(-1, 3072)
    y_test = np.array(testset.targets)
    test_df = pd.DataFrame(
        np.column_stack((X_test, y_test)),
        columns=[f'pixel_{i}' for i in range(3072)] + ['target']
    )
    test_df.to_csv(data_dir / 'cifar10_test.csv', index=False)
    
    print("CIFAR-10 dataset prepared successfully!")

def download_california_housing():
    """Download and prepare California Housing dataset"""
    print("\nDownloading California Housing dataset...")
    data_dir = Path("data")
    create_directory(data_dir)
    
    # Load dataset
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(
        data=np.c_[housing.data, housing.target],
        columns=list(housing.feature_names) + ['target']
    )
    
    # Scale features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    # Save to CSV
    df.to_csv(data_dir / 'california_housing.csv', index=False)
    print("California Housing dataset prepared successfully!")

def download_energy_efficiency():
    """Download and prepare Energy Efficiency dataset"""
    print("\nPreparing Energy Efficiency dataset...")
    data_dir = Path("data")
    create_directory(data_dir)
    
    # Load dataset (you need to manually download this from UCI ML Repository)
    print("NOTE: Please download the Energy Efficiency dataset manually from:")
    print("https://archive.ics.uci.edu/ml/datasets/Energy+efficiency")
    print("and place it in the data directory as 'ENB2012_data.xlsx'")
    
    # Check if file exists
    data_file = data_dir / 'ENB2012_data.xlsx'
    if data_file.exists():
        # Load and prepare dataset
        df = pd.read_excel(data_file)
        
        # Rename columns
        columns = [
            'relative_compactness', 'surface_area', 'wall_area', 'roof_area',
            'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution',
            'heating_load', 'cooling_load'
        ]
        df.columns = columns
        
        # Scale features
        scaler = StandardScaler()
        df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])
        
        # Save to CSV
        df.to_csv(data_dir / 'energy_efficiency.csv', index=False)
        print("Energy Efficiency dataset prepared successfully!")
    else:
        print("Energy Efficiency dataset file not found!")
        print("Please download it manually and try again.")

def main():
    """Download and prepare all datasets"""
    # Create main data directory
    create_directory("data")
    
    # Download datasets
    download_mnist()
    download_cifar10()
    download_california_housing()
    download_energy_efficiency()
    
    print("\nAll datasets prepared successfully!")
    print("\nNote: For Energy Efficiency dataset, manual download is required.")
    print("Please visit: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency")
    print("Download the dataset and place 'ENB2012_data.xlsx' in the data directory.")

if __name__ == "__main__":
    main() 