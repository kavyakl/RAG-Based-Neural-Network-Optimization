import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath, target_column=None, test_size=0.2, random_state=42):
    """
    Load and preprocess data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        target_column (str): Name of the target column. If None, assumes last column is target
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, num_features, num_classes)
    """
    # Ensure the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    # Load data
    df = pd.read_csv(filepath)
    
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Get number of unique classes
    unique_values = np.unique(y_encoded)
    num_classes = len(unique_values)
    
    # Get number of input features
    num_features = X.shape[1]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, num_features, num_classes

def prepare_data_for_training(data):
    """
    Convert numpy arrays to PyTorch tensors and format them for training.
    
    Args:
        data (tuple): (X_train, X_test, y_train, y_test, num_features, num_classes)
        
    Returns:
        tuple: ((X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor))
    """
    X_train, X_test, y_train, y_test, _, num_classes = data
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # For both binary and multiclass classification, use LongTensor for labels
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)

def save_dataset(save_dir, dataset_name, X_train, X_test, y_train, y_test):
    """
    Save dataset splits to npz files.
    
    Args:
        save_dir (str): Directory to save the files
        dataset_name (str): Name of the dataset
        X_train, X_test, y_train, y_test: Data to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(save_dir, f"{dataset_name}_train_data.npz")
    test_path = os.path.join(save_dir, f"{dataset_name}_test_data.npz")
    
    # Save training data
    if isinstance(X_train, torch.Tensor):
        np.savez(train_path, X_train=X_train.numpy(), y_train=y_train.numpy())
    else:
        np.savez(train_path, X_train=X_train, y_train=y_train)
    
    # Save test data
    if isinstance(X_test, torch.Tensor):
        np.savez(test_path, X_test=X_test.numpy(), y_test=y_test.numpy())
    else:
        np.savez(test_path, X_test=X_test, y_test=y_test)
    
    return train_path, test_path

def load_saved_dataset(train_path, test_path):
    """
    Load dataset from saved npz files.
    
    Args:
        train_path (str): Path to the training data npz file
        test_path (str): Path to the test data npz file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    X_train, y_train = train_data['X_train'], train_data['y_train']
    X_test, y_test = test_data['X_test'], test_data['y_test']
    
    return X_train, X_test, y_train, y_test 