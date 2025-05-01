"""
Script to prepare a binary classification MNIST dataset using scikit-learn's digits dataset.
This is a smaller dataset than the full MNIST but follows the same format.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_binary_digits_dataset(digit=0, output_dir="data"):
    """
    Prepare a binary classification dataset from scikit-learn's digits dataset.
    
    Args:
        digit (int): The digit to use as the positive class (0-9)
        output_dir (str): Directory to save the processed dataset
    """
    logger.info(f"Preparing binary classification for digit {digit} using scikit-learn's digits dataset...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load scikit-learn's digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Scale pixel values to [0, 1]
    X = X / 16.0  # Values in digits dataset are 0-16
    
    # Convert to binary classification: target_digit (1) vs all other digits (0)
    binary_y = (y == digit).astype(int)
    
    # Create a balanced dataset
    pos_indices = np.where(binary_y == 1)[0]
    neg_indices = np.where(binary_y == 0)[0]
    
    # Sample an equal number from each class
    np.random.seed(42)  # For reproducibility
    n_samples = min(len(pos_indices), len(neg_indices))
    pos_sample = np.random.choice(pos_indices, n_samples, replace=False)
    neg_sample = np.random.choice(neg_indices, n_samples, replace=False)
    
    # Combine samples
    indices = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(indices)
    
    # Select data
    X_balanced = X[indices]
    y_balanced = binary_y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(
        data=np.column_stack((X_balanced, y_balanced)),
        columns=[f'pixel_{i}' for i in range(X.shape[1])] + ['target']  # Change 'label' to 'target' to match expected format
    )
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"digits_binary_digit{digit}.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Binary digits dataset saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Positive samples (digit {digit}): {sum(y_balanced == 1)}")
    logger.info(f"Negative samples (other digits): {sum(y_balanced == 0)}")
    
    return output_path

if __name__ == "__main__":
    # Create a binary classification dataset for digit 0
    create_binary_digits_dataset(digit=0) 