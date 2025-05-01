import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def convert_csv_to_npy(csv_path, output_dir):
    """Convert a CSV file to NPY format"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Split features and target
        # Assuming the last column is the target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        base_name = Path(csv_path).stem
        X_path = os.path.join(output_dir, f"{base_name}_X.npy")
        y_path = os.path.join(output_dir, f"{base_name}_y.npy")
        
        # Save as NPY files
        np.save(X_path, X)
        np.save(y_path, y)
        
        logger.info(f"✅ Converted {csv_path} to NPY format")
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Target shape: {y.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error converting {csv_path}: {str(e)}")
        return False

def main():
    # Get data directory
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("❌ Data directory not found")
        return
    
    # Create output directory for NPY files
    npy_dir = Path("data/npy")
    os.makedirs(npy_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("❌ No CSV files found in data directory")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to convert")
    
    # Convert each CSV file
    success_count = 0
    for csv_file in tqdm(csv_files, desc="Converting datasets"):
        if convert_csv_to_npy(csv_file, npy_dir):
            success_count += 1
    
    logger.info(f"\n✅ Successfully converted {success_count}/{len(csv_files)} datasets to NPY format")
    logger.info(f"NPY files saved to: {npy_dir}")

if __name__ == "__main__":
    main() 