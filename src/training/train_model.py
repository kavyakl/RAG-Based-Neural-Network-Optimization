import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.categorical_model import CategoricalModel
from src.models.binary_model import BinaryModel
from src.utils.data_handling import load_and_preprocess_data, prepare_data_for_training, save_dataset
from src.utils.metrics import calculate_accuracy, save_model_metrics
from src.utils.model_io import save_model_weights, save_pytorch_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple) and all(isinstance(i, np.integer) for i in obj):
            return tuple(int(i) for i in obj)
        return super(NumpyEncoder, self).default(obj)

def train_model(config: Dict[str, Any], dataset_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Train a neural network model with the given configuration."""
    logger.info(f"Training model for dataset: {dataset_name}")
    
    # Load and preprocess data
    data = load_and_preprocess_data(
        config['dataset']['data_path'],
        config['dataset']['target_column'],
        test_size=0.2,
        random_state=42
    )
    
    # Convert data to PyTorch tensors
    train_data, test_data = prepare_data_for_training(data)
    
    # Create model directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config['training']['save_dir'], dataset_name, f"{dataset_name}_original_{config['model']['hidden_neurons']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save training and test data in the model directory
    train_data_path = os.path.join(model_dir, f"{dataset_name}_train_data.npz")
    test_data_path = os.path.join(model_dir, f"{dataset_name}_test_data.npz")
    
    np.savez(train_data_path,
             X_train=train_data[0].numpy(),
             y_train=train_data[1].numpy())
    np.savez(test_data_path,
             X_test=test_data[0].numpy(),
             y_test=test_data[1].numpy())
    
    logger.info(f"Saved training data to {train_data_path}")
    logger.info(f"Saved test data to {test_data_path}")
    
    # Create model
    if config['dataset']['num_classes'] == 2:
        model = BinaryModel(
            input_size=config['dataset']['input_features'],
            hidden_size=config['model']['hidden_neurons'],
            activation=config['model']['activation']
        )
    else:
        model = CategoricalModel(
            input_size=config['dataset']['input_features'],
            hidden_size=config['model']['hidden_neurons'],
            output_size=config['dataset']['num_classes'],
            activation=config['model']['activation']
        )
    
    # Train model
    if config['dataset']['num_classes'] == 2:
        criterion = nn.BCELoss()
        # Reshape target tensor for binary classification
        train_data = (train_data[0], train_data[1].float().unsqueeze(1))
        test_data = (test_data[0], test_data[1].float().unsqueeze(1))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, return_hidden=False)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_data[0], return_hidden=False)
        train_pred = torch.argmax(train_outputs, dim=1)
        train_acc = (train_pred == train_data[1]).float().mean()
        
        test_outputs = model(test_data[0], return_hidden=False)
        test_pred = torch.argmax(test_outputs, dim=1)
        test_acc = (test_pred == test_data[1]).float().mean()
    
    logger.info(f"Train Accuracy: {train_acc:.2%}")
    logger.info(f"Test Accuracy: {test_acc:.2%}")
    
    # Save model and metrics
    model_save_path = os.path.join(model_dir, f"{dataset_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    
    # Save model info
    model_info = {
        'model_type': 'original',
        'dataset': dataset_name,
        'hidden_neurons': config['model']['hidden_neurons'],
        'activation': config['model']['activation'],
        'timestamp': timestamp,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'input_features': model.hidden.in_features,
        'output_classes': model.output.out_features
    }
    
    with open(os.path.join(model_dir, f"{dataset_name}_model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    metrics = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'timestamp': timestamp
    }
    
    metrics_path = os.path.join('results', dataset_name, f"{dataset_name}_original_metrics_{timestamp}.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return model, metrics

def main():
    """
    Main function to run the training process independently.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a neural network model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Train model
    results = train_model(config, config['dataset']['name'])
    
    logger.info("Training completed successfully")
    logger.info(f"Model saved to {os.path.dirname(results[0])}")

if __name__ == '__main__':
    main() 