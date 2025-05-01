import os
import torch
import numpy as np
import onnxruntime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_onnx_model(onnx_path, test_data_path, input_features=4, num_classes=3, batch_size=32):
    """
    Validate ONNX model accuracy on test set
    
    Args:
        onnx_path: Path to the ONNX model
        test_data_path: Path to the test data
        input_features: Number of input features
        num_classes: Number of output classes
        batch_size: Batch size for inference
    
    Returns:
        float: Test accuracy
    """
    try:
        # Load test data
        test_data = np.load(test_data_path)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Create ONNX Runtime session
        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        # Process in batches
        predictions = []
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            
            # Run inference on batch
            output = session.run(None, {input_name: batch_X.astype(np.float32)})[0]
            
            # Handle different output shapes
            if len(output.shape) == 1:
                # For 1D output (num_classes,), reshape to (1, num_classes)
                output = output.reshape(1, -1)
            elif len(output.shape) == 2 and output.shape[1] == 1:
                # For binary classification
                batch_preds = (output > 0.5).astype(int)
            else:
                # For multi-class classification
                batch_preds = np.argmax(output, axis=1)
            
            predictions.extend(batch_preds)
        
        # Calculate accuracy
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test) * 100
        
        logger.info(f"ONNX Model Test Accuracy: {accuracy:.2f}%")
        return accuracy
        
    except Exception as e:
        logger.error(f"Error validating ONNX model: {str(e)}")
        return None

def main():
    # Path to test data
    test_data_path = "models/original/iris/iris_test_data.npz"
    
    # Paths to ONNX models
    original_onnx_path = "models/onnx/original/iris_model/iris_model.onnx"
    original_mcu_path = "models/onnx/original/iris_model/iris_model_mcu_compatible.onnx"
    
    pruned_onnx_path = "models/onnx/pruned/iris_model/iris_model.onnx"
    pruned_mcu_path = "models/onnx/pruned/iris_model/iris_model_mcu_compatible.onnx"
    
    # Validate original model
    logger.info("Validating original ONNX model...")
    original_accuracy = validate_onnx_model(original_onnx_path, test_data_path)
    
    # Validate original MCU compatible model
    logger.info("Validating original MCU compatible model...")
    original_mcu_accuracy = validate_onnx_model(original_mcu_path, test_data_path)
    
    # Validate pruned model
    logger.info("Validating pruned ONNX model...")
    pruned_accuracy = validate_onnx_model(pruned_onnx_path, test_data_path)
    
    # Validate pruned MCU compatible model
    logger.info("Validating pruned MCU compatible model...")
    pruned_mcu_accuracy = validate_onnx_model(pruned_mcu_path, test_data_path)
    
    # Print summary
    logger.info("\nValidation Results:")
    logger.info(f"Original ONNX Model: {original_accuracy:.2f}%")
    logger.info(f"Original MCU Compatible Model: {original_mcu_accuracy:.2f}%")
    logger.info(f"Pruned ONNX Model: {pruned_accuracy:.2f}%")
    logger.info(f"Pruned MCU Compatible Model: {pruned_mcu_accuracy:.2f}%")

if __name__ == "__main__":
    main() 