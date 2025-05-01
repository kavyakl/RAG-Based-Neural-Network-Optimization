import os
import shutil
import logging
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def generate_arduino_sketch(
    dataset_name: str,
    model_type: str,
    model_header_path: str,
    test_data_header_path: str,
    output_dir: str,
    template_path: str = "src/deployment/arduino_template.ino"
) -> str:
    """
    Generate an Arduino sketch for the specified model and dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        model_type (str): Type of model ('original' or 'pruned')
        model_header_path (str): Path to the C header file containing model weights
        test_data_header_path (str): Path to the C header file containing test data
        output_dir (str): Directory to save the generated sketch
        template_path (str): Path to the Arduino template file
        
    Returns:
        str: Path to the generated Arduino sketch directory
    """
    logger.info(f"Generating Arduino sketch for {model_type} model on {dataset_name} dataset")
    
    # Create sketch directory
    sketch_dir = os.path.join(output_dir, f"{dataset_name}_{model_type}_sketch")
    os.makedirs(sketch_dir, exist_ok=True)
    
    # Get the model and test data header filenames (without path)
    model_header_filename = os.path.basename(model_header_path)
    test_data_header_filename = os.path.basename(test_data_header_path)
    
    # Copy header files to sketch directory
    shutil.copy(model_header_path, os.path.join(sketch_dir, model_header_filename))
    shutil.copy(test_data_header_path, os.path.join(sketch_dir, test_data_header_filename))
    
    # Read the template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders
    sketch_content = template_content.replace("{{MODEL_HEADER}}", model_header_filename)
    sketch_content = sketch_content.replace("{{TEST_DATA_HEADER}}", test_data_header_filename)
    sketch_content = sketch_content.replace("{{DATASET_NAME}}", dataset_name)
    sketch_content = sketch_content.replace("{{MODEL_TYPE}}", model_type)
    
    # Write the sketch
    sketch_path = os.path.join(sketch_dir, f"{dataset_name}_{model_type}_sketch.ino")
    with open(sketch_path, 'w') as f:
        f.write(sketch_content)
    
    logger.info(f"Generated Arduino sketch at {sketch_path}")
    
    # Create README.md with instructions
    readme_path = os.path.join(sketch_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# {dataset_name.capitalize()} {model_type.capitalize()} Neural Network for RP2040\n\n")
        f.write("This Arduino sketch implements a neural network for inference on the Raspberry Pi Pico (RP2040) microcontroller.\n\n")
        
        f.write("## Setup Instructions\n\n")
        f.write("1. Install the Arduino IDE (version 2.0 or later recommended)\n")
        f.write("2. Install the Arduino Pico core following the instructions at: https://github.com/earlephilhower/arduino-pico\n")
        f.write("3. Open the sketch file in the Arduino IDE\n")
        f.write("4. Select the Raspberry Pi Pico board from the Tools > Board menu\n")
        f.write("5. Connect your Pico to your computer via USB\n")
        f.write("6. Click Upload to compile and flash the sketch to the board\n")
        f.write("7. Open the Serial Monitor (Tools > Serial Monitor) with baud rate 115200 to view the inference results\n\n")
        
        f.write("## Files\n\n")
        f.write(f"- `{os.path.basename(sketch_path)}`: Main Arduino sketch file\n")
        f.write(f"- `{model_header_filename}`: C header containing neural network weights\n")
        f.write(f"- `{test_data_header_filename}`: C header containing test data samples\n\n")
        
        f.write("## Performance\n\n")
        f.write("The sketch will first run inference on the test samples and output accuracy information. Then it will run a 1-minute continuous inference benchmark and report:\n")
        f.write("- Total number of inferences performed in 1 minute\n")
        f.write("- Average inference time per sample (microseconds)\n")
        f.write("- Inferences per second (throughput)\n")
        f.write("- Approximate memory usage\n\n")
        
        f.write("## Benchmarking\n\n")
        f.write("The sketch includes a 1-minute continuous inference benchmark that provides real-world performance metrics. After the initial test, you can re-run the benchmark at any time by sending any character through the Serial Monitor.\n\n")
        
        f.write("## Notes\n\n")
        f.write("- This model was trained, pruned, and quantized using the Neural Network Optimization Pipeline\n")
        f.write("- The implementation uses 8-bit quantized weights for reduced memory usage\n")
        f.write("- For better performance, consider implementing fixed-point arithmetic or integer-only quantization\n")
    
    logger.info(f"Generated README at {readme_path}")
    
    return sketch_dir

def generate_all_arduino_sketches(
    datasets: List[str],
    model_types: List[str],
    models_output_dir: str = "models/arduino",
    output_dir: str = "models/arduino"
) -> Dict[str, Dict[str, str]]:
    """
    Generate Arduino sketches for all datasets and model types.
    
    Args:
        datasets (List[str]): List of dataset names
        model_types (List[str]): List of model types ('original', 'pruned')
        models_output_dir (str): Directory containing the exported model headers
        output_dir (str): Directory to save the generated sketches
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary of generated sketch paths
    """
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for model_type in model_types:
            # Paths to the model and test data headers
            model_header_path = os.path.join(models_output_dir, dataset, f"{dataset}_{model_type}_weights.h")
            test_data_header_path = os.path.join(models_output_dir, dataset, f"{dataset}_test_data.h")
            
            # Check if the required files exist
            if not os.path.exists(model_header_path):
                logger.warning(f"Model header not found: {model_header_path}")
                continue
                
            if not os.path.exists(test_data_header_path):
                logger.warning(f"Test data header not found: {test_data_header_path}")
                continue
            
            # Generate the sketch
            sketch_dir = generate_arduino_sketch(
                dataset_name=dataset,
                model_type=model_type,
                model_header_path=model_header_path,
                test_data_header_path=test_data_header_path,
                output_dir=output_dir
            )
            
            results[dataset][model_type] = sketch_dir
    
    return results 