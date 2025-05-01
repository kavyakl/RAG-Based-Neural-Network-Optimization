import os
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_config_file(config_path):
    """Update a config file to remove quantization and deployment sections."""
    try:
        # Load the config file
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        # Remove quantization and deployment sections if they exist
        if 'quantization' in config:
            logger.info(f"Removing quantization section from {config_path}")
            del config['quantization']
        
        if 'deployment' in config:
            logger.info(f"Removing deployment section from {config_path}")
            del config['deployment']
        
        # Save the updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Successfully updated {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating {config_path}: {str(e)}")
        return False

def main():
    # Get the config directory
    config_dir = Path('config')
    
    # Get all YAML files in the config directory
    config_files = list(config_dir.glob('*.yaml'))
    
    # Skip the template file if it exists
    config_files = [f for f in config_files if f.name != 'config_template.yaml']
    
    logger.info(f"Found {len(config_files)} config files to update")
    
    # Update each config file
    success_count = 0
    for config_file in config_files:
        if update_config_file(str(config_file)):
            success_count += 1
    
    logger.info(f"Successfully updated {success_count} out of {len(config_files)} config files")

if __name__ == "__main__":
    main() 