#!/bin/bash

# Create timestamp for backup directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Copy main directories
echo "Backing up source code..."
cp -r src "${BACKUP_DIR}/"
cp -r scripts "${BACKUP_DIR}/"
cp -r config "${BACKUP_DIR}/"
cp -r configs "${BACKUP_DIR}/"

# Copy important files
echo "Backing up configuration and documentation files..."
cp requirements.txt "${BACKUP_DIR}/"
cp README.md "${BACKUP_DIR}/"
cp LICENSE "${BACKUP_DIR}/"
cp .gitignore "${BACKUP_DIR}/"
cp RECIPES.md "${BACKUP_DIR}/"
cp run_all_datasets.sh "${BACKUP_DIR}/"
cp update_config.py "${BACKUP_DIR}/"
cp update_configs.sh "${BACKUP_DIR}/"
cp fix_configs.sh "${BACKUP_DIR}/"
cp run_iris_pruning.py "${BACKUP_DIR}/"
cp cleanup_results.sh "${BACKUP_DIR}/"
cp validate_onnx_models.py "${BACKUP_DIR}/"

# Create a backup info file
echo "Creating backup info file..."
cat > "${BACKUP_DIR}/backup_info.txt" << EOF
Backup created on: $(date)
Original directory: $(pwd)
Backup contents:
- src/ (source code)
- scripts/ (pipeline scripts)
- config/ (configuration files)
- configs/ (additional configurations)
- Various utility scripts and documentation

Excluded directories:
- .venv/ (Python virtual environment)
- .vscode/ (VS Code settings)
- .cursor/ (Cursor IDE settings)
- .dist/ (Distribution files)
- backups/ (Previous backups)
- logs/ (Log files)
- models/ (Generated models)
- results/ (Generated results)
- data/ (Dataset files)
- python-sdk/ (SDK files)
- reports/ (Generated reports)
EOF

# Create a compressed archive
echo "Creating compressed archive..."
tar -czf "${BACKUP_DIR}.tar.gz" "${BACKUP_DIR}"

# Clean up the temporary backup directory
echo "Cleaning up temporary files..."
rm -rf "${BACKUP_DIR}"

echo "Backup completed successfully!"
echo "Backup file: ${BACKUP_DIR}.tar.gz" 