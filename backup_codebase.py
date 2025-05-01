#!/usr/bin/env python3
import os
import shutil
import datetime
import tarfile
from pathlib import Path

def create_backup():
    # Create timestamp for backup directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Directories to backup
    dirs_to_backup = ['src', 'scripts', 'config', 'configs']
    
    # Files to backup
    files_to_backup = [
        'requirements.txt', 'README.md', 'LICENSE', '.gitignore',
        'RECIPES.md', 'run_all_datasets.sh', 'update_config.py',
        'update_configs.sh', 'fix_configs.sh', 'run_iris_pruning.py',
        'cleanup_results.sh', 'validate_onnx_models.py'
    ]
    
    print("Backing up source code...")
    # Copy directories
    for dir_name in dirs_to_backup:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(backup_dir, dir_name))
    
    print("Backing up configuration and documentation files...")
    # Copy files
    for file_name in files_to_backup:
        if os.path.exists(file_name):
            shutil.copy2(file_name, backup_dir)
    
    # Create backup info file
    print("Creating backup info file...")
    with open(os.path.join(backup_dir, 'backup_info.txt'), 'w') as f:
        f.write(f"""Backup created on: {datetime.datetime.now()}
Original directory: {os.getcwd()}
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
""")
    
    # Create compressed archive
    print("Creating compressed archive...")
    archive_name = f"{backup_dir}.tar.gz"
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(backup_dir, arcname=os.path.basename(backup_dir))
    
    # Clean up temporary backup directory
    print("Cleaning up temporary files...")
    shutil.rmtree(backup_dir)
    
    print("Backup completed successfully!")
    print(f"Backup file: {archive_name}")

if __name__ == "__main__":
    create_backup() 