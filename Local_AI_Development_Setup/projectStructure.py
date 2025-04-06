import os
from pathlib import Path

while True:
    project_name = input("Enter the Src folder Name: ")
    if project_name != '':
        break

list_of_files = [
    ".GitHub/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_preprocessing.py",
    f"{project_name}/components/model_loading.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/pipeline/inference.py",
    f"{project_name}/pipeline/training.py",
    f"{project_name}/logger.py",
    f"{project_name}/exception.py",
    "templates/index.html",
    "statics/style.css",
    "notebook/experiment.ipynb",
    "init_setup.sh",
    "requirements.txt",
    "Dockerfile",
    "app.py",
    "setup.py",
    ".env",
    ".gitignore",
]

# Creating directories and files
for filepath in list_of_files:
    filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)     # Create directories if they don't exist
    if not filepath.exists():
        filepath.touch()     # Create the file if it doesn't exist
        print(f"Created: {filepath}")
    else:
        print(f"Already exists: {filepath}")
