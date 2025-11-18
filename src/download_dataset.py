import kagglehub
import os
from pathlib import Path

def main():
    # Download latest version
    path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

    print("Path to dataset files:", path)

    # Create .env file with the dataset path
    env_file = Path(__file__).parent.parent / ".env"
    with open(env_file, "w") as f:
        f.write(f"DATASET_PATH={path}\n")

    print(f"\nCreated .env file at: {env_file}")
    print(f"Set DATASET_PATH={path}")

    # List files in the dataset directory
    print("\nDataset contents:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "main":
    main()