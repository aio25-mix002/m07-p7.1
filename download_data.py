import os
import argparse
import shutil
import zipfile
from pathlib import Path

# Kaggle dataset paths
KAGGLE_DATA_TRAIN = '/kaggle/input/action-video/data/data_train'
KAGGLE_DATA_TEST = '/kaggle/input/action-video/data/test'

# Local data directories
LOCAL_DATA_TRAIN = './data/train'
LOCAL_DATA_TEST = './data/test'

def is_kaggle_environment():
    """Check if running in Kaggle environment."""
    return os.path.exists('/kaggle')

def setup_kaggle_data(symlink=True):
    """
    Setup data in Kaggle environment.

    Args:
        symlink: If True, create symlinks. If False, copy data.
    """
    print("Detected Kaggle environment")

    # Create local data directory structure
    Path('./data').mkdir(parents=True, exist_ok=True)

    # Check if Kaggle data exists
    if not os.path.exists(KAGGLE_DATA_TRAIN):
        print(f"ERROR: Kaggle training data not found at {KAGGLE_DATA_TRAIN}")
        print("Make sure you have added the 'action-video' dataset to your Kaggle notebook.")
        return False

    if not os.path.exists(KAGGLE_DATA_TEST):
        print(f"WARNING: Kaggle test data not found at {KAGGLE_DATA_TEST}")

    # Setup training data
    if os.path.exists(LOCAL_DATA_TRAIN):
        print(f"Local training directory already exists: {LOCAL_DATA_TRAIN}")
    else:
        if symlink:
            print(f"Creating symlink: {LOCAL_DATA_TRAIN} -> {KAGGLE_DATA_TRAIN}")
            os.symlink(KAGGLE_DATA_TRAIN, LOCAL_DATA_TRAIN)
        else:
            print(f"Copying data from {KAGGLE_DATA_TRAIN} to {LOCAL_DATA_TRAIN}...")
            shutil.copytree(KAGGLE_DATA_TRAIN, LOCAL_DATA_TRAIN)

    # Setup test data
    if os.path.exists(KAGGLE_DATA_TEST):
        if os.path.exists(LOCAL_DATA_TEST):
            print(f"Local test directory already exists: {LOCAL_DATA_TEST}")
        else:
            if symlink:
                print(f"Creating symlink: {LOCAL_DATA_TEST} -> {KAGGLE_DATA_TEST}")
                os.symlink(KAGGLE_DATA_TEST, LOCAL_DATA_TEST)
            else:
                print(f"Copying data from {KAGGLE_DATA_TEST} to {LOCAL_DATA_TEST}...")
                shutil.copytree(KAGGLE_DATA_TEST, LOCAL_DATA_TEST)

    print(f"Data setup complete!")
    print(f"Training data: {os.path.abspath(LOCAL_DATA_TRAIN)}")
    print(f"Test data: {os.path.abspath(LOCAL_DATA_TEST)}")

    # Count samples
    if os.path.exists(LOCAL_DATA_TRAIN):
        train_classes = [d for d in os.listdir(LOCAL_DATA_TRAIN) if os.path.isdir(os.path.join(LOCAL_DATA_TRAIN, d))]
        print(f"Found {len(train_classes)} training classes")

    return True

def download_from_kaggle_api(competition='action-video'):
    """
    Download dataset using Kaggle API (for local development).

    Args:
        competition: Kaggle competition name
    """
    print("Downloading dataset from Kaggle API...")

    try:
        import kaggle

        # Create data directory
        Path('./data').mkdir(parents=True, exist_ok=True)

        # Download competition files
        print(f"Downloading files from competition: {competition}")
        kaggle.api.competition_download_files(competition, path='./data', quiet=False)

        # Extract downloaded zip files
        for file in Path('./data').glob('*.zip'):
            print(f"Extracting {file.name}...")
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('./data')
            print(f"Removing {file.name}...")
            file.unlink()

        print("Download and extraction complete!")
        return True

    except ImportError:
        print("ERROR: Kaggle API not installed. Install with: pip install kaggle")
        print("Also ensure you have set up your Kaggle API credentials.")
        print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

def setup_data(use_kaggle_api=False, symlink=True):
    """
    Main function to setup data based on environment.

    Args:
        use_kaggle_api: Force download using Kaggle API (for local dev)
        symlink: Use symlinks in Kaggle environment (faster than copying)
    """
    if is_kaggle_environment():
        return setup_kaggle_data(symlink=symlink)
    else:
        print("Running in local environment")

        # Check if data already exists
        if os.path.exists(LOCAL_DATA_TRAIN) and os.path.isdir(LOCAL_DATA_TRAIN):
            if len(os.listdir(LOCAL_DATA_TRAIN)) > 0:
                print(f"Data directory '{LOCAL_DATA_TRAIN}' already exists and is not empty.")
                return True

        if use_kaggle_api:
            return download_from_kaggle_api()
        else:
            print("Data not found locally.")
            print("\nOptions:")
            print("1. Run with --use-kaggle-api to download using Kaggle API")
            print("2. Manually download data from https://www.kaggle.com/competitions/action-video")
            print("3. Place training data in: ./data/train")
            print("4. Place test data in: ./data/test")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description='Setup data for action video classification')
    parser.add_argument('--use-kaggle-api', action='store_true',
                        help='Download data using Kaggle API (for local development)')
    parser.add_argument('--no-symlink', action='store_true',
                        help='Copy data instead of creating symlinks (in Kaggle env)')
    parser.add_argument('--competition', type=str, default='action-video',
                        help='Kaggle competition name')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_data(use_kaggle_api=args.use_kaggle_api, symlink=not args.no_symlink)