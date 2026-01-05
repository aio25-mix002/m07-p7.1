# m07-p7.1

# LS-ViT Action Recognition on HMDB51

This project implements the LS-ViT (Long Short-term Video Transformer) architecture for action recognition tasks using the HMDB51 dataset. The codebase has been refactored to follow modern MLOps standards, ensuring modularity, scalability, and ease of collaboration.

It supports training on standard NVIDIA GPUs (CUDA) as well as Apple Silicon GPUs (MPS - M1/M2/M3 chips).

## Project Structure

The project is organized into modular components to separate configuration, data processing, modeling, and training logic.

lsvit-hmdb51/
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters and device configuration
│   ├── dataset.py          # HMDB51 dataset loading and transformation pipeline
│   ├── model.py            # LS-ViT architecture (Backbone, SMIF, LMI modules)
│   ├── engine.py           # Training and evaluation loops
│   └── utils.py            # Utility functions (seeding, logging, checkpointing)
├── checkpoints/            # Directory for saving best model artifacts
├── hmdb51_data/            # Directory for dataset storage
├── download_data.py        # Script to automatically download and extract data
├── train.py                # Main entry point for training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Internet connection (for downloading the dataset and pretrained weights)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/aio25-mix002/m07-p7.1.git
   cd m07-p7.1
   ```
2. Create a virtual environment (Recommended):

   # For macOS/Linux


   ```
   python -m venv venv
   source venv/bin/activate
   ```

   # For Windows

   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## Data Preparation

The project includes an automated script to download the HMDB51 dataset from Google Drive and prepare it for training.

Run the following command:

```
python download_data.py
```

This script will:

1. Download the dataset zip file.
2. Extract it into the 'hmdb51_data/' directory.
3. Organize the file structure if necessary.
4. Clean up temporary zip files.

## Usage

### Training

To start the training process, run:

```
python train.py
```

The script will:

1. Initialize the LS-ViT model.
2. Download ImageNet pretrained weights for the backbone (if not present).
3. Train the model for the number of epochs specified in the configuration.
4. Evaluate on the validation set after each epoch.
5. Save the model with the highest validation accuracy to 'checkpoints/best_model.pth'.

### Configuration

You can modify training hyperparameters and model settings in 'src/config.py'.

Key parameters in 'TrainingConfig':

- batch_size: Number of videos per batch.
- num_frames: Number of frames sampled per video.
- epochs: Total training epochs.
- lr: Learning rate.
- num_workers: Number of subprocesses for data loading.

## Apple Silicon (MacBook) Support

This project is optimized for macOS devices with Apple Silicon (M1/M2/M3/M4). The code automatically detects the hardware and uses the Metal Performance Shaders (MPS) backend instead of CPU.

### Important Notes for Mac Users

1. Batch Size:
   MacBooks use Unified Memory (RAM shared between CPU and GPU). Video Transformers are memory-intensive.
   Recommendation: Set 'batch_size' to 2 or 4 in 'src/config.py' to avoid Out-Of-Memory errors.
2. Data Loaders:
   High 'num_workers' on macOS can sometimes cause "Too many open files" errors or system instability.
   Recommendation: Set 'num_workers' to 0 (main thread) or 2 in 'src/config.py'.
3. Mixed Precision:
   The training engine automatically handles autocast for MPS. If you encounter NaN losses, you can disable mixed precision by modifying 'src/engine.py'.

## Outputs

- Checkpoints: The best performing model is saved as './checkpoints/best_model.pth'.
- Weights: Pretrained backbone weights are cached in './weights/'.
- Logs: Training progress (Loss and Accuracy) is printed to the console standard output.
