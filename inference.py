# Make inference on Kaggle test set and create the submission file
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.dataset import TestDataset, test_collate_fn
from src.model import LSViTForAction
from src.config import ModelConfig

def save_submission(predictions: list[str], submission_file: Path) -> Path:
    """Save predictions to CSV file and return the file path."""
    import pandas as pd
    import os

    df = pd.DataFrame({'id': range(len(predictions)), 'class': predictions})

    # Check if running on Kaggle platform
    is_kaggle_env = os.path.exists('/kaggle/working')

    if is_kaggle_env:
        # When running on Kaggle, save to /kaggle/working/ for auto-submission
        output_path = Path('/kaggle/working/submission.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✓ Submission file created at: {output_path}")
        return output_path
    else:
        # When running locally, save to specified path
        df.to_csv(submission_file, index=False)
        print(f"\n✓ Submission file created at: {submission_file}")
        return submission_file

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on test set')
    parser.add_argument('--checkpoint', type=str, default='./lightweight_vit_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data/test',
                        help='Path to test data directory')
    parser.add_argument('--pretrained_name', type=str, default='vit_base_patch16_224',
                        help='Pretrained model name')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample')
    parser.add_argument('--frame_stride', type=int, default=2,
                        help='Stride between frames')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for model input')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio (for consistency with training dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--submission_file', type=str, default='./submission.csv',
                        help='Path to save submission file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print("INFERENCE ON TEST SET")

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model config
    model_config = ModelConfig()
    if 'classes' in checkpoint:
        model_config.num_classes = len(checkpoint['classes'])
    else:
        model_config.num_classes = 51  # Default HMDB51 classes

    # Set image size from args
    model_config.image_size = args.image_size

    # Initialize model
    model = LSViTForAction(config=model_config).to(device)

    # Load weights
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    if 'acc' in checkpoint:
        print(f"Model loaded (trained acc: {checkpoint['acc']:.4f})")
    else:
        print("Model loaded")

    print("\nLoading test dataset...")
    test_dataset = TestDataset(
        root=args.data_root,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        image_size=args.image_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn
    )
    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    predictions = []
    video_ids = []

    with torch.no_grad():
        for batch_idx, (videos, ids) in enumerate(test_loader):
            videos = videos.to(device)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            video_ids.extend(ids.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples")

    print(f"\nInference complete! Processed {len(predictions)} videos")

    # Map predictions to class names
    if 'classes' in checkpoint:
        class_names = checkpoint['classes']
        predicted_classes = [class_names[pred] for pred in predictions]
    else:
        predicted_classes = [str(pred) for pred in predictions]

    # Create submission
    submission_path = Path(args.submission_file)
    saved_path = save_submission(predicted_classes, submission_path)

    return saved_path


if __name__ == "__main__":
    submission_path = main()
    print(f"\nSubmission saved to: {submission_path}")