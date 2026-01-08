# Make inference on Kaggle test set and create the submission file, submit directly to Kaggle if needed
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.config import ModelConfig

def kaggle_submit(predictions: list[str], submission_file: Path, submit: bool = False):
    import pandas as pd
    import kaggle

    df = pd.DataFrame({'Id': range(len(predictions)), 'Category': predictions})
    df.to_csv(submission_file, index=False)

    if submit:
        kaggle.api.competition_submit(submission_file, "LSViT", "LSViT submission")

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
    parser.add_argument('--submit', action='store_true',
                        help='Submit to Kaggle after inference')
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
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    if 'acc' in checkpoint:
        print(f"Model loaded (trained acc: {checkpoint['acc']:.4f})")
    else:
        print("Model loaded")

    print("\nLoading test dataset...")
    test_dataset = HMDB51Dataset(
        root=args.data_root,
        split='val',
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()