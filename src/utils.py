import torch
import random
import numpy as np
import json
from pathlib import Path
import timm
from datetime import datetime


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def ensure_dir(path: str):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_vit_checkpoint(backbone, pretrained_name: str, weights_dir: str):
    """Tự động tải và load weights pretrained từ timm"""
    ensure_dir(weights_dir)
    auto_path = Path(weights_dir) / f"{pretrained_name}_timm.pth"

    if auto_path.is_file():
        state = torch.load(auto_path, map_location="cpu")
    else:
        print(f"Downloading {pretrained_name} weights via timm...")
        pretrained_model = timm.create_model(pretrained_name, pretrained=True)
        state = pretrained_model.state_dict()
        torch.save(state, auto_path)

    # Lọc bỏ phần head để load vào backbone
    filtered_state = {}
    for k, v in state.items():
        if k.startswith("head"):
            continue
        key = k
        # Xử lý prefix nếu cần
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        
        # Handle patch_embed weight conversion from 2D to 3D
        if key == "patch_embed.proj.weight" and v.dim() == 4:
            # Original shape: [out_channels, in_channels, H, W]
            # Target shape: [out_channels, in_channels, T, H, W]
            # Inflate by repeating along temporal dimension and dividing by T
            tubelet_size = 2  # Match config.tubelet_size
            v = v.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / tubelet_size
            print(f"Inflated patch_embed.proj.weight from 2D to 3D: {state[k].shape} -> {v.shape}")
        
        filtered_state[key] = v

    missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)
    print(
        f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )


def save_checkpoint(
    model,
    checkpoint_root_dir: str,
    checkpoint_name: str,
    metrics: dict,
    training_params: dict,
    val_acc: float,
    train_classes: list[str],
) -> Path:
    """Save checkpoint in a timestamped folder with metrics.json

    Args:
        model: The model to save.
        checkpoint_root_dir: Base directory where checkpoints for different runs are stored.
        checkpoint_name: Name of the current run/checkpoint subdirectory.
        metrics: Dictionary containing training metrics (e.g., train_loss, train_acc, val_loss, val_acc, epoch).
        training_params: Dictionary containing training parameters (e.g., lr, batch_size, num_frames, etc.).
        val_acc: Best validation accuracy achieved for this checkpoint.
        train_classes: List of class labels used during training.
    """
    run_dir = Path(checkpoint_root_dir) / checkpoint_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    timestamp = datetime.now().isoformat()
    model_path = run_dir / "best_model.pth"

    # Extract the underlying model if it's compiled or wrapped
    model_to_save = model
    if hasattr(model_to_save, "_orig_mod"):  # torch.compile wrapper
        model_to_save = model_to_save._orig_mod
    if hasattr(model_to_save, "module"):  # DataParallel/DistributedDataParallel wrapper
        model_to_save = model_to_save.module

    torch.save(
        {
            "model": model_to_save.state_dict(),
            "timestamp": timestamp,
            "val_acc": val_acc,
            "train_classes": train_classes,
            "metrics": metrics,
            "training_params": training_params,
        },
        model_path,
    )

    # Combine metrics and training params
    full_metrics = {
        "timestamp": timestamp,
        "training_params": training_params,
        "metrics": metrics,
        "train_classes": train_classes,
    }

    # Save metrics.json file so that it's easy to read without loading the model
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"Checkpoint saved to {run_dir}")
    return run_dir


def find_latest_checkpoint(checkpoint_dir: str, expr_name: str) -> Path | None:
    """Find the latest checkpoint directory, optionally filtered by experiment name.

    Args:
        checkpoint_dir: Base checkpoint directory
        expr_name: Optional experiment name to filter checkpoints

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None

    # Get all subdirectories
    checkpoints = [d for d in checkpoint_path.iterdir() if d.is_dir()]

    # Filter by experiment name if provided
    if expr_name:
        checkpoints = [d for d in checkpoints if expr_name in d.name]

    if not checkpoints:
        print(
            f"No checkpoints found{' for experiment: ' + expr_name if expr_name else ''}"
        )
        return None

    # Sort by timestamp (directory name starts with timestamp YYYYMMDD_HHMMSS)
    checkpoints.sort(key=lambda x: x.name, reverse=True)
    latest = checkpoints[0]

    print(f"Found latest checkpoint: {latest}")
    return latest
