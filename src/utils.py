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
        model: The model to save
        checkpoint_dir: Base checkpoint directory
        metrics: Dictionary containing training metrics (e.g., train_loss, train_acc, val_loss, val_acc, epoch)
        training_params: Dictionary containing training parameters (e.g., lr, batch_size, num_frames, etc.)
    """
    run_dir = Path(checkpoint_root_dir) / checkpoint_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    timestamp = datetime.now().isoformat()
    model_path = run_dir / "best_model.pth"
    model_to_save = model.module if hasattr(model, "module") else model
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
