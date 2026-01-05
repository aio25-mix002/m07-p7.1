import os
import torch
import random
import numpy as np
from pathlib import Path
import timm

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
                key = key[len(prefix):]
        filtered_state[key] = v

    missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)
    print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")