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
   
    ensure_dir(weights_dir)
    auto_path = Path(weights_dir) / f"{pretrained_name}_timm.pth"

    if auto_path.is_file():
        state = torch.load(auto_path, map_location="cpu")
    else:
        print(f"Downloading {pretrained_name} weights via timm...")
        pretrained_model = timm.create_model(pretrained_name, pretrained=True)
        state = pretrained_model.state_dict()
        torch.save(state, auto_path)

    
    model_state = backbone.state_dict()
    
    filtered_state = {}
    for k, v in state.items():
        if k.startswith("head"):
            continue
        
        key = k
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        
        # Logic ghép weight
        if key in model_state:
            target_shape = model_state[key].shape
            loaded_shape = v.shape
            
            if len(target_shape) == 5 and len(loaded_shape) == 4:
                print(f"Inflating weight '{key}': {loaded_shape} -> {target_shape}")
                
                # 1. Thêm chiều thời gian vào vị trí index 2: [Out, In, 1, H, W]
                v = v.unsqueeze(2)
                
                # 2. Lặp lại weight theo kích thước thời gian (tubelet_size)
                # target_shape[2] chính là tubelet_size (ví dụ = 2)
                repeat_times = target_shape[2]
                v = v.repeat(1, 1, repeat_times, 1, 1)
                
                # 3. Chia tỉ lệ (Scale) để giữ nguyên variance của output
                # Đây là kỹ thuật chuẩn khi inflate weights (I3D paper)
                v = v / repeat_times
                
            filtered_state[key] = v
        else:
            filtered_state[key] = v

    missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)
    print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")