import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module của bạn
from src.config import ModelConfig
from src.model import LSViTForAction
from src.dataset import TestDataset, test_collate_fn

# === DANH SÁCH 51 CLASS HMDB51 (Đã sort A-Z để khớp với Index của Model) ===
# Lưu ý: Model học theo thứ tự thư mục alpha-b, nên list này phải chuẩn A-Z
HMDB51_CLASSES = [
    'brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 
    'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble', 
    'drink', 'eat', 'fall_floor', 'fencer', 'flic_flac', 
    'golf', 'handstand', 'hit', 'hug', 'jump', 
    'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 
    'pour', 'pullup', 'punch', 'push', 'pushup', 
    'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 
    'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 
    'smoke', 'somersault', 'stand', 'swing_baseball', 'sword', 
    'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Kaggle Submission')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data folder (containing 0, 1, 2...)')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config & Model
    print("Creating model...")
    cfg = ModelConfig()
    cfg.num_classes = 51 # HMDB51 luôn là 51
    model = LSViTForAction(config=cfg).to(device)

    # 2. Load Checkpoint (Xử lý các loại prefix module./_orig_mod.)
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Nếu checkpoint lưu cả optimizer/epoch, chỉ lấy phần 'model' hoặc state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model'] # Trường hợp lưu kiểu dictionary
    else:
        state_dict = checkpoint # Trường hợp lưu trực tiếp state_dict
        
    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith('module.'): clean_k = clean_k[7:]
        if clean_k.startswith('_orig_mod.'): clean_k = clean_k[10:]
        new_state_dict[clean_k] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Setup Test Dataset & Loader
    print(f"Loading Test Data from {args.test_dir}...")
    # Lưu ý: TestDataset của bạn đã return (video_tensor, video_id)
    test_ds = TestDataset(
        root=args.test_dir,
        num_frames=cfg.num_frames,   # Mặc định lấy từ config (thường là 16)
        frame_stride=cfg.frame_stride,
        image_size=cfg.image_size
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn, # Hàm collate trả về video_ids
        pin_memory=True
    )
    print(f"Found {len(test_ds)} test videos.")

    # 4. Inference Loop
    results = [] # List chứa dict {'id': ..., 'class': ...}
    
    print("Running Inference...")
    with torch.no_grad():
        for videos, video_ids in tqdm(test_loader):
            videos = videos.to(device)
            
            # Forward
            logits = model(videos)
            preds = logits.argmax(dim=1).cpu().numpy()
            video_ids = video_ids.numpy()
            
            # Map Index -> Class Name
            for vid_id, pred_idx in zip(video_ids, preds):
                class_name = HMDB51_CLASSES[pred_idx]
                results.append({'id': vid_id, 'class': class_name})

    # 5. Create DataFrame & Save CSV
    df = pd.DataFrame(results)
    
    # Sort theo ID để đẹp đội hình (0, 1, 2...)
    df = df.sort_values(by='id').reset_index(drop=True)
    
    # Lưu file
    df.to_csv(args.output, index=False)
    print(f"\n✅ Submission file created successfully: {args.output}")
    print("Example rows:")
    print(df.head())

if __name__ == "__main__":
    main()