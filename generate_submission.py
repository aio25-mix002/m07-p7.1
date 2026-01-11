import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Import model của bạn
from src.model import LSViTForAction
from src.config import ModelConfig

# === DANH SÁCH 51 CLASS HMDB51 ===
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

class DenseTestDataset(Dataset):
    def __init__(self, root, num_clips=10, num_frames=16, image_size=224):
        self.root = Path(root)
        self.num_clips = num_clips
        self.num_frames = num_frames
        
        if not self.root.exists():
             raise FileNotFoundError(f"Không tìm thấy thư mục: {self.root}")

        self.video_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_folders)

    def _load_clip(self, frame_paths, start_idx):
        frames = []
        total = len(frame_paths)
        for i in range(self.num_frames):
            idx = min(start_idx + i, total - 1) 
            path = frame_paths[idx]
            with Image.open(path) as img:
                img = img.convert("RGB")
                frames.append(self.transform(img))
        
        # Stack lại: [Num_Frames, Channels, H, W]
        # Ví dụ: [16, 3, 224, 224]
        return torch.stack(frames) 

    def __getitem__(self, idx):
        video_dir = self.video_folders[idx]
        video_id = video_dir.name
        
        frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            dummy_size = self.transform.transforms[0].size[0]
            return torch.zeros(self.num_clips, self.num_frames, 3, dummy_size, dummy_size), video_id

        max_start = max(0, total_frames - self.num_frames)
        start_indices = np.linspace(0, max_start, self.num_clips, dtype=int)
        
        clips = []
        for start_idx in start_indices:
            clip = self._load_clip(frame_paths, start_idx) 
            # === SỬA LỖI Ở ĐÂY ===
            # KHÔNG dùng permute(1,0,2,3) nữa. 
            # Giữ nguyên thứ tự [T, C, H, W] (Frames, Channels, H, W)
            # Vì Model của bạn mong đợi [B, T, C, H, W]
            clips.append(clip)
            
        # Output: [Num_Clips, T, C, H, W]
        return torch.stack(clips), video_id

def save_submission(ids, predictions, output_path):
    df = pd.DataFrame({'id': ids, 'class': predictions})
    try:
        df['id_num'] = df['id'].astype(int)
        df = df.sort_values(by='id_num').drop(columns=['id_num'])
    except:
        df = df.sort_values(by='id')
        
    if os.path.exists('/kaggle/working'):
        final_path = os.path.join('/kaggle/working', os.path.basename(output_path))
    else:
        final_path = output_path

    df.to_csv(final_path, index=False)
    print(f"✅ Submission saved to: {final_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_clips', type=int, default=10)
    parser.add_argument('--num_frames', type=int, default=16) 
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Configuring Model: Frames={args.num_frames}, Size={args.image_size}")
    config = ModelConfig()
    config.num_classes = 51
    config.image_size = args.image_size
    if hasattr(config, 'num_frames'):
        config.num_frames = args.num_frames
    
    model = LSViTForAction(config=config).to(device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. Load Data
    print(f"Loading Dataset from {args.data_root}")
    dataset = DenseTestDataset(
        root=args.data_root, 
        num_clips=args.num_clips,
        num_frames=args.num_frames, 
        image_size=args.image_size
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_ids = []

    print("Running Inference...")
    with torch.no_grad():
        for i, (videos, ids) in enumerate(tqdm(loader)):
            # videos shape: [Batch, Num_Clips, T, C, H, W]
            # Ví dụ: [4, 10, 16, 3, 224, 224]
            
            b, n_clips, t, c, h, w = videos.shape
            
            # Gộp Batch và Num_Clips
            # Input thành: [Batch*Clips, T, C, H, W]
            inputs = videos.view(-1, t, c, h, w).to(device)
            
            # Forward
            logits = model(inputs) 
            
            logits = logits.view(b, n_clips, -1)
            mean_logits = logits.mean(dim=1)
            preds = mean_logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(ids)

    final_names = [HMDB51_CLASSES[p] if p < 51 else "unknown" for p in all_preds]
    save_submission(all_ids, final_names, args.output)

if __name__ == "__main__":
    main()