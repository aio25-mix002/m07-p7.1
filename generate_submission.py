import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
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

class UltimateTestDataset(Dataset):
    def __init__(self, root, num_clips=10, num_frames=16, image_size=224):
        self.root = Path(root)
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.image_size = image_size
        
        if not self.root.exists():
             raise FileNotFoundError(f"Không tìm thấy thư mục: {self.root}")

        self.video_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        # 1. Resize giữ tỷ lệ (Cạnh ngắn = 256)
        # 2. Chuẩn hóa
        self.resize_transform = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
        self.norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.video_folders)

    def _process_frame(self, img_path):
        """Đọc ảnh, Resize 256, trả về 3 Crop (Left, Center, Right)"""
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # Resize cạnh ngắn lên 256, giữ aspect ratio
            img = self.resize_transform(img)
            
            # Chuyển sang Tensor để crop chính xác
            tensor = self.to_tensor(img) 
            
            # --- THREE CROP LOGIC ---
            # Ảnh đang là [C, H, W] (ví dụ 3, 256, 340) hoặc (3, 340, 256)
            c, h, w = tensor.shape
            th, tw = self.image_size, self.image_size # 224
            
            # 1. Center Crop
            center = F.center_crop(tensor, (th, tw))
            
            # 2. First Crop (Top-Left)
            first = F.crop(tensor, 0, 0, th, tw)
            
            # 3. Last Crop (Bottom-Right)
            last = F.crop(tensor, h - th, w - tw, th, tw)
            
            # Normalize từng crop
            return [self.norm_transform(center), self.norm_transform(first), self.norm_transform(last)]

    def _load_clip_3views(self, frame_paths, start_idx):
        """
        Lấy 1 clip (16 frames).
        Với mỗi frame, sinh ra 3 crops.
        Kết quả trả về: 3 clips riêng biệt (Clip_Center, Clip_Left, Clip_Right)
        """
        frames_center, frames_first, frames_last = [], [], []
        
        total = len(frame_paths)
        for i in range(self.num_frames):
            idx = min(start_idx + i, total - 1)
            path = frame_paths[idx]
            
            # Nhận về 3 crop của frame này
            c, f, l = self._process_frame(path)
            
            frames_center.append(c)
            frames_first.append(f)
            frames_last.append(l)
        
        # Stack lại thành [T, C, H, W]
        clip_c = torch.stack(frames_center)
        clip_f = torch.stack(frames_first)
        clip_l = torch.stack(frames_last)
        
        return [clip_c, clip_f, clip_l]

    def __getitem__(self, idx):
        video_dir = self.video_folders[idx]
        video_id = video_dir.name
        
        frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
        total_frames = len(frame_paths)
        
        # Handle video lỗi/rỗng
        if total_frames == 0:
            dummy = torch.zeros(self.num_clips * 3, self.num_frames, 3, self.image_size, self.image_size)
            return dummy, video_id

        # Chia đều video lấy start indices
        max_start = max(0, total_frames - self.num_frames)
        start_indices = np.linspace(0, max_start, self.num_clips, dtype=int)
        
        all_clips = []
        for start_idx in start_indices:
            # Mỗi lần gọi trả về list 3 clips (3 views)
            three_views = self._load_clip_3views(frame_paths, start_idx)
            all_clips.extend(three_views)
            
        # Output shape: [Num_Clips * 3, T, C, H, W]
        # Ví dụ: 10 clips * 3 views = 30 clips
        return torch.stack(all_clips), video_id

def save_submission(ids, predictions, output_path):
    # Logic cũ giữ nguyên
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
    parser.add_argument('--num_clips', type=int, default=10, help="Số lượng đoạn clip temporal")
    parser.add_argument('--num_frames', type=int, default=16) 
    parser.add_argument('--image_size', type=int, default=224)
    # GIẢM BATCH SIZE VÌ SỐ LƯỢNG CLIP TĂNG GẤP 3
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output', type=str, default='submission_3crop.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
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

    print(f"Loading Dataset (10 Clips x 3 Crops = 30 Views/Video)...")
    dataset = UltimateTestDataset(
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
        for videos, ids in tqdm(loader):
            # videos shape: [Batch, 30, T, C, H, W]
            b, total_views, t, c, h, w = videos.shape
            
            # Flatten: [Batch * 30, T, C, H, W]
            inputs = videos.view(-1, t, c, h, w).to(device)
            
            # Forward
            logits = model(inputs) 
            
            # Reshape lại để tính trung bình
            logits = logits.view(b, total_views, -1)
            
            # Average tất cả 30 views (Robust prediction)
            mean_logits = logits.mean(dim=1)
            
            preds = mean_logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(ids)

    final_names = [HMDB51_CLASSES[p] if p < 51 else "unknown" for p in all_preds]
    save_submission(all_ids, final_names, args.output)

if __name__ == "__main__":
    main()