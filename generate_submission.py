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
    """
    Dataset đặc biệt cho Dense Sampling:
    - Input: 1 Video folder
    - Output: Tensor [Num_Clips, 3, T, H, W] (Thay vì [3, T, H, W])
    """
    def __init__(self, root, num_clips=10, num_frames=16, image_size=224):
        self.root = Path(root)
        self.num_clips = num_clips
        self.num_frames = num_frames
        
        # Tìm tất cả folder video (0, 1, 2...)
        if not self.root.exists():
             raise FileNotFoundError(f"Không tìm thấy thư mục: {self.root}")

        self.video_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        # Transform: Chỉ Resize và Normalize (Không crop, không flip để giữ nguyên gốc)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_folders)

    def _load_clip(self, frame_paths, start_idx):
        """Load 1 clip gồm num_frames bắt đầu từ start_idx"""
        frames = []
        total = len(frame_paths)
        
        for i in range(self.num_frames):
            # Lấy frame tiếp theo, nếu hết video thì lấy frame cuối cùng (padding)
            idx = min(start_idx + i, total - 1) 
            path = frame_paths[idx]
            
            with Image.open(path) as img:
                img = img.convert("RGB")
                frames.append(self.transform(img))
                
        return torch.stack(frames) # [num_frames, 3, H, W] -> cần permute sau

    def __getitem__(self, idx):
        video_dir = self.video_folders[idx]
        video_id = video_dir.name
        
        # Lấy danh sách ảnh trong folder
        frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            # Xử lý trường hợp folder rỗng (trả về tensor 0)
            # Lưu ý: Cần trả về đúng kích thước ảnh image_size đã cấu hình
            dummy_size = self.transform.transforms[0].size[0] # Lấy size từ transform
            return torch.zeros(self.num_clips, 3, self.num_frames, dummy_size, dummy_size), video_id

        # === LOGIC DENSE SAMPLING ===
        # Chọn 10 điểm bắt đầu rải đều từ đầu đến (cuối - độ dài clip)
        max_start = max(0, total_frames - self.num_frames)
        start_indices = np.linspace(0, max_start, self.num_clips, dtype=int)
        
        clips = []
        for start_idx in start_indices:
            clip = self._load_clip(frame_paths, start_idx) # [T, 3, H, W]
            clip = clip.permute(1, 0, 2, 3) # Đổi thành [3, T, H, W] cho đúng model
            clips.append(clip)
            
        # Stack lại: [Num_Clips, 3, T, H, W]
        dense_tensor = torch.stack(clips)
        
        return dense_tensor, video_id

def save_submission(ids, predictions, output_path):
    df = pd.DataFrame({'id': ids, 'class': predictions})
    
    # Sort theo ID số
    try:
        df['id_num'] = df['id'].astype(int)
        df = df.sort_values(by='id_num').drop(columns=['id_num'])
    except:
        print("Warning: Không thể convert ID sang số, sort theo string.")
        df = df.sort_values(by='id')
        
    # Check môi trường Kaggle
    if os.path.exists('/kaggle/working'):
        final_path = os.path.join('/kaggle/working', os.path.basename(output_path))
    else:
        final_path = output_path

    df.to_csv(final_path, index=False)
    print(f"✅ Submission saved to: {final_path}")
    print(df.head())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to test folder')
    
    # Các tham số quan trọng đã được thêm vào
    parser.add_argument('--num_clips', type=int, default=10, help='Số lượng clip cắt ra từ 1 video')
    parser.add_argument('--num_frames', type=int, default=16, help='Số frame input của model (16 hoặc 32)')
    parser.add_argument('--image_size', type=int, default=224, help='Kích thước ảnh (224, 256...)')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (nên nhỏ vì nhân với num_clips)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output', type=str, default='submission.csv')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading model config: frames={args.num_frames}, size={args.image_size}...")
    config = ModelConfig()
    config.num_classes = 51
    config.image_size = args.image_size # Cập nhật size
    
    model = LSViTForAction(config=config).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Clean keys
    new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Load weights (strict=False để tránh lỗi nhỏ nếu config lệch nhẹ)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. Dataset & Loader
    print(f"Loading data from {args.data_root}...")
    print(f"Strategy: Dense Sampling ({args.num_clips} clips x {args.num_frames} frames)")
    
    dataset = DenseTestDataset(
        root=args.data_root, 
        num_clips=args.num_clips,
        num_frames=args.num_frames, # Lấy từ args
        image_size=args.image_size  # Lấy từ args
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # 3. Inference Loop
    all_preds = []
    all_ids = []

    print("Running Inference...")
    with torch.no_grad():
        for videos, ids in tqdm(loader):
            # videos shape: [Batch_Size, Num_Clips, 3, T, H, W]
            
            b, n_clips, c, t, h, w = videos.shape
            
            # Gộp Batch và Num_Clips lại
            # Input thành: [Batch_Size * Num_Clips, 3, T, H, W]
            inputs = videos.view(-1, c, t, h, w).to(device)
            
            # Forward
            logits = model(inputs) # Kết quả: [Batch*Clips, 51]
            
            # Tách lại Batch và Clips
            # [Batch_Size, Num_Clips, 51]
            logits = logits.view(b, n_clips, -1)
            
            # === CHÌA KHÓA: LẤY TRUNG BÌNH CỘNG (MEAN) ===
            mean_logits = logits.mean(dim=1) # [Batch_Size, 51]
            
            # Lấy nhãn max
            preds = mean_logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(ids)

    # 4. Map to Names & Save
    final_names = [HMDB51_CLASSES[p] if p < 51 else "unknown" for p in all_preds]
    save_submission(all_ids, final_names, args.output)

if __name__ == "__main__":
    main()