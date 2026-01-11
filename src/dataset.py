import torch
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomErasing
from PIL import Image
import numpy as np
from torchvision.transforms import RandAugment
from torchvision import transforms



class VideoTransform:
    def __init__(self, image_size: int, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406] 
        self.std = [0.229, 0.224, 0.225]
        
        if is_train:
            self.rand_aug = RandAugment(num_ops=2, magnitude=9)
            
            self.random_erasing = RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] tensor (0-1 float)
        
        if self.is_train:
            
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0) 
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize cả khối video
            frames = TF.resize(frames, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)


            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            
            
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)


            pil_frames = [TF.to_pil_image(f) for f in frames]
            
            # Fix seed để augmentation giống hệt nhau trên các frame
            seed = random.randint(0, 1000000)
            aug_frames = []
            for img in pil_frames:
                random.seed(seed)
                torch.manual_seed(seed)
                # RandAugment tự lo Brightness, Contrast, Shear, Translate...
                img_aug = self.rand_aug(img)
                aug_frames.append(TF.to_tensor(img_aug))
            
            frames = torch.stack(aug_frames) # [T, C, H, W]


            if random.random() < 0.5:
                frames = TF.hflip(frames)
            
   
            frames = TF.normalize(frames, self.mean, self.std)

            if random.random() < 0.25:
                # get_params trả về i, j, h, w, v
                i, j, h, w, v = self.random_erasing.get_params(frames[0], scale=(0.02, 0.33), ratio=(0.3, 3.3))
                # Áp dụng vùng xóa đó cho TẤT CẢ frame
                frames = torch.stack([TF.erase(f, i, j, h, w, v, inplace=False) for f in frames])

        else:
            # Validation Strategy chuẩn: Resize cạnh ngắn nhất lên 256 -> Center Crop 224
            # Giúp giữ tỷ lệ ảnh, không bị méo hình
            frames = TF.resize(frames, 256, interpolation=InterpolationMode.BILINEAR)
            frames = TF.center_crop(frames, self.image_size)
            frames = TF.normalize(frames, self.mean, self.std)

        return frames

class HMDB51Dataset(Dataset):
    def __init__(self, root: str, split: str, num_frames: int, frame_stride: int, 
                 image_size: int = 224, val_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.root}")
            
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # --- (Giữ nguyên logic load path và split train/val của bạn) ---
        grouped_samples = {}
        for cls in self.classes:
            cls_dir = self.root / cls
            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):
                frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
                if not frame_paths: continue
                base_name = self._base_video_name(video_dir.name)
                group_key = (cls, base_name)
                grouped_samples.setdefault(group_key, []).append((frame_paths, self.class_to_idx[cls]))

        group_values = list(grouped_samples.values())
        rng = np.random.RandomState(seed)
        group_indices = np.arange(len(group_values))
        rng.shuffle(group_indices)
        
        split_point = int(len(group_indices) * (1 - val_ratio))
        if split == "train":
            selected_groups = group_indices[:split_point]
        else:
            selected_groups = group_indices[split_point:]
            
        self.samples = []
        for idx in selected_groups:
            self.samples.extend(group_values[int(idx)])

        self.split = split
        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        
        # === PHẦN CHỈNH SỬA QUAN TRỌNG NHẤT (AUGMENTATION COOLDOWN) ===
        # Thay vì dùng VideoTransform phức tạp, ta dùng torchvision thuần.
        # Mục tiêu: Đưa dữ liệu về dạng "sạch" nhất (Resize + Normalize) để Fine-tune.
        
        self.to_tensor = transforms.ToTensor()
        
        # Cấu hình Normalize chuẩn ImageNet
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        if split == "train":
            # Giai đoạn Cooldown: Bỏ RandomCrop, Bỏ RandomRotation.
            # Chỉ Resize về đúng kích thước và Normalize.
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)), 
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            # Validation: Giữ nguyên như cũ (Resize + Normalize)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            
        # Lưu ý: Tại sao không dùng RandomHorizontalFlip ở đây? 
        # Vì nếu áp dụng từng frame, video sẽ bị nhấp nháy (frame lật, frame không).
        # Tốt nhất giai đoạn cuối nên bỏ qua để dữ liệu ổn định.

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, total_frames: int) -> torch.Tensor:
        # --- (Giữ nguyên logic Sampling của bạn) ---
        if total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)
        if total_frames < self.num_frames:
            idxs = torch.arange(total_frames)
            pad = idxs.new_full((self.num_frames - total_frames,), idxs[-1].item())
            return torch.cat([idxs, pad], dim=0)
        
        segments = np.linspace(0, total_frames, self.num_frames + 1)
        indices = []
        for i in range(self.num_frames):
            start, end = int(segments[i]), int(segments[i+1])
            if start == end: end = start + 1
            if self.split == 'train':
                idx = np.random.randint(start, end)
            else:
                idx = (start + end) // 2
            idx = min(max(0, idx), total_frames - 1)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _base_video_name(name: str) -> str:
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name

    def __getitem__(self, idx: int):
        frame_paths, label = self.samples[idx]
        total = len(frame_paths)
        idxs = self._select_indices(total)
        
        frames = []
        for i in idxs:
            path = frame_paths[int(i.item())]
            with Image.open(path) as img:
                img = img.convert("RGB")
                
                # 1. Chuyển sang Tensor [C, H, W]
                tensor_img = self.to_tensor(img)
                
                # 2. Áp dụng Transform ngay trên từng frame
                # (Để tránh lỗi dimension khi stack thành video)
                transformed_img = self.transform(tensor_img)
                
                frames.append(transformed_img)
        
        # 3. Stack lại thành video [T, C, H, W]
        video = torch.stack(frames) 
        
        # Lưu ý: Không gọi self.transform(video) ở đây nữa vì đã làm trong vòng lặp rồi
        
        return video, label


class TestDataset(Dataset):
    """Dataset for test data without labels (Kaggle competition format)."""

    def __init__(self, root: str, num_frames: int, frame_stride: int, image_size: int = 224):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Test data root not found: {self.root}")

        # Collect all video directories (numbered folders: 0, 1, 2, ...)
        self.video_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name
        )

        # Collect frame paths for each video
        self.samples = []
        for video_dir in self.video_dirs:
            frame_paths = sorted([
                p for p in video_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
            ])
            if frame_paths:
                self.samples.append((int(video_dir.name), frame_paths))

        if not self.samples:
            raise ValueError(f"No valid video samples found in {self.root}")

        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        self.transform = VideoTransform(image_size, is_train=False)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, total_frames: int) -> torch.Tensor:
        # Sử dụng logic giống Validation (Center Sampling) cho Test
        if total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)

        if total_frames < self.num_frames:
            idxs = torch.arange(total_frames)
            pad = idxs.new_full((self.num_frames - total_frames,), idxs[-1].item())
            return torch.cat([idxs, pad], dim=0)

        segments = np.linspace(0, total_frames, self.num_frames + 1)
        indices = []
        for i in range(self.num_frames):
            start, end = int(segments[i]), int(segments[i+1])
            idx = (start + end) // 2 # Luôn lấy giữa
            idx = min(max(0, idx), total_frames - 1)
            indices.append(idx)
            
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_id, frame_paths = self.samples[idx]
        total = len(frame_paths)
        idxs = self._select_indices(total)

        frames = []
        for i in idxs:
            path = frame_paths[int(i.item())]
            with Image.open(path) as img:
                img = img.convert("RGB")
                frames.append(self.to_tensor(img))

        video = torch.stack(frames)
        video = self.transform(video)
        return video, video_id

def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, labels

def test_collate_fn(batch):
    """Collate function for test dataset (returns video IDs instead of labels)."""
    videos = torch.stack([item[0] for item in batch])
    video_ids = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, video_ids