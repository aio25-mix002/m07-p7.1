import torch
import random
import re
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

class VideoTransform:
    def __init__(self, image_size: int, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        if self.is_train:
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

            if random.random() < 0.5:
                frames = TF.hflip(frames)
            
            # Color jitter nhẹ
            if random.random() < 0.3:
                frames = TF.adjust_brightness(frames, random.uniform(0.9, 1.1))
            if random.random() < 0.3:
                frames = TF.adjust_contrast(frames, random.uniform(0.9, 1.1))
        else:
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized)

class HMDB51Dataset(Dataset):
    def __init__(self, root: str, split: str, num_frames: int, frame_stride: int, 
                 image_size: int = 224, val_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.root}")
            
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Group frames by video
        grouped_samples = {}
        for cls in self.classes:
            cls_dir = self.root / cls
            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):
                frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
                if not frame_paths: 
                    continue
                
                # Extract base name to handle potential splits/clips
                base_name = self._base_video_name(video_dir.name)
                group_key = (cls, base_name)
                grouped_samples.setdefault(group_key, []).append((frame_paths, self.class_to_idx[cls]))

        # Train/Val split
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

        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        self.transform = VideoTransform(image_size, is_train=(split == "train"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, total: int) -> torch.Tensor:
        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)
        steps = max(self.num_frames * self.frame_stride, self.num_frames)
        grid = torch.linspace(0, total - 1, steps=steps)
        idxs = grid[:: self.frame_stride].long()
        if idxs.numel() < self.num_frames:
            pad = idxs.new_full((self.num_frames - idxs.numel(),), idxs[-1].item())
            idxs = torch.cat([idxs, pad], dim=0)
        return idxs[: self.num_frames]

    @staticmethod
    def _base_video_name(name: str) -> str:
        # Loại bỏ phần suffix nếu có dạng _1, _2, ...
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_paths, label = self.samples[idx]
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
        return video, label

def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, labels

class TestDataset(Dataset):
    def __init__(self, root, num_frames=16, frame_stride=2, image_size=224):
        self.root = Path(root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = VideoTransform(image_size, is_train=False)
        self.to_tensor = transforms.ToTensor()
        self.video_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()], key=lambda x: int(x.name))
        self.video_ids = [int(d.name) for d in self.video_dirs]
    
    def __len__(self):
        return len(self.video_dirs)
    
    def _select_indices(self, total):
        if total <= 0:
            raise ValueError("No frames")
        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)
        steps = max(self.num_frames * self.frame_stride, self.num_frames)
        grid = torch.linspace(0, total - 1, steps=steps)
        idxs = grid[::self.frame_stride].long()
        if idxs.numel() < self.num_frames:
            pad = idxs.new_full((self.num_frames - idxs.numel(),), idxs[-1].item())
            idxs = torch.cat([idxs, pad], dim=0)
        return idxs[:self.num_frames]
    
    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        video_id = self.video_ids[idx]
        frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
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