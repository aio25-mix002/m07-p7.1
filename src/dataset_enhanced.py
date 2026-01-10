import torch
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

class VideoTransformEnhanced:
    """Enhanced video transform with stronger augmentation."""
    
    def __init__(self, image_size: int, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        if self.is_train:
            # 1. Random resized crop (stronger than before)
            h, w = frames.shape[-2:]
            scale = random.uniform(0.7, 1.0)  # More aggressive scaling
            new_h, new_w = int(h * scale), int(w * scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

            # 2. Horizontal flip
            if random.random() < 0.5:
                frames = TF.hflip(frames)
            
            # 3. Rotation (±15 degrees)
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                frames = TF.rotate(frames, angle, interpolation=InterpolationMode.BILINEAR)
            
            # 4. Color jittering (more aggressive)
            if random.random() < 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                frames = TF.adjust_brightness(frames, brightness_factor)
            
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                frames = TF.adjust_contrast(frames, contrast_factor)
            
            if random.random() < 0.5:
                saturation_factor = random.uniform(0.8, 1.2)
                frames = TF.adjust_saturation(frames, saturation_factor)
            
            if random.random() < 0.3:
                hue_factor = random.uniform(-0.1, 0.1)
                frames = TF.adjust_hue(frames, hue_factor)
            
            # 5. Random grayscale
            if random.random() < 0.1:
                frames = TF.rgb_to_grayscale(frames, num_output_channels=3)
            
            # 6. Gaussian blur (simulate motion blur)
            if random.random() < 0.2:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.1, 2.0)
                frames = TF.gaussian_blur(frames, kernel_size, sigma)
        else:
            # Validation: center crop
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        # Normalize
        normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized)

class HMDB51DatasetEnhanced(Dataset):
    """Enhanced HMDB51 dataset with better augmentation and temporal sampling."""
    
    def __init__(self, root: str, split: str, num_frames: int, frame_stride: int, 
                 image_size: int = 224, val_ratio: float = 0.1, seed: int = 42,
                 temporal_jitter: bool = True):
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
                if not frame_paths: continue
                
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
        self.temporal_jitter = temporal_jitter and (split == "train")
        self.transform = VideoTransformEnhanced(image_size, is_train=(split == "train"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, total: int) -> torch.Tensor:
        """Select frame indices with optional temporal jittering."""
        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)
        
        # Calculate required span
        required_span = (self.num_frames - 1) * self.frame_stride + 1
        
        if self.temporal_jitter and total > required_span:
            # Random start position (temporal jittering)
            max_start = total - required_span
            start_idx = random.randint(0, max_start)
            idxs = torch.arange(start_idx, start_idx + required_span, self.frame_stride)[:self.num_frames]
        else:
            # Uniform sampling across the video
            steps = max(self.num_frames * self.frame_stride, self.num_frames)
            grid = torch.linspace(0, total - 1, steps=steps)
            idxs = grid[:: self.frame_stride].long()
        
        # Pad if necessary
        if idxs.numel() < self.num_frames:
            pad = idxs.new_full((self.num_frames - idxs.numel(),), idxs[-1].item())
            idxs = torch.cat([idxs, pad], dim=0)
        
        return idxs[: self.num_frames]

    @staticmethod
    def _base_video_name(name: str) -> str:
        match = re.match(r"(.+)_\\d+$", name)
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
        self.transform = VideoTransformEnhanced(image_size, is_train=False)
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
