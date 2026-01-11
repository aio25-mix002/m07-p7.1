import torch
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import cv2

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

        if frames.shape == 3:
            normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        elif frames.shape == 4:
            # Chuẩn hóa từng frame trong 1 tensor 4D [T, C, H, W]
            # Chú ý: TF.normalize có thể nhận tensor 4D nếu version torchvision đủ mới, 
            # nhưng an toàn nhất là làm thế này:
            normalized = frames / 255.0 if frames.max() > 1.0 else frames
            
            # Biến đổi mean/std thành dạng [1, 3, 1, 1] để broadcast trừ cho [T, 3, H, W]
            m = torch.tensor(self.mean).view(1, 3, 1, 1)
            s = torch.tensor(self.std).view(1, 3, 1, 1)
            normalized = (normalized - m) / s
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
        self.transform = VideoTransform(image_size, is_train=False)
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

class VideoDataset(Dataset):
    def __init__(self, root,num_segments=4,clip_len=16, image_size=224, is_train=True):
        self.root = Path(root)
        self.num_segments = num_segments
        self.clip_len = clip_len
        self.is_train = is_train
        self.transform = VideoTransform(image_size, is_train)
        self.to_tensor = transforms.ToTensor()
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        self.label_idx = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):
                frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                if frame_paths:
                    self.samples.append((frame_paths, self.class_to_idx[cls]))
                    self.label_idx.append(self.class_to_idx[cls])
                    
    def __len__(self):
        return len(self.samples)

    def _tsn_sample(self, frames):
        total = len(frames)
        seg_size = total / self.num_segments
        clips = []

        for i in range(self.num_segments):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)
            if self.is_train:
                center = np.random.randint(start, max(start + 1, end))
            else:
                center = (start + end) // 2

            idxs = np.linspace(
                max(0, center - self.clip_len // 2),
                min(total - 1, center + self.clip_len // 2),
                self.clip_len
            ).astype(int)

            clips.append([frames[i] for i in idxs])

        return clips

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        clips = self._tsn_sample(frame_paths)

        processed_clips = []
        for clip in clips:
            # Đọc tất cả ảnh vào 1 mảng numpy duy nhất [T, H, W, C]
            video_array = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in clip])
            
            # Chuyển sang Tensor [T, H, W, C]
            video_tensor = torch.from_numpy(video_array).float()
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            # Thực hiện Transform trên khối T-C-H-W
            if self.transform:
                video_tensor = self.transform(video_tensor)
            
            # Cuối cùng mới Permute sang chuẩn mô hình 3D: [C, T, H, W]
            # Giả sử video_tensor lúc này là [T, H, W, C]
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            processed_clips.append(video_tensor)

        video = torch.stack(processed_clips, dim=0)
        return video, label


class TestTNSDataset(Dataset):
    def __init__(self, root,num_segments=4,clip_len=16, image_size=224):
        self.root = Path(root)
        self.num_segments = num_segments
        self.clip_len = clip_len
        self.transform = VideoTransform(image_size, is_train=False)
        self.to_tensor = transforms.ToTensor()
        self.video_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()], key=lambda x: int(x.name))
        self.video_ids = [int(d.name) for d in self.video_dirs]
    
    def __len__(self):
        return len(self.video_dirs)
    
    def _tsn_sample(self, frames):
        total = len(frames)
        seg_size = total / self.num_segments
        clips = []

        for i in range(self.num_segments):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)
            center = (start + end) // 2

            idxs = np.linspace(
                max(0, center - self.clip_len // 2),
                min(total - 1, center + self.clip_len // 2),
                self.clip_len
            ).astype(int)

            clips.append([frames[i] for i in idxs])

        return clips

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        video_id = self.video_ids[idx]
        frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        total = len(frame_paths)

        clips = self._tsn_sample(frame_paths)
        processed_clips = []
        for clip in clips:
            # Đọc tất cả ảnh vào 1 mảng numpy duy nhất [T, H, W, C]
            video_array = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in clip])
            
            # Chuyển sang Tensor [T, H, W, C]
            video_tensor = torch.from_numpy(video_array).float()
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            # Thực hiện Transform trên khối T-C-H-W
            if self.transform:
                video_tensor = self.transform(video_tensor)
            
            # Cuối cùng mới Permute sang chuẩn mô hình 3D: [C, T, H, W]
            # Giả sử video_tensor lúc này là [T, H, W, C]
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            processed_clips.append(video_tensor)

        video = torch.stack(processed_clips, dim=0)
        return video, video_id

def create_balanced_sampler(dataset):
    """Create balanced sampler for imbalanced dataset"""
    if hasattr(dataset, 'dataset'):
        all_labels = [dataset.dataset.label_idx[i] for i in dataset.indices]
    else:
        all_labels = dataset.label_idx

    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in all_labels]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Balanced Sampler: class counts min={class_counts.min()}, max={class_counts.max()}")
    return sampler