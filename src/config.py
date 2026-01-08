from dataclasses import dataclass, field
import os
import torch

@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    qkv_bias: bool = True
    num_classes: int = 51
    smif_window: int = 5

def _get_default_data_root() -> str:
    """Get default data root based on environment."""
    if os.path.exists('/kaggle'):
        # In Kaggle environment, use /kaggle/working/data
        return '/kaggle/working/data/train'
    else:
        # Local environment
        return './hmdb51_data'

def _get_default_weights_dir() -> str:
    """Get default weights directory based on environment."""
    if os.path.exists('/kaggle'):
        return '/kaggle/working/weights'
    else:
        return './weights'

@dataclass
class TrainingConfig:
    data_root: str = field(default_factory=_get_default_data_root)
    weights_dir: str = field(default_factory=_get_default_weights_dir)
    pretrained_name: str = 'vit_base_patch16_224'
    batch_size: int = 4  # Trên Mac có thể cần giảm batch size nếu RAM ít
    num_frames: int = 16
    frame_stride: int = 2
    lr: float = 1e-4
    epochs: int = 10
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 2  # Mac thường tối ưu tốt hơn với num_workers thấp hơn (0 hoặc 2)

    # LOGIC CHỌN DEVICE: Ưu tiên MPS cho Mac -> CUDA -> CPU
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'