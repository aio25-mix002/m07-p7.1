from dataclasses import dataclass, field
import os
import torch

@dataclass
class ModelConfig:
    image_size: int = int(os.getenv('APPCONFIG__IMAGE_SIZE', 224))
    patch_size: int = int(os.getenv('APPCONFIG__PATCH_SIZE', 16))
    in_chans: int = int(os.getenv('APPCONFIG__IN_CHANS', 3))
    embed_dim: int = int(os.getenv('APPCONFIG__EMBED_DIM', 768))
    depth: int = int(os.getenv('APPCONFIG__DEPTH', 12))
    num_heads: int = int(os.getenv('APPCONFIG__NUM_HEADS', 12))
    mlp_ratio: float = float(os.getenv('APPCONFIG__MLP_RATIO', 4.0))
    drop_rate: float = float(os.getenv('APPCONFIG__DROP_RATE', 0.1))
    attn_drop_rate: float = float(os.getenv('APPCONFIG__ATTN_DROP_RATE', 0.1))
    drop_path_rate: float = float(os.getenv('APPCONFIG__DROP_PATH_RATE', 0.1))
    qkv_bias: bool = os.getenv('APPCONFIG__QKV_BIAS', 'True').lower() in ('true', '1', 'yes')
    num_classes: int = int(os.getenv('APPCONFIG__NUM_CLASSES', 51))
    smif_window: int = int(os.getenv('APPCONFIG__SMIF_WINDOW', 5))
    tubelet_size: int = int(os.getenv('APPCONFIG__TUBELET_SIZE', 2)) #3D EMBEDDING

def _get_default_data_root() -> str:
    if os.path.exists('/kaggle'):
        return '/kaggle/input/action-video/data/data_train'
    else:
        return './hmdb51_data'

def _get_default_weights_dir() -> str:
    """Get default weights directory based on environment."""
    if os.path.exists('/kaggle'):
        return '/kaggle/working/weights'
    else:
        return './weights'

@dataclass
class TrainingConfig:
    data_root: str = field(default_factory=lambda: os.getenv('APPCONFIG__DATA_ROOT') or _get_default_data_root())
    weights_dir: str = field(default_factory=lambda: os.getenv('APPCONFIG__WEIGHTS_DIR') or _get_default_weights_dir())
    pretrained_name: str = os.getenv('APPCONFIG__PRETRAINED_NAME', 'vit_base_patch16_224.augreg_in21k')
    
    batch_size: int = int(os.getenv('APPCONFIG__BATCH_SIZE', 16))
    num_frames: int = int(os.getenv('APPCONFIG__NUM_FRAMES', 16))
    frame_stride: int = int(os.getenv('APPCONFIG__FRAME_STRIDE', 2))
    lr: float = float(os.getenv('APPCONFIG__LR', 1e-4))
    epochs: int = int(os.getenv('APPCONFIG__EPOCHS', 25)) 
    val_ratio: float = float(os.getenv('APPCONFIG__VAL_RATIO', 0.1))
    seed: int = int(os.getenv('APPCONFIG__SEED', 42))
    num_workers: int = int(os.getenv('APPCONFIG__NUM_WORKERS', 4))
    
    # MIXUP
    mixup_alpha: float = float(os.getenv('APPCONFIG__MIXUP_ALPHA', 0.8))
    cutmix_alpha: float = float(os.getenv('APPCONFIG__CUTMIX_ALPHA', 1.0))
    mixup_prob: float = float(os.getenv('APPCONFIG__MIXUP_PROB', 1.0))
    mixup_switch_prob: float = 0.5
    mixup_mode: str = 'batch'
    
    # === LABEL SMOOTHING ===
    label_smoothing: float = float(os.getenv('APPCONFIG__LABEL_SMOOTHING', 0.1))
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
