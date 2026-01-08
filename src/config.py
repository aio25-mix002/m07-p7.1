from dataclasses import dataclass
import torch
import os

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

@dataclass
class TrainingConfig:
    data_root: str = os.getenv('APPCONFIG__DATA_ROOT', './hmdb51_data')
    weights_dir: str = os.getenv('APPCONFIG__WEIGHTS_DIR', './weights')
    pretrained_name: str = os.getenv('APPCONFIG__PRETRAINED_NAME', 'vit_base_patch16_224')
    batch_size: int = int(os.getenv('APPCONFIG__BATCH_SIZE', 8))  # Trên Mac có thể cần giảm batch size nếu RAM ít
    num_frames: int = int(os.getenv('APPCONFIG__NUM_FRAMES', 16))
    frame_stride: int = int(os.getenv('APPCONFIG__FRAME_STRIDE', 2))
    lr: float = float(os.getenv('APPCONFIG__LR', 1e-4))
    epochs: int = int(os.getenv('APPCONFIG__EPOCHS', 10))
    val_ratio: float = float(os.getenv('APPCONFIG__VAL_RATIO', 0.1))
    seed: int = int(os.getenv('APPCONFIG__SEED', 42))
    num_workers: int = int(os.getenv('APPCONFIG__NUM_WORKERS', 4))  # Mac thường tối ưu tốt hơn với num_workers thấp hơn (0 hoặc 2)
    
    # LOGIC CHỌN DEVICE: Ưu tiên MPS cho Mac -> CUDA -> CPU
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
