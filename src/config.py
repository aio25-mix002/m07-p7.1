import torch
import os


class ModelConfig:
    """Singleton config for model hyperparameters. Loads once at first access."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """Load configuration from environment variables."""
        self.image_size: int = int(os.getenv('APPCONFIG__IMAGE_SIZE', 224))
        self.patch_size: int = int(os.getenv('APPCONFIG__PATCH_SIZE', 16))
        self.in_chans: int = int(os.getenv('APPCONFIG__IN_CHANS', 3))
        self.embed_dim: int = int(os.getenv('APPCONFIG__EMBED_DIM', 768))
        self.depth: int = int(os.getenv('APPCONFIG__DEPTH', 12))
        self.num_heads: int = int(os.getenv('APPCONFIG__NUM_HEADS', 12))
        self.mlp_ratio: float = float(os.getenv('APPCONFIG__MLP_RATIO', 4.0))
        self.drop_rate: float = float(os.getenv('APPCONFIG__DROP_RATE', 0.1))
        self.attn_drop_rate: float = float(os.getenv('APPCONFIG__ATTN_DROP_RATE', 0.1))
        self.drop_path_rate: float = float(os.getenv('APPCONFIG__DROP_PATH_RATE', 0.1))
        self.qkv_bias: bool = os.getenv('APPCONFIG__QKV_BIAS', 'True').lower() in ('true', '1', 'yes')
        self.num_classes: int = int(os.getenv('APPCONFIG__NUM_CLASSES', 51))
        self.smif_window: int = int(os.getenv('APPCONFIG__SMIF_WINDOW', 5))
    
    def reload(self):
        """Reload configuration from environment variables. Call explicitly if env vars change."""
        self._load()


class TrainingConfig:
    """Singleton config for training parameters. Loads once at first access."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """Load configuration from environment variables."""
        self.expr_name: str = os.getenv('APPCONFIG__EXPR_NAME', 'default_expr')
        self.data_root: str = os.getenv('APPCONFIG__DATA_ROOT', './hmdb51_data')
        self.weights_dir: str = os.getenv('APPCONFIG__WEIGHTS_DIR', './weights')
        self.pretrained_name: str = os.getenv('APPCONFIG__PRETRAINED_NAME', 'vit_base_patch16_224')
        self.batch_size: int = int(os.getenv('APPCONFIG__BATCH_SIZE', 8))
        self.num_frames: int = int(os.getenv('APPCONFIG__NUM_FRAMES', 16))
        self.frame_stride: int = int(os.getenv('APPCONFIG__FRAME_STRIDE', 2))
        self.lr: float = float(os.getenv('APPCONFIG__LR', 1e-4))
        self.epochs: int = int(os.getenv('APPCONFIG__EPOCHS', 10))
        self.val_ratio: float = float(os.getenv('APPCONFIG__VAL_RATIO', 0.1))
        self.seed: int = int(os.getenv('APPCONFIG__SEED', 42))
        self.num_workers: int = int(os.getenv('APPCONFIG__NUM_WORKERS', 4))
        self.checkpoint_dir: str = os.getenv('APPCONFIG__CHECKPOINT_DIR', './checkpoints')
        #self.log_dir: str = os.getenv('APPCONFIG__LOG_DIR', './logs')
    
    # LOGIC CHỌN DEVICE: Ưu tiên MPS cho Mac -> CUDA -> CPU
    @property
    def device(self) -> str:
        """Get the device (checks dynamically at each access)."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def reload(self):
        """Reload configuration from environment variables. Call explicitly if env vars change."""
        self._load()


class TestConfig:
    """Singleton config for testing parameters. Loads once at first access."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """Load configuration from environment variables."""
        self.data_root: str = os.getenv('APPCONFIG__TEST_DATA_ROOT', './kaggle/competitions/action-video/data/test')
    
    def reload(self):
        """Reload configuration from environment variables. Call explicitly if env vars change."""
        self._load()
