import argparse
import os
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.utils import set_seed, load_vit_checkpoint, ensure_dir
from src.engine import train_one_epoch, evaluate
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

def parse_args():
    parser = argparse.ArgumentParser(description='Train LS-ViT model for action recognition')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--data_root', type=str, default=None, help='Path to training data directory')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of frames to sample from each video')
    parser.add_argument('--frame_stride', type=int, default=None, help='Stride between sampled frames')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--val_ratio', type=float, default=None, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    # Config
    t_cfg = TrainingConfig()
    m_cfg = ModelConfig()

    # Override config with command line arguments
    if args.epochs is not None: t_cfg.epochs = args.epochs
    if args.batch_size is not None: t_cfg.batch_size = args.batch_size
    if args.lr is not None: t_cfg.lr = args.lr
    if args.data_root is not None: t_cfg.data_root = args.data_root
    if args.num_frames is not None: t_cfg.num_frames = args.num_frames
    if args.frame_stride is not None: t_cfg.frame_stride = args.frame_stride
    if args.num_workers is not None: t_cfg.num_workers = args.num_workers
    if args.val_ratio is not None: t_cfg.val_ratio = args.val_ratio
    if args.seed is not None: t_cfg.seed = args.seed

    set_seed(t_cfg.seed)
    
    device = torch.device(t_cfg.device)
    print(f"Using device: {device}")

    # Dataset
    print("Initializing datasets...")
    train_ds = HMDB51Dataset(
        root=t_cfg.data_root, split='train', 
        num_frames=t_cfg.num_frames, frame_stride=t_cfg.frame_stride,
        val_ratio=t_cfg.val_ratio, seed=t_cfg.seed
    )
    val_ds = HMDB51Dataset(
        root=t_cfg.data_root, split='val', 
        num_frames=t_cfg.num_frames, frame_stride=t_cfg.frame_stride,
        val_ratio=t_cfg.val_ratio, seed=t_cfg.seed
    )

    # === CẤU HÌNH DATALOADER TỐI ƯU RAM ===
    loader_kwargs = {
        "batch_size": t_cfg.batch_size,
        "num_workers": t_cfg.num_workers,
        "pin_memory": True,
        
        # Tối ưu hóa: Giữ worker sống trong RAM và prefetch dữ liệu
        "persistent_workers": True if t_cfg.num_workers > 0 else False,
        "prefetch_factor": 4 if t_cfg.num_workers > 0 else None,
    }

    # Train Loader: Cần drop_last=True cho Mixup và shuffle=True
    train_loader = DataLoader(
        train_ds, 
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs 
    )


    val_kwargs = loader_kwargs.copy()
    
    val_loader = DataLoader(
        val_ds, 
        shuffle=False,
        drop_last=False, # Val cần giữ lại hết mẫu
        collate_fn=collate_fn,
        **val_kwargs
    )
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model Setup 
    print("Creating model...")
    model = LSViTForAction(config=m_cfg)
    
    # Load weights
    load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)
    model = model.to(device)
    
    # Multi-GPU & Compile
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Kích hoạt chế độ Multi-GPU trên {torch.cuda.device_count()} card!")
        model = nn.DataParallel(model)
        
    if os.name != 'nt' and torch.cuda.is_available():
        # Lưu ý: compile đôi khi có thể gây lỗi với Mixup trên một số ver torch cũ
        # Nếu gặp lỗi lạ, thử comment dòng này lại
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Không thể compile model: {e}. Chạy chế độ thường.")
    else:
        print("Chạy trên Single GPU/CPU.")

    # Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr)
    
    # Mixup Setup
    mixup_fn = None
    # Kiểm tra an toàn cho cutmix_minmax (nếu config cũ chưa có)
    cutmix_minmax = getattr(t_cfg, 'cutmix_minmax', None)
    
    mixup_active = t_cfg.mixup_alpha > 0 or t_cfg.cutmix_alpha > 0 or cutmix_minmax is not None
    if mixup_active:
        print("Data Augmentation: Enabled Mixup & CutMix!")
        mixup_fn = Mixup(
            mixup_alpha=t_cfg.mixup_alpha, 
            cutmix_alpha=t_cfg.cutmix_alpha, 
            prob=t_cfg.mixup_prob, 
            switch_prob=t_cfg.mixup_switch_prob, 
            mode=t_cfg.mixup_mode,
            label_smoothing=t_cfg.label_smoothing, 
            num_classes=m_cfg.num_classes
        )
        
    # Loss Function
    if mixup_fn is not None:
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.label_smoothing)
        
    val_criterion = nn.CrossEntropyLoss()
    
    # Scaler
    use_amp = (device.type == 'cuda') or (device.type == 'mps')
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    
    # Loop
    best_acc = 0.0
    ensure_dir(args.checkpoint_dir)

    def set_freeze_status(model, freeze_backbone=True):
        real_model = model.module if hasattr(model, 'module') else model
        for param in real_model.backbone.parameters():
            param.requires_grad = not freeze_backbone
        
        if freeze_backbone:
            print("Backbone FROZEN (Chỉ train SMIF & Head)")
        else:
            print("Backbone UN-FROZEN (Train toàn bộ)")
            
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Epochs: {t_cfg.epochs}")
    print(f"  Batch size: {t_cfg.batch_size}")
    print(f"  Workers: {t_cfg.num_workers}")
    print(f"  Persistent Workers: {loader_kwargs.get('persistent_workers')}")
    print(f"  Prefetch Factor: {loader_kwargs.get('prefetch_factor')}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*60}\n")

    for epoch in range(t_cfg.epochs):
        
        if epoch < 3:
            set_freeze_status(model, freeze_backbone=True)
        else:
            set_freeze_status(model, freeze_backbone=False)
            
        print(f"\nEpoch {epoch+1}/{t_cfg.epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, 
            mixup_fn=mixup_fn, criterion=train_criterion
        )
        
        # Đánh giá dùng val_criterion riêng
        val_acc, val_loss = evaluate(model, val_loader, device, criterion=val_criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = f"{args.checkpoint_dir}/best_model.pth"
            torch.save(model_to_save.state_dict(), checkpoint_path)
            print(f"New best model saved! ({best_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()