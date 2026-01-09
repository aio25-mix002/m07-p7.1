# train.py
import argparse
import torch
from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.utils import set_seed, load_vit_checkpoint, ensure_dir
from src.engine import train_one_epoch, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Train LS-ViT model for action recognition')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to training data directory')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to sample from each video')
    parser.add_argument('--frame_stride', type=int, default=None,
                        help='Stride between sampled frames')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--val_ratio', type=float, default=None,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    # Config
    t_cfg = TrainingConfig()
    m_cfg = ModelConfig()

    # Override config with command line arguments
    if args.epochs is not None:
        t_cfg.epochs = args.epochs
    if args.batch_size is not None:
        t_cfg.batch_size = args.batch_size
    if args.lr is not None:
        t_cfg.lr = args.lr
    if args.data_root is not None:
        t_cfg.data_root = args.data_root
    if args.num_frames is not None:
        t_cfg.num_frames = args.num_frames
    if args.frame_stride is not None:
        t_cfg.frame_stride = args.frame_stride
    if args.num_workers is not None:
        t_cfg.num_workers = args.num_workers
    if args.val_ratio is not None:
        t_cfg.val_ratio = args.val_ratio
    if args.seed is not None:
        t_cfg.seed = args.seed

    set_seed(t_cfg.seed)
    
    # Lấy device từ property đã sửa
    device = torch.device(t_cfg.device)
    print(f"Using device: {device}")
    
    if device.type == 'mps':
        print("⚡ Running on Apple Silicon GPU (Metal Performance Shaders)")

    # Dataset & Dataloader
    # Lưu ý: Trên Mac, num_workers quá cao có thể gây lỗi 'Too many open files'
    # Nếu gặp lỗi, hãy set num_workers=0 trong TrainingConfig
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

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg.batch_size, shuffle=True,
        num_workers=t_cfg.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg.batch_size, shuffle=False,
        num_workers=t_cfg.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Model Setup
    model = LSViTForAction(config=m_cfg).to(device)
    load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)

    # Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr)
    
    # Scaler chỉ cần thiết nếu dùng AMP. 
    # PyTorch tự handle device trong GradScaler (nhưng tốt nhất enable=True/False)
    use_amp = (device.type == 'cuda') or (device.type == 'mps')
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    
    # Loop
    best_acc = 0.0
    ensure_dir(args.checkpoint_dir)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Epochs: {t_cfg.epochs}")
    print(f"  Batch size: {t_cfg.batch_size}")
    print(f"  Learning rate: {t_cfg.lr}")
    print(f"  Num frames: {t_cfg.num_frames}")
    print(f"  Frame stride: {t_cfg.frame_stride}")
    print(f"  Val ratio: {t_cfg.val_ratio}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*60}\n")
    
    for epoch in range(t_cfg.epochs):
        print(f"\nEpoch {epoch+1}/{t_cfg.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = f"{args.checkpoint_dir}/best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved! ({best_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()