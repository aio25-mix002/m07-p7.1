# train_improved.py
import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.utils import set_seed, load_vit_checkpoint, ensure_dir
from src.engine import train_one_epoch, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Train LS-ViT model for action recognition (Improved)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for head/SMIF')
    parser.add_argument('--backbone_lr', type=float, default=None,
                        help='Learning rate for backbone (default: lr/10)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    
    # Data parameters
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
    
    # Checkpoint & resume
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., ./checkpoints/best_model.pth)')
    parser.add_argument('--resume_full', type=str, default=None,
                        help='Path to full checkpoint with optimizer state (e.g., ./checkpoints/checkpoint_epoch_10.pth)')
    
    # Training strategy
    parser.add_argument('--freeze_epochs', type=int, default=3,
                        help='Number of epochs to keep backbone frozen')
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """Load checkpoint and return starting epoch and best accuracy."""
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Full checkpoint with training state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Loaded optimizer state")
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Loaded scheduler state")
            
            print(f"✅ Resuming from epoch {start_epoch}, best acc: {best_acc:.4f}")
            return start_epoch, best_acc
        else:
            # Just model state dict
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights only")
            return 0, 0.0
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights only")
        return 0, 0.0

def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, checkpoint_dir, is_best=False):
    """Save full checkpoint with all training state."""
    ensure_dir(checkpoint_dir)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_acc': best_acc,
    }
    
    # Save periodic checkpoint
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = f"{checkpoint_dir}/best_model.pth"
        torch.save(model_to_save.state_dict(), best_path)
        print(f"🏆 New best model saved: {best_path} (acc: {best_acc:.4f})")

def set_freeze_status(model, freeze_backbone=True):
    """Freeze or unfreeze backbone parameters."""
    real_model = model.module if hasattr(model, 'module') else model
    
    for param in real_model.backbone.parameters():
        param.requires_grad = not freeze_backbone
    
    if freeze_backbone:
        print("🔒 Backbone FROZEN (Training SMIF & Head only)")
    else:
        print("🔓 Backbone UNFROZEN (Training full model)")

def get_optimizer(model, args, t_cfg):
    """Create optimizer with differential learning rates."""
    real_model = model.module if hasattr(model, 'module') else model
    
    lr = args.lr if args.lr is not None else t_cfg.lr
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else lr / 10
    
    # Separate parameters
    backbone_params = list(real_model.backbone.parameters())
    other_params = [p for n, p in real_model.named_parameters() if 'backbone' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': other_params, 'lr': lr}
    ], weight_decay=args.weight_decay)
    
    print(f"📊 Optimizer config:")
    print(f"   Head/SMIF LR: {lr}")
    print(f"   Backbone LR: {backbone_lr}")
    print(f"   Weight decay: {args.weight_decay}")
    
    return optimizer

def get_scheduler(optimizer, args, t_cfg, steps_per_epoch):
    """Create learning rate scheduler."""
    if args.lr_scheduler == 'none':
        return None
    
    epochs = args.epochs if args.epochs is not None else t_cfg.epochs
    
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - args.warmup_epochs,
            eta_min=1e-6
        )
        print(f"📈 Using CosineAnnealingLR scheduler (T_max={epochs - args.warmup_epochs})")
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        print(f"📈 Using ReduceLROnPlateau scheduler (patience=3)")
    
    return scheduler

def warmup_lr(optimizer, epoch, warmup_epochs, base_lrs):
    """Apply linear warmup to learning rate."""
    if epoch >= warmup_epochs:
        return
    
    warmup_factor = (epoch + 1) / warmup_epochs
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group['lr'] = base_lr * warmup_factor

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
    
    device = torch.device(t_cfg.device)
    print(f"🖥️  Using device: {device}")

    # Dataset & Dataloader
    print("\n📁 Initializing datasets...")
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
    
    print(f"✅ Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model Setup 
    print("\n🤖 Creating model...")
    model = LSViTForAction(config=m_cfg)
    
    # Load pretrained weights if not resuming
    if args.resume is None and args.resume_full is None:
        print("📥 Loading pretrained ViT weights...")
        load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)

    # Move to device
    model = model.to(device)
    
    # DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Compile model (Linux/CUDA only)
    if os.name != 'nt' and torch.cuda.is_available():
        print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer with differential learning rates
    optimizer = get_optimizer(model, args, t_cfg)
    
    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, args, t_cfg, len(train_loader))
    
    # Store base learning rates for warmup
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume_full is not None:
        start_epoch, best_acc = load_checkpoint(
            args.resume_full, model, optimizer, scheduler, device
        )
    elif args.resume is not None:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, None, None, device
        )
    
    # Scaler for Mixed Precision
    use_amp = (device.type == 'cuda') or (device.type == 'mps')
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping
    patience_counter = 0
    
    ensure_dir(args.checkpoint_dir)
    
    print(f"\n{'='*70}")
    print(f"🎯 Training Configuration:")
    print(f"{'='*70}")
    print(f"  Epochs: {t_cfg.epochs} (starting from {start_epoch})")
    print(f"  Batch size: {t_cfg.batch_size}")
    print(f"  Learning rate: {base_lrs[1]:.2e} (head/SMIF), {base_lrs[0]:.2e} (backbone)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Num frames: {t_cfg.num_frames}")
    print(f"  Frame stride: {t_cfg.frame_stride}")
    print(f"  Val ratio: {t_cfg.val_ratio}")
    print(f"  Freeze epochs: {args.freeze_epochs}")
    print(f"  LR scheduler: {args.lr_scheduler}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Resume from: {args.resume_full or args.resume or 'None'}")
    print(f"{'='*70}\n")

    # Training loop
    for epoch in range(start_epoch, t_cfg.epochs):
        
        # Freeze/unfreeze backbone
        if epoch < args.freeze_epochs:
            set_freeze_status(model, freeze_backbone=True)
        else:
            set_freeze_status(model, freeze_backbone=False)
        
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr(optimizer, epoch, args.warmup_epochs, base_lrs)
            current_lr = optimizer.param_groups[1]['lr']
            print(f"🔥 Warmup epoch {epoch+1}/{args.warmup_epochs} - LR: {current_lr:.2e}")
        
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch+1}/{t_cfg.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        
        # Validate
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[1]['lr'])
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Best Val Acc: {best_acc:.4f}")
        
        # Learning rate scheduler step
        if scheduler is not None and epoch >= args.warmup_epochs:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            print(f"   Current LR: {optimizer.param_groups[1]['lr']:.2e}")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, 
                          args.checkpoint_dir, is_best=is_best)
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
            break
        
        if patience_counter > 0:
            print(f"   ⏳ Patience: {patience_counter}/{args.early_stopping_patience}")

    # Save final checkpoint
    save_checkpoint(epoch, model, optimizer, scheduler, best_acc, 
                   args.checkpoint_dir, is_best=False)
    
    # Save training history
    history_path = f"{args.checkpoint_dir}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📈 Training history saved to: {history_path}")

    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
