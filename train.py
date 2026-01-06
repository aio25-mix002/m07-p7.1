import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.utils import set_seed, load_vit_checkpoint, ensure_dir
from src.engine import train_one_epoch, evaluate

def main():
    # Config
    t_cfg = TrainingConfig()
    m_cfg = ModelConfig()
    set_seed(t_cfg.seed)
    
    device = torch.device(t_cfg.device)
    print(f"Using device: {device}")

    # Dataset & Dataloader
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
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model Setup 
    print("Creating model...")
    model = LSViTForAction(config=m_cfg)
    
    # BƯỚC 1: Load weights VÀO RAM trước khi đẩy vào GPU
    load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)

    # BƯỚC 2: Đẩy model vào GPU chính
    model = model.to(device)
    
    # BƯỚC 3: Kích hoạt DataParallel nếu có > 1 GPU
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"🔥 Kích hoạt chế độ Multi-GPU trên {torch.cuda.device_count()} card!")
        model = nn.DataParallel(model)
    else:
        print("Chạy trên Single GPU.")

    # Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr)
    
    # Scaler cho Mixed Precision
    use_amp = (device.type == 'cuda') or (device.type == 'mps')
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    
    # Loop
    best_acc = 0.0
    ensure_dir('./checkpoints')
    
    for epoch in range(t_cfg.epochs):
        print(f"\nEpoch {epoch+1}/{t_cfg.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "./checkpoints/best_model.pth")
            
            print(f"New best model saved! ({best_acc:.4f})")

if __name__ == "__main__":
    main()
