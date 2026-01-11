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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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
    
    # === THÊM THAM SỐ RESUME ===
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    
    return parser.parse_args()

def build_optimizer_params(model, base_lr, weight_decay, layer_decay=0.75):
    """
    Phân chia tham số để áp dụng Layer-wise Learning Rate Decay.
    Hỗ trợ cả DataParallel (Multi-GPU) và torch.compile.
    """
    param_groups = {}
    
    # 1. "Bóc vỏ" model để lấy thông tin cấu trúc
    if hasattr(model, 'module'):
        real_model = model.module
    elif hasattr(model, '_orig_mod'):
        real_model = model._orig_mod
    else:
        real_model = model

    num_layers = len(real_model.backbone.blocks) + 1 
    scales = list(layer_decay ** i for i in reversed(range(num_layers)))
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        clean_name = name
        if clean_name.startswith("module."): clean_name = clean_name[7:]
        if clean_name.startswith("_orig_mod."): clean_name = clean_name[10:]
            
        if clean_name.startswith("backbone.cls_token") or \
           clean_name.startswith("backbone.pos_embed") or \
           clean_name.startswith("backbone.patch_embed"):
            layer_id = 0
        elif clean_name.startswith("backbone.blocks"):
            try:
                layer_id = int(clean_name.split('.')[2]) + 1
            except:
                layer_id = num_layers - 1
        else:
            layer_id = num_layers - 1
            
        if param.ndim == 1 or clean_name.endswith(".bias"):
            group_name = f"no_decay_layer_{layer_id}"
            this_decay = 0.0
        else:
            group_name = f"decay_layer_{layer_id}"
            this_decay = weight_decay
            
        if group_name not in param_groups:
            param_groups[group_name] = {
                "params": [],
                "weight_decay": this_decay,
                "lr": base_lr * scales[layer_id],
            }
        param_groups[group_name]["params"].append(param)
        
    return list(param_groups.values())

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
    t_cfg.val_ratio = 0.0
    if args.seed is not None: t_cfg.seed = args.seed

    set_seed(t_cfg.seed)
    device = torch.device(t_cfg.device)
    print(f"Using device: {device}")

    # Dataset
    print("Initializing FULL datasets...")
    train_ds = HMDB51Dataset(root=t_cfg.data_root, split='train', 
                             num_frames=t_cfg.num_frames, frame_stride=t_cfg.frame_stride,
                             val_ratio=0.0, seed=t_cfg.seed)
    
    # Không cần val_ds thực sự, nhưng tạo để code không lỗi nếu có đoạn nào gọi len()
    loader_kwargs = {
        "batch_size": t_cfg.batch_size,
        "num_workers": t_cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": True if t_cfg.num_workers > 0 else False,
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, collate_fn=collate_fn, **loader_kwargs)
    
    print(f"FULL Train size: {len(train_ds)}")

    # Model Setup (Giữ nguyên)
    print("Creating model...")
    model = LSViTForAction(config=m_cfg)
    
    # Load checkpoint logic (Giữ nguyên logic Resume của bạn)
    if args.resume:
        print(f"--> RESUMING TRAINING from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)
        
    model = model.to(device)
    
   
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Kích hoạt chế độ Multi-GPU trên {torch.cuda.device_count()} card!")
        model = nn.DataParallel(model)
    # ======================================
    
    # Optimizer & Scheduler (Tiếp tục code cũ)
    params = build_optimizer_params(model, base_lr=t_cfg.lr, weight_decay=0.05)
    optimizer = torch.optim.AdamW(params, lr=t_cfg.lr)
    
    if args.resume:
        scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs, eta_min=1e-7)
    else:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs - 5, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5])

    # Loss (Giữ nguyên)
    # Tắt Mixup nếu muốn train sạch ở bước cuối
    mixup_fn = None 
    train_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    ensure_dir(args.checkpoint_dir)
    
    # === VÒNG LẶP TRAIN TOÀN BỘ (FULL DATA LOOP) ===
    print(f"\nStart Training ALL DATA for {t_cfg.epochs} epochs...")
    
    for epoch in range(t_cfg.epochs):
        
        # Luôn Unfreeze khi train full (hoặc tùy logic của bạn)
        real_model = model.module if hasattr(model, 'module') else model
        for param in real_model.backbone.parameters():
            param.requires_grad = True
            
        print(f"\nEpoch {epoch+1}/{t_cfg.epochs}")
        
        # 1. Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, 
            mixup_fn=mixup_fn, criterion=train_criterion
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[-1]
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | LR: {current_lr:.6f}")
        
        # 2. KHÔNG CHẠY EVALUATE (Tiết kiệm thời gian)
        # Vì Val set = Train set, chạy evaluate là vô nghĩa.
        
        # 3. LƯU MODEL (Luôn lưu model cuối cùng)
        model_to_save = model.module if hasattr(model, 'module') else model
        last_path = f"{args.checkpoint_dir}/last_model_full.pth"
        torch.save(model_to_save.state_dict(), last_path)
        print(f"--> Saved: {last_path}")

    print(f"\nTraining complete! Final model: {last_path}")

if __name__ == "__main__":
    main()