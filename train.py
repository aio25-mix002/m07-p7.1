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

    loader_kwargs = {
        "batch_size": t_cfg.batch_size,
        "num_workers": t_cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": True if t_cfg.num_workers > 0 else False,
        "prefetch_factor": 4 if t_cfg.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True, collate_fn=collate_fn, **loader_kwargs 
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False, collate_fn=collate_fn, **loader_kwargs
    )
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model Setup 
    print("Creating model...")
    model = LSViTForAction(config=m_cfg)
    
    # === LOGIC LOAD CHECKPOINT ĐỂ RESUME ===
    if args.resume:
        print(f"--> RESUMING TRAINING from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Xử lý key module. hoặc _orig_mod.
        state_dict = checkpoint
        # Nếu checkpoint lưu dict bao gồm optimizer, epoch... thì chỉ lấy phần model
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith('module.'): clean_k = clean_k[7:]
            if clean_k.startswith('_orig_mod.'): clean_k = clean_k[10:]
            new_state_dict[clean_k] = v
                
        # Load weights (strict=False để an toàn nếu có sự khác biệt nhỏ về key tên)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"--> Weights loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        # Load pre-trained ImageNet gốc như bình thường
        load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)
        
    model = model.to(device)
    
    # Compile & Multi-GPU
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
        
    if os.name != 'nt' and torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except:
            pass

    # Optimizer
    params = build_optimizer_params(model, base_lr=t_cfg.lr, weight_decay=0.05, layer_decay=0.75)
    optimizer = torch.optim.AdamW(params, lr=t_cfg.lr)

    # === SCHEDULER LOGIC ===
    if args.resume:
        # Nếu Resume: Bỏ qua Warmup, dùng Cosine Decay ngay lập tức (thích hợp cho Fine-tuning tiếp)
        print("Scheduler: Resume mode (Cosine Annealing only - No Warmup)")
        scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs, eta_min=1e-7)
    else:
        # Nếu Train mới: Có Warmup 5 epoch
        warmup_epochs = 5
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # Mixup
    mixup_fn = None
    cutmix_minmax = getattr(t_cfg, 'cutmix_minmax', None)
    mixup_active = t_cfg.mixup_alpha > 0 or t_cfg.cutmix_alpha > 0 or cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=t_cfg.mixup_alpha, cutmix_alpha=t_cfg.cutmix_alpha, 
            prob=t_cfg.mixup_prob, switch_prob=t_cfg.mixup_switch_prob, 
            mode=t_cfg.mixup_mode, label_smoothing=t_cfg.label_smoothing, 
            num_classes=m_cfg.num_classes
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.label_smoothing) if t_cfg.label_smoothing > 0 else nn.CrossEntropyLoss()
        
    val_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=((device.type == 'cuda') or (device.type == 'mps'))) if ((device.type == 'cuda') or (device.type == 'mps')) else None
    
    ensure_dir(args.checkpoint_dir)
    best_acc = 0.0
    
    if args.resume:
        print("Note: Resuming process. Previous best accuracy is unknown here, but model will save if current val acc is high.")

    print(f"\nStart Training for {t_cfg.epochs} epochs...")
    for epoch in range(t_cfg.epochs):
        
        # === FREEZE LOGIC ===
        if args.resume:
            # Khi resume (thường là fine-tune tiếp), ta luôn UNFREEZE backbone
            real_model = model.module if hasattr(model, 'module') else model
            for param in real_model.backbone.parameters():
                param.requires_grad = True
            if epoch == 0: print("Backbone UN-FROZEN (Resume Mode)")
        else:
            # Logic cũ: Freeze 3 epoch đầu
            real_model = model.module if hasattr(model, 'module') else model
            if epoch < 3:
                for param in real_model.backbone.parameters(): param.requires_grad = False
                print("Backbone FROZEN")
            else:
                for param in real_model.backbone.parameters(): param.requires_grad = True
                print("Backbone UN-FROZEN")
            
        print(f"\nEpoch {epoch+1}/{t_cfg.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, mixup_fn=mixup_fn, criterion=train_criterion)
        
        scheduler.step()
        
        val_acc, val_loss = evaluate(model, val_loader, device, criterion=val_criterion)
        
        current_lr = scheduler.get_last_lr()[-1]
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | LR: {current_lr:.6f}")
        print(f"Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")
        
        # Save model
        if val_acc > best_acc:
            best_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            # Lưu tên khác một chút để phân biệt
            suffix = "_resumed" if args.resume else ""
            checkpoint_path = f"{args.checkpoint_dir}/best_model{suffix}.pth"
            torch.save(model_to_save.state_dict(), checkpoint_path)
            print(f"New best model saved! ({best_acc:.4f}) -> {checkpoint_path}")

if __name__ == "__main__":
    main()