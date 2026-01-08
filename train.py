import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig
from src.dataset import HMDB51Dataset, collate_fn
from src.model import LSViTForAction
from src.utils import set_seed, load_vit_checkpoint, ensure_dir, save_checkpoint
from src.engine import train_one_epoch, evaluate
from src.logging_utils import logger


def main():
    # Config
    t_cfg = TrainingConfig()
    m_cfg = ModelConfig()
    set_seed(t_cfg.seed)

    device = torch.device(t_cfg.device)
    logger.info(f"EXPR NAME: {t_cfg.expr_name}")
    logger.info(f"Using device: {device}")

    # Dataset & Dataloader
    logger.info("Initializing datasets...")
    train_ds = HMDB51Dataset(
        root=t_cfg.data_root,
        split="train",
        num_frames=t_cfg.num_frames,
        frame_stride=t_cfg.frame_stride,
        val_ratio=t_cfg.val_ratio,
        seed=t_cfg.seed,
    )
    val_ds = HMDB51Dataset(
        root=t_cfg.data_root,
        split="val",
        num_frames=t_cfg.num_frames,
        frame_stride=t_cfg.frame_stride,
        val_ratio=t_cfg.val_ratio,
        seed=t_cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg.batch_size,
        shuffle=True,
        num_workers=t_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg.batch_size,
        shuffle=False,
        num_workers=t_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model Setup
    logger.info("Creating model...")
    model = LSViTForAction(config=m_cfg)

    # BƯỚC 1: Load weights VÀO RAM trước khi đẩy vào GPU
    load_vit_checkpoint(model.backbone, t_cfg.pretrained_name, t_cfg.weights_dir)

    # BƯỚC 2: Đẩy model vào GPU chính
    model = model.to(device)

    # BƯỚC 3: Kích hoạt DataParallel nếu có > 1 GPU
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        logger.info(
            f"🔥 Kích hoạt chế độ Multi-GPU trên {torch.cuda.device_count()} card!"
        )
        model = nn.DataParallel(model)
    if os.name != "nt" and torch.cuda.is_available():
        logger.info("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)
    else:
        logger.info("Chạy trên Single GPU.")

    # Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr)

    # Scaler cho Mixed Precision
    use_amp = (device.type == "cuda") or (device.type == "mps")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    # Loop
    best_acc = 0.0
    
    # Create checkpoint directory with optional experiment name suffix
    checkpoint_dir = t_cfg.checkpoint_dir
    ensure_dir(checkpoint_dir)

    def set_freeze_status(model, freeze_backbone=True):

        real_model = model.module if hasattr(model, "module") else model

        for param in real_model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        if freeze_backbone:
            logger.info("Backbone FROZEN (Chỉ train SMIF & Head)")
        else:
            logger.info("Backbone UN-FROZEN (Train toàn bộ)")

    for epoch in range(t_cfg.epochs):

        if epoch < 3:
            set_freeze_status(model, freeze_backbone=True)
        else:
            set_freeze_status(model, freeze_backbone=False)

        logger.info(f"\nEpoch {epoch+1}/{t_cfg.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device
        )
        val_acc, val_loss = evaluate(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            
            # Prepare training parameters
            training_params = {
                "lr": t_cfg.lr,
                "batch_size": t_cfg.batch_size,
                "num_frames": t_cfg.num_frames,
                "frame_stride": t_cfg.frame_stride,
                "epochs": t_cfg.epochs,
                "seed": t_cfg.seed,
                "pretrained_name": t_cfg.pretrained_name,
                "val_ratio": t_cfg.val_ratio,
                "image_size": m_cfg.image_size,
                "patch_size": m_cfg.patch_size,
                "embed_dim": m_cfg.embed_dim,
                "depth": m_cfg.depth,
                "num_heads": m_cfg.num_heads,
                "mlp_ratio": m_cfg.mlp_ratio,
                "drop_rate": m_cfg.drop_rate,
                "attn_drop_rate": m_cfg.attn_drop_rate,
                "drop_path_rate": m_cfg.drop_path_rate,
                "smif_window": m_cfg.smif_window,
                "num_classes": m_cfg.num_classes,
            }
            
            # Prepare metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
            }
            
            # Save checkpoint with metrics
            save_checkpoint(model, checkpoint_dir, metrics, training_params, t_cfg.expr_name, val_acc=val_acc)
            logger.info(f"New best model saved! ({best_acc:.4f})")


if __name__ == "__main__":
    main()
