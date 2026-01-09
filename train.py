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
from datetime import datetime

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
    best_epoch = 0
    epochs_without_improvement = 0

    # Create checkpoint directory with optional experiment name suffix
    checkpoint_dir = t_cfg.checkpoint_dir
    checkpoint_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{t_cfg.expr_name if t_cfg.expr_name else ''}"
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
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Prepare training parameters
            training_params = {
                "train": vars(t_cfg),
                "model": vars(m_cfg),
            }

            # Prepare metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
            }

            # Save checkpoint with metrics
            
            save_checkpoint(
                model,
                checkpoint_dir,
                checkpoint_name,
                metrics,
                training_params,
                val_acc=val_acc,
                train_classes=train_ds.classes,
            )
            logger.info(f"New best model saved! ({best_acc:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= t_cfg.patience:
                logger.info(f"Early stopping triggered! No improvement for {t_cfg.patience} epochs.")
                logger.info(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
                break


if __name__ == "__main__":
    main()
