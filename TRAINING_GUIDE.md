# Improved Training Guide

## 🎯 What's New in `train_improved.py`

This enhanced training script addresses the overfitting issues and includes:

### ✨ Key Improvements

1. **Resume Training** - Continue from your existing checkpoint
2. **Learning Rate Scheduling** - Cosine annealing or ReduceLROnPlateau
3. **Differential Learning Rates** - Lower LR for backbone, higher for head/SMIF
4. **Weight Decay** - L2 regularization to prevent overfitting
5. **Early Stopping** - Stop training when validation stops improving
6. **Better Checkpointing** - Saves optimizer and scheduler state
7. **Training History** - Tracks all metrics in JSON format
8. **Warmup** - Gradual learning rate warmup
9. **Enhanced Augmentation** - Optional stronger data augmentation

---

## 🚀 Quick Start

### 1. Resume from Your Existing Model

```bash
# Resume from model weights only (will restart optimizer)
python train_improved.py --resume ./checkpoints/best_model.pth --epochs 30

# Resume from full checkpoint (includes optimizer state) - RECOMMENDED
python train_improved.py --resume_full ./checkpoints/checkpoint_epoch_10.pth --epochs 30
```

### 2. Train from Scratch with Improved Settings

```bash
python train_improved.py \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --lr_scheduler cosine \
  --early_stopping_patience 7 \
  --freeze_epochs 3
```

### 3. Use Enhanced Data Augmentation

To use the stronger augmentation, modify the imports in `train_improved.py`:

```python
# Change line 8 from:
from src.dataset import HMDB51Dataset, collate_fn

# To:
from src.dataset_enhanced import HMDB51DatasetEnhanced as HMDB51Dataset, collate_fn
```

---

## 📋 Command Line Arguments

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Total number of epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate for head/SMIF |
| `--backbone_lr` | lr/10 | Learning rate for backbone |
| `--weight_decay` | 0.01 | Weight decay (L2 regularization) |

### Resume & Checkpoints

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | None | Path to model weights (.pth) |
| `--resume_full` | None | Path to full checkpoint with optimizer state |
| `--checkpoint_dir` | ./checkpoints | Directory to save checkpoints |
| `--save_every` | 5 | Save checkpoint every N epochs |

### Training Strategy

| Argument | Default | Description |
|----------|---------|-------------|
| `--freeze_epochs` | 3 | Epochs to keep backbone frozen |
| `--lr_scheduler` | cosine | Scheduler: cosine, plateau, or none |
| `--warmup_epochs` | 2 | Number of warmup epochs |
| `--early_stopping_patience` | 7 | Stop if no improvement for N epochs |

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | auto | Path to training data |
| `--num_frames` | 16 | Frames per video |
| `--frame_stride` | 2 | Stride between frames |
| `--val_ratio` | 0.1 | Validation split ratio |
| `--num_workers` | 4 | Data loading workers |

---

## 💡 Recommended Training Strategies

### Strategy 1: Continue Your Current Training (Conservative)

```bash
python train_improved.py \
  --resume ./checkpoints/best_model.pth \
  --epochs 30 \
  --lr 5e-5 \
  --weight_decay 0.02 \
  --lr_scheduler cosine \
  --freeze_epochs 0 \
  --early_stopping_patience 10
```

**Why?** Lower LR, higher weight decay, no freezing since you already trained 10 epochs.

### Strategy 2: Aggressive Improvement (Recommended)

```bash
python train_improved.py \
  --resume ./checkpoints/best_model.pth \
  --epochs 40 \
  --lr 1e-4 \
  --backbone_lr 1e-5 \
  --weight_decay 0.05 \
  --lr_scheduler plateau \
  --freeze_epochs 0 \
  --early_stopping_patience 8 \
  --val_ratio 0.15
```

**Why?** Differential LRs, higher weight decay, larger validation set, plateau scheduler.

### Strategy 3: Start Fresh with Best Practices

```bash
python train_improved.py \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --lr_scheduler cosine \
  --warmup_epochs 3 \
  --freeze_epochs 5 \
  --early_stopping_patience 10 \
  --val_ratio 0.15
```

**Why?** Longer training, proper warmup, more validation data.

---

## 📊 Understanding the Output

### During Training

```
🔒 Backbone FROZEN (Training SMIF & Head only)
====================================================================
📅 Epoch 1/30
====================================================================
Train: 100%|██████████| 704/704 [05:23<00:00,  2.18it/s, loss=2.3456, acc=0.4123]
Val: 100%|██████████| 79/79 [00:32<00:00,  2.45it/s]

📊 Results:
   Train Loss: 2.3456 | Train Acc: 0.4123
   Val Loss:   2.5678 | Val Acc:   0.3890
   Best Val Acc: 0.3890
   Current LR: 9.80e-05
💾 Saved checkpoint: ./checkpoints/checkpoint_epoch_1.pth
🏆 New best model saved: ./checkpoints/best_model.pth (acc: 0.3890)
```

### Checkpoint Files

- `best_model.pth` - Best model weights only (for inference)
- `checkpoint_epoch_N.pth` - Full checkpoint with optimizer state (for resuming)
- `training_history.json` - All metrics for plotting

---

## 🔍 Monitoring Training

### Check Training History

```python
import json
import matplotlib.pyplot as plt

with open('./checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Time')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Time')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🎓 Tips for Better Results

### 1. **Increase Validation Set**
```bash
--val_ratio 0.15  # or 0.20
```
More validation data = better generalization estimate

### 2. **Use Stronger Regularization**
```bash
--weight_decay 0.05  # Higher weight decay
```
Helps prevent overfitting on small datasets

### 3. **Longer Training with Early Stopping**
```bash
--epochs 50 --early_stopping_patience 10
```
Let the model train longer but stop if it plateaus

### 4. **Differential Learning Rates**
```bash
--lr 1e-4 --backbone_lr 1e-5
```
Pretrained backbone needs smaller updates

### 5. **Use Enhanced Augmentation**
Edit `train_improved.py` to use `dataset_enhanced.py` for stronger augmentation

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
--batch_size 4  # Reduce batch size
```

### "Training too slow"
```bash
--num_workers 8  # Increase data loading workers
```

### "Validation accuracy not improving"
- Increase `--weight_decay` (try 0.02, 0.05)
- Use `--lr_scheduler plateau`
- Increase `--val_ratio` to 0.15 or 0.20
- Switch to enhanced augmentation

### "Training loss not decreasing"
- Increase `--lr` (try 2e-4)
- Reduce `--weight_decay` (try 0.005)
- Increase `--warmup_epochs` to 3 or 5

---

## 📈 Expected Results

With the improved training, you should see:

- **Smaller train-val gap** (< 10% difference)
- **Higher validation accuracy** (45-55% on HMDB51)
- **Smoother training curves**
- **Better generalization**

Your current results:
- Train: 55.5% | Val: 38.0% (17.5% gap) ❌

Target results:
- Train: 50-55% | Val: 45-50% (5-10% gap) ✅

---

## 🔄 Migration from Old Script

The new script is **fully compatible** with your existing checkpoints:

```bash
# Old training
python train.py --epochs 10
# Creates: ./checkpoints/best_model.pth

# Continue with new script
python train_improved.py --resume ./checkpoints/best_model.pth --epochs 30
```

---

## 📝 Example Training Session

```bash
# Step 1: Resume from your existing model
python train_improved.py \
  --resume ./checkpoints/best_model.pth \
  --epochs 30 \
  --lr 5e-5 \
  --weight_decay 0.02 \
  --lr_scheduler cosine \
  --early_stopping_patience 8 \
  --val_ratio 0.15

# Step 2: Monitor results
# Check ./checkpoints/training_history.json

# Step 3: If still overfitting, increase regularization
python train_improved.py \
  --resume_full ./checkpoints/checkpoint_epoch_30.pth \
  --epochs 50 \
  --weight_decay 0.05 \
  --lr_scheduler plateau
```

---

## 🎯 Next Steps

1. **Start with Strategy 2** (Aggressive Improvement)
2. **Monitor the train-val gap** - should decrease
3. **If still overfitting**: increase weight_decay or use enhanced augmentation
4. **If underfitting**: decrease weight_decay or increase learning rate
5. **Use early stopping** to prevent wasting compute

Good luck! 🚀
