#!/bin/bash

# Quick Start Script for Improved Training
# This script provides easy commands to resume training with recommended settings

echo "=================================================="
echo "  LS-ViT Improved Training - Quick Start"
echo "=================================================="
echo ""

# Check if checkpoint exists
if [ ! -f "./checkpoints/best_model.pth" ]; then
    echo "❌ Error: No checkpoint found at ./checkpoints/best_model.pth"
    echo ""
    echo "Please either:"
    echo "  1. Copy your Colab checkpoint to ./checkpoints/best_model.pth"
    echo "  2. Train from scratch (see option 4 below)"
    echo ""
    exit 1
fi

echo "✅ Found checkpoint: ./checkpoints/best_model.pth"
echo ""
echo "Choose a training strategy:"
echo ""
echo "1. Conservative (Recommended for first try)"
echo "   - Resume from checkpoint"
echo "   - 30 epochs, lower LR, moderate regularization"
echo "   - Expected: Val acc 42-46%, Gap ~5-8%"
echo ""
echo "2. Aggressive (Best for fixing overfitting)"
echo "   - Resume from checkpoint"
echo "   - 40 epochs, differential LR, strong regularization"
echo "   - Expected: Val acc 45-50%, Gap ~3-5%"
echo ""
echo "3. Long Training (Maximum performance)"
echo "   - Resume from checkpoint"
echo "   - 50 epochs, all improvements, larger val set"
echo "   - Expected: Val acc 48-52%, Gap ~2-4%"
echo ""
echo "4. Start Fresh (From scratch with best practices)"
echo "   - No checkpoint, train from pretrained ViT"
echo "   - 50 epochs, all improvements"
echo "   - Expected: Val acc 50-55%, Gap ~2-5%"
echo ""
echo "5. Custom (Enter your own parameters)"
echo ""

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Conservative Training..."
        echo "=================================================="
        python train_improved.py \
            --resume ./checkpoints/best_model.pth \
            --epochs 30 \
            --lr 5e-5 \
            --weight_decay 0.02 \
            --lr_scheduler cosine \
            --freeze_epochs 0 \
            --early_stopping_patience 10 \
            --checkpoint_dir ./checkpoints_improved
        ;;
    
    2)
        echo ""
        echo "🚀 Starting Aggressive Training..."
        echo "=================================================="
        python train_improved.py \
            --resume ./checkpoints/best_model.pth \
            --epochs 40 \
            --lr 1e-4 \
            --backbone_lr 1e-5 \
            --weight_decay 0.05 \
            --lr_scheduler plateau \
            --freeze_epochs 0 \
            --early_stopping_patience 8 \
            --val_ratio 0.15 \
            --checkpoint_dir ./checkpoints_improved
        ;;
    
    3)
        echo ""
        echo "🚀 Starting Long Training..."
        echo "=================================================="
        python train_improved.py \
            --resume ./checkpoints/best_model.pth \
            --epochs 50 \
            --lr 1e-4 \
            --backbone_lr 1e-5 \
            --weight_decay 0.03 \
            --lr_scheduler cosine \
            --warmup_epochs 3 \
            --freeze_epochs 0 \
            --early_stopping_patience 12 \
            --val_ratio 0.15 \
            --save_every 5 \
            --checkpoint_dir ./checkpoints_improved
        ;;
    
    4)
        echo ""
        echo "🚀 Starting Fresh Training..."
        echo "=================================================="
        python train_improved.py \
            --epochs 50 \
            --lr 1e-4 \
            --weight_decay 0.01 \
            --lr_scheduler cosine \
            --warmup_epochs 3 \
            --freeze_epochs 5 \
            --early_stopping_patience 10 \
            --val_ratio 0.15 \
            --checkpoint_dir ./checkpoints_fresh
        ;;
    
    5)
        echo ""
        echo "📝 Custom Training"
        echo "=================================================="
        read -p "Epochs [30]: " epochs
        epochs=${epochs:-30}
        
        read -p "Learning rate [1e-4]: " lr
        lr=${lr:-1e-4}
        
        read -p "Weight decay [0.02]: " wd
        wd=${wd:-0.02}
        
        read -p "Resume from checkpoint? (y/n) [y]: " resume
        resume=${resume:-y}
        
        if [ "$resume" = "y" ]; then
            resume_arg="--resume ./checkpoints/best_model.pth"
        else
            resume_arg=""
        fi
        
        echo ""
        echo "🚀 Starting Custom Training..."
        echo "=================================================="
        python train_improved.py \
            $resume_arg \
            --epochs $epochs \
            --lr $lr \
            --weight_decay $wd \
            --lr_scheduler cosine \
            --checkpoint_dir ./checkpoints_custom
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# After training, visualize results
echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
echo ""
echo "📊 Visualizing results..."
python visualize_training.py --history ./checkpoints_*/training_history.json

echo ""
echo "✅ Done! Check the generated plots and checkpoints."
echo ""
