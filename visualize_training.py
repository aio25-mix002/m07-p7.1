#!/usr/bin/env python3
"""
Visualize training history from training_history.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_path, save_path=None):
    """Plot training curves from history JSON."""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 1. Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add best val acc annotation
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
    ax.text(best_epoch, best_val_acc, f' {best_val_acc:.4f}', 
            verticalalignment='bottom', fontsize=10, color='green', fontweight='bold')
    
    # 2. Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Train-Val Gap (Overfitting indicator)
    ax = axes[1, 0]
    acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    ax.plot(epochs, acc_gap, 'purple', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.fill_between(epochs, 0, acc_gap, alpha=0.3, color='purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Acc - Val Acc', fontsize=12)
    ax.set_title('Overfitting Gap (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final gap
    final_gap = acc_gap[-1]
    gap_color = 'red' if final_gap > 0.15 else 'orange' if final_gap > 0.10 else 'green'
    ax.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3))
    
    # 4. Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'orange', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"\nFinal Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"\nOverfitting Gap: {final_gap:.4f}")
    
    if final_gap > 0.15:
        print("⚠️  High overfitting detected! Consider:")
        print("   - Increasing weight_decay")
        print("   - Using stronger data augmentation")
        print("   - Increasing validation set size")
    elif final_gap > 0.10:
        print("⚠️  Moderate overfitting. Consider:")
        print("   - Slightly increasing weight_decay")
        print("   - Using learning rate scheduler")
    else:
        print("✅ Good generalization!")
    
    print(f"\nFinal Learning Rate: {history['lr'][-1]:.2e}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--history', type=str, default='./checkpoints/training_history.json',
                        help='Path to training_history.json')
    parser.add_argument('--output', type=str, default='./training_curves.png',
                        help='Output path for plot image')
    parser.add_argument('--show', action='store_true',
                        help='Show plot instead of saving')
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"❌ Error: History file not found: {history_path}")
        return
    
    save_path = None if args.show else args.output
    plot_training_history(history_path, save_path)

if __name__ == '__main__':
    main()
