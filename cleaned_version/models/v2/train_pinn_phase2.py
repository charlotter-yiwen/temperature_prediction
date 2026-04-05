"""
train_pinn_phase2.py
====================
Phase 2: 在 Phase 1 模型基础上，加入 physics loss 继续训练

Phase 1: 仅 data loss (lambda_pde=0, lambda_bc=0)
Phase 2: data loss + physics loss (lambda_pde>0, lambda_bc>0)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from pinn_v3_physics_fix import (
    ThermalPINNPhysicsFix,
    PINNDataset,
    train_pinn_physics,
)
from train_pinn_30w import load_dataset


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Train PINN with physics loss')
    parser.add_argument('--phase1-model', type=str,
                        default='./results_phase1/pinn_30W_final.pth',
                        help='Path to Phase 1 model')
    parser.add_argument('--data-dir', type=str,
                        default=r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test',
                        help='Directory containing JSON files')
    parser.add_argument('--out-dir', type=str, default='./results_phase2',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (lower than Phase 1)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lambda-pde', type=float, default=0.001,
                        help='PDE loss weight')
    parser.add_argument('--lambda-bc', type=float, default=0.0001,
                        help='BC loss weight')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--log-every', type=int, default=50,
                        help='Log every N epochs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n=== Loading Data ===")
    params, temps, powers = load_dataset(args.data_dir)
    print(f"Loaded: params={params.shape}, temps={temps.shape}, powers={powers.shape}")
    print(f"Power range: {powers.min():.2f}W - {powers.max():.2f}W")
    print(f"Temperature range: {temps.min():.2f}C - {temps.max():.2f}C")

    # Split: same as Phase 1
    n_samples = len(params)
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"\nSplit: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    p_train = params[train_idx].astype(np.float32)
    t_train = temps[train_idx].astype(np.float32)
    pw_train = powers[train_idx].astype(np.float32)

    p_val = params[val_idx].astype(np.float32)
    t_val = temps[val_idx].astype(np.float32)
    pw_val = powers[val_idx].astype(np.float32)

    p_test = params[test_idx].astype(np.float32)
    t_test = temps[test_idx].astype(np.float32)
    pw_test = powers[test_idx].astype(np.float32)

    # Create datasets
    train_dataset = PINNDataset(p_train, t_train, pw_train)
    val_dataset = PINNDataset(p_val, t_val, pw_val)
    test_dataset = PINNDataset(p_test, t_test, pw_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Phase 1 model
    print("\n=== Loading Phase 1 Model ===")
    checkpoint = torch.load(args.phase1_model, map_location=device)
    model = ThermalPINNPhysicsFix(
        d_hidden=256,
        n_layers=4,
        n_freqs=64,
        n_sources=9,
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded Phase 1 model from: {args.phase1_model}")
    print(f"Phase 1 Test R²: {checkpoint.get('test_r2', 'N/A')}")

    # Phase 2 training: lower learning rate, with physics loss
    print(f"\n=== Phase 2 Training ===")
    print(f"Physics loss: lambda_pde={args.lambda_pde}, lambda_bc={args.lambda_bc}")
    print(f"Learning rate: {args.lr}")

    train_losses, val_losses, info = train_pinn_physics(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        log_every=args.log_every,
        device=device,
        early_stopping=True,
        patience=args.patience,
        out_dir=args.out_dir
    )

    # Evaluate on test set
    print("\n=== Test Evaluation ===")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb, pb in test_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            pred = model.predict_grid(xb, pb)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate R² for each sample
    r2_scores = []
    for i in range(len(preds)):
        r2 = r2_score(targets[i].flatten(), preds[i].flatten())
        r2_scores.append(r2)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Min R²: {np.min(r2_scores):.4f}")
    print(f"Max R²: {np.max(r2_scores):.4f}")

    # Save model
    model_path = os.path.join(args.out_dir, 'pinn_phase2_final.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'phase1_args': checkpoint.get('args', {}),
        'phase2_args': vars(args),
        'phase1_test_r2': checkpoint.get('test_r2', None),
        'test_r2': mean_r2,
        'test_r2_std': std_r2,
        'r2_scores': r2_scores,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.title("Phase 2: PINN Training with Physics Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss_curves_phase2.png'), dpi=150)
    plt.close()
    print(f"[Saved] {args.out_dir}/loss_curves_phase2.png")

    # Plot some predictions
    print("\n=== Sample Predictions ===")
    n_vis = min(6, len(preds))
    fig, axes = plt.subplots(2, n_vis, figsize=(4*n_vis, 8))

    for i in range(n_vis):
        ax = axes[0, i] if n_vis > 1 else axes[0]
        im = ax.imshow(targets[i], cmap='hot', origin='lower', vmin=targets[i].min(), vmax=targets[i].max())
        ax.set_title(f'True R²={r2_scores[i]:.3f}')
        ax.axis('off')

        ax = axes[1, i] if n_vis > 1 else axes[1]
        im = ax.imshow(preds[i], cmap='hot', origin='lower', vmin=targets[i].min(), vmax=targets[i].max())
        ax.set_title(f'Pred')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'test_predictions_phase2.png'), dpi=150)
    plt.close()
    print(f"[Saved] {args.out_dir}/test_predictions_phase2.png")

    print("\nDone!")


if __name__ == '__main__':
    main()