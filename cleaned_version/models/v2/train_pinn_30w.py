"""
train_pinn_30w.py
=================
用 training_data_30W_test 数据训练 PINN 模型。

数据格式：JSON 文件包含
- components: [{x_range_mm, y_range_mm, center_mm, power_W}]
- temperature_data: [{x, y, temperature}] (10000 points, x,y in 0-10 range)

坐标说明：
- x, y 在 0-10 范围，代表 0-100mm 物理位置
- 转换为网格索引：xi = int(round(x * 99 / 10))
- 网格大小：100x100 (索引 0-99)

用法:
  python train_pinn_30w.py --data-dir ../training_data_30W_test --epochs 500 --batch-size 16
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from pinn_v3_physics_fix import (
    ThermalPINNPhysicsFix,
    PINNDataset,
    train_pinn_physics,
)


# ============================================================================
# Data Loading
# ============================================================================

def load_json_sample(json_path):
    """
    Load a single JSON sample and return (params, temps_2d, total_power)

    坐标转换（关键修复）：
    - x, y 在 0-10 范围（代表 0-100mm 物理位置）
    - 正确转换：xi = int(round(x * 99 / 10))
    - 错误转换：xi = int(x * 10 / 100 * 99)  # 这会导致 199 个格子为空
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get component info
    num_comp = data['num_components']
    components = data['components']

    # Build params array [x_center, y_center, power] for each component
    # Use max 9 components for padding
    # 关键修复：用 0 替代 nan，避免 source_encoder 输出 nan
    max_comp = 9
    params = np.zeros((max_comp, 3), dtype=np.float32)
    total_power = 0.0

    for i, comp in enumerate(components):
        cx = comp['center_mm'][0] / 100.0  # 归一化到 0-1 范围
        cy = comp['center_mm'][1] / 100.0  # 归一化到 0-1 范围
        p = comp['power_W']
        params[i] = [cx, cy, p]
        total_power += p

    # Build 100x100 temperature grid
    temp_data = data['temperature_data']
    grid_size = data['simulation_params']['grid_size']
    ambient_temp = data['simulation_params']['ambient_temp_C']
    T = np.zeros((grid_size, grid_size), dtype=np.float32)

    # 正确坐标转换：x 是 0-10 范围，直接映射到 0-99 索引
    for td in temp_data:
        x = td['x']
        y = td['y']
        temp = td['temperature']

        # 关键修复：使用 round() 避免截断误差
        xi = int(round(x * 99 / 10))
        yi = int(round(y * 99 / 10))
        xi = max(0, min(grid_size - 1, xi))
        yi = max(0, min(grid_size - 1, yi))
        T[xi, yi] = temp

    # 验证：确保没有零值格子（如果有，填充环境温度）
    zero_count = np.sum(T == 0)
    if zero_count > 0:
        print(f"  Warning: {zero_count} zero cells in {os.path.basename(json_path)}, filling with ambient={ambient_temp}")
        T[T == 0] = ambient_temp

    return params, T, total_power


def load_dataset(data_dir):
    """Load all JSON files from directory"""
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    print(f"Found {len(json_files)} JSON files")

    all_params = []
    all_temps = []
    all_powers = []

    for i, fname in enumerate(json_files):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(json_files)}...")
        json_path = os.path.join(data_dir, fname)
        params, temps, total_power = load_json_sample(json_path)
        all_params.append(params)
        all_temps.append(temps)
        all_powers.append(total_power)

    return np.array(all_params), np.array(all_temps), np.array(all_powers)


# ============================================================================
# Main Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PINN on 30W data')
    parser.add_argument('--data-dir', type=str,
                        default=r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test',
                        help='Directory containing JSON files')
    parser.add_argument('--out-dir', type=str, default='./results_30W',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--d-hidden', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of MLP layers')
    parser.add_argument('--n-freqs', type=int, default=64,
                        help='Number of Fourier frequencies')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--lambda-pde', type=float, default=0.001,
                        help='PDE loss weight')
    parser.add_argument('--lambda-bc', type=float, default=0.0001,
                        help='BC loss weight')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--max-components', type=int, default=9,
                        help='Max components in data')
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

    # Verify data quality
    zero_cells = np.sum(temps == 0)
    print(f"Data quality: {zero_cells} zero cells in temperature arrays")
    if zero_cells > 0:
        print("  Note: These should have been filled with ambient temperature")

    # Split: 80% train, 10% val, 10% test
    n_samples = len(params)
    indices = np.arange(n_samples)

    # First split: 80% train, 20% temp (val+test)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Second split: 50% of temp = 10%, 50% = 10%
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

    # Create model
    print("\n=== Creating Model ===")
    model = ThermalPINNPhysicsFix(
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        n_freqs=args.n_freqs,
        n_sources=args.max_components,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Save run metadata
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata = {
        "timestamp": timestamp,
        "args": vars(args),
        "model_params": total_params,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    with open(os.path.join(args.out_dir, f"run_config_{timestamp}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.out_dir, "run_config_latest.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Train
    print("\n=== Training ===")
    # 关键修复：正确处理返回值
    # train_pinn_physics 返回 (train_losses, val_losses, info)
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
    model_path = os.path.join(args.out_dir, 'pinn_30W_final.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'args': vars(args),
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
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.legend()
    plt.title("PINN 30W Training / Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"[Saved] {args.out_dir}/loss_curves.png")

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
    plt.savefig(os.path.join(args.out_dir, 'test_predictions.png'), dpi=150)
    plt.close()
    print(f"[Saved] {args.out_dir}/test_predictions.png")

    print("\nDone!")


if __name__ == '__main__':
    main()