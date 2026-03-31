"""
train_pinn_physics_fix.py
=========================
修正物理方程的 PINN 训练脚本。

物理方程修正（基于 thermal_prediction.py 的 SOR 离散格式）:

【边界条件】
SOR 迭代中，每个边界节点的离散方程为：
  count = k_edge/dx² + h/dx
  (k_edge/dx²)·T_adj + (h/dx)·T_amb = count · T_edge

残差形式（应为 0）：
  L_edge = (k/dx² + h/dx)·T_edge - k/dx²·T_adj - h/dx·T_amb

其中（归一化坐标 dx_norm = 1/99）：
  k_edge  = 0.35  (FR4, W/m·K)
  h       = 30.0  (对流系数, W/m²·K)
  dx      = 1/99 ≈ 0.0101

系数：
  k/dx² = 0.35 / (1/99)² ≈ 3433.5
  h/dx  = 30.0 / (1/99)  ≈ 2970
  count ≈ 6403.5

【PDE】
稳态：∇·(k∇T) = -Q
在非热源 interior 点：∇²T = 0

用法:
  # Phase 1: data-only
  python train_pinn_physics_fix.py \
      --count-sweep-params ../training_data/params_count_sweep.npy \
      --count-sweep-temps ../training_data/temps_count_sweep.npy \
      --n-components 5 \
      --physics-norm \
      --t-ambient 25.0 \
      --d-hidden 256 \
      --n-layers 4 \
      --n-freqs 64 \
      --dropout 0.0 \
      --lambda-pde 0.0 \
      --lambda-bc 0.0 \
      --epochs 500 \
      --batch-size 16 \
      --lr 1e-4 \
      --val-ratio 0.1 \
      --early-stopping \
      --patience 100 \
      --out-dir ./results_pinn_physics_fix \
      --model-out pinn_physics_fix_phase1.pth

  # Phase 2: 加弱 physics
  python train_pinn_physics_fix.py \
      --count-sweep-params ../training_data/params_count_sweep.npy \
      --count-sweep-temps ../training_data/temps_count_sweep.npy \
      --n-components 5 \
      --physics-norm \
      --t-ambient 25.0 \
      --d-hidden 256 \
      --n-layers 4 \
      --n-freqs 64 \
      --dropout 0.0 \
      --lambda-pde 0.001 \
      --lambda-bc 0.0001 \
      --epochs 1500 \
      --batch-size 16 \
      --lr 5e-5 \
      --val-ratio 0.1 \
      --early-stopping \
      --patience 200 \
      --out-dir ./results_pinn_physics_fix \
      --model-out pinn_physics_fix_phase2.pth
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from models.pinn_v3_physics_fix import (
    ThermalPINNPhysicsFix,
    PINNDataset,
    train_pinn_physics,
    predict_all_pinn,
    compute_r2,
    plot_loss_curves,
    plot_r2_bar,
    plot_thermal_comparisons,
)


def load_count_sweep_data_pinn(params_path, temps_path, max_components=5,
                                test_ratio=0.2, val_ratio=0.1, split_seed=42,
                                board_size=100.0, physics_norm=False, T_ambient=25.0):
    """
    Load count_sweep data for PINN training.
    Returns params_normalized, temps_raw (for loss computation).
    """
    params_raw = np.load(params_path).astype(np.float32)
    temps_raw  = np.load(temps_path).astype(np.float32)
    n_samples  = params_raw.shape[0]
    grid_total = temps_raw.shape[1]
    grid_size  = int(round(np.sqrt(grid_total)))
    assert grid_size * grid_size == grid_total

    params_3d = params_raw.reshape(n_samples, max_components, 3).copy()
    valid_mask = ~np.isnan(params_3d[:, :, 0])
    total_power = np.nansum(params_3d[:, :, 2], axis=1)
    total_power = np.maximum(total_power, 0.1)

    params_3d = np.nan_to_num(params_3d, nan=0.0)
    params_3d[:, :, 0] /= board_size
    params_3d[:, :, 1] /= board_size

    temps_2d_raw = temps_raw.reshape(n_samples, grid_size, grid_size)

    norm_info = {
        "board_size": board_size,
        "max_components": max_components,
        "physics_norm": physics_norm,
        "T_ambient": T_ambient,
    }

    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=test_ratio,
        random_state=split_seed, stratify=valid_mask.sum(axis=1))

    p_train  = params_3d[train_idx]
    t_train  = temps_2d_raw[train_idx]
    tp_train = total_power[train_idx]

    p_test   = params_3d[test_idx]
    t_test   = temps_2d_raw[test_idx]
    tp_test  = total_power[test_idx]

    if val_ratio > 0 and len(p_train) > 4:
        try:
            p_train, p_val, t_train, t_val, tp_train, tp_val = train_test_split(
                p_train, t_train, tp_train, test_size=val_ratio,
                random_state=split_seed)
        except ValueError:
            p_train, p_val, t_train, t_val, tp_train, tp_val = train_test_split(
                p_train, t_train, tp_train, test_size=val_ratio, random_state=split_seed)
    else:
        p_val, t_val, tp_val = None, None, None

    return (p_train.astype(np.float32), t_train.astype(np.float32), tp_train.astype(np.float32),
            p_val.astype(np.float32) if p_val is not None else None,
            t_val.astype(np.float32) if t_val is not None else None,
            tp_val.astype(np.float32) if tp_val is not None else None,
            p_test.astype(np.float32), t_test.astype(np.float32),
            tp_test.astype(np.float32),
            grid_size, norm_info)


def save_run_metadata(args, out_dir, model, n_params):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    payload = {
        "timestamp":    timestamp,
        "run_args":     vars(args),
        "model_params": int(n_params),
    }
    run_path    = os.path.join(out_dir, f"run_config_{timestamp}.json")
    latest_path = os.path.join(out_dir, "run_config_latest.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved run metadata: {run_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train PINN with corrected physics")

    parser.add_argument("--count-sweep-params", required=True)
    parser.add_argument("--count-sweep-temps",  required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=9)

    parser.add_argument("--physics-norm", action="store_true",
                        help="Normalise temps by total power: T_norm=(T-T_amb)/P_total")
    parser.add_argument("--t-ambient", type=float, default=25.0)

    parser.add_argument("--d-hidden",  type=int,   default=256)
    parser.add_argument("--n-layers",  type=int,   default=4)
    parser.add_argument("--n-freqs",   type=int,   default=64)
    parser.add_argument("--dropout",   type=float, default=0.0)

    # Physics loss weights (修正后的方程)
    parser.add_argument("--lambda-pde", type=float, default=0.001,
                        help="Weight for PDE residual loss")
    parser.add_argument("--lambda-bc", type=float, default=0.0001,
                        help="Weight for corrected BC loss")

    parser.add_argument("--epochs",       type=int,   default=1500)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--weight-decay",  type=float, default=1e-5)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--log-every",   type=int,   default=50)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience",     type=int,   default=200)
    parser.add_argument("--min-delta",    type=float, default=0.0)

    parser.add_argument("--out-dir",   default="results_pinn_physics_fix")
    parser.add_argument("--model-out", default="pinn_physics_fix_model.pth")
    parser.add_argument("--n-vis",     type=int, default=6)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  PINN with Corrected Physics (Phase Fix)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Architecture:", flush=True)
    print(f"  d_hidden={args.d_hidden}, n_layers={args.n_layers}, n_freqs={args.n_freqs}", flush=True)
    print(f"Physics loss (CORRECTED):", flush=True)
    print(f"  lambda_pde={args.lambda_pde}, lambda_bc={args.lambda_bc}", flush=True)
    print(f"BC equation: (k/dx2+h/dx)*T_edge - k/dx2*T_adj - h/dx*T_amb = 0", flush=True)
    print(f"  k=0.35 (FR4), h=30, dx=1/99=0.0101", flush=True)
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}", flush=True)
    print(f"T_ambient={args.t_ambient}", flush=True)

    (p_train, t_train, tp_train,
     p_val, t_val, tp_val,
     p_test, t_test, tp_test,
     grid_size, norm_info) = load_count_sweep_data_pinn(
        args.count_sweep_params, args.count_sweep_temps,
        max_components=args.n_components,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        split_seed=42,
        physics_norm=args.physics_norm,
        T_ambient=args.t_ambient)

    print(f"Train: {p_train.shape[0]}, Val: {p_val.shape[0] if p_val is not None else 0}, "
          f"Test: {p_test.shape[0]}", flush=True)
    print(f"Grid: {grid_size}, Param shape: {p_train.shape}", flush=True)

    train_ds = PINNDataset(p_train, t_train, tp_train)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                         shuffle=True, drop_last=True)
    val_ld = None
    if p_val is not None:
        val_ds = PINNDataset(p_val, t_val, tp_val)
        val_ld = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, drop_last=False)

    model = ThermalPINNPhysicsFix(
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        n_freqs=args.n_freqs,
        n_sources=args.max_components,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)
    save_run_metadata(args, args.out_dir, model, n_params)

    train_losses, val_losses, train_info = train_pinn_physics(
        model, train_ld, val_ld,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        log_every=args.log_every,
        device=device,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        out_dir=args.out_dir,
    )

    ckpt = {
        "state_dict":    model.state_dict(),
        "args":          vars(args),
        "train_info":    train_info,
        "grid_size":     grid_size,
        "norm_info":     norm_info,
    }
    torch.save(ckpt, os.path.join(args.out_dir, args.model_out))
    print(f"Model saved -> {args.out_dir}/{args.model_out}", flush=True)

    preds_scaled = predict_all_pinn(model, p_test, tp_test, device, args.batch_size)

    r2_vals, r2_avg = compute_r2(preds_scaled, t_test)
    print(f"\nPer-sample R²: {np.round(r2_vals, 4)}", flush=True)
    print(f"Average  R²  : {r2_avg:.4f}", flush=True)

    print("\nGenerating visualisations ...", flush=True)
    plot_loss_curves(train_losses, val_losses, args.out_dir)
    plot_r2_bar(r2_vals, args.out_dir)
    plot_thermal_comparisons(preds_scaled, t_test, args.out_dir, n_samples=args.n_vis)

    summary = {"r2_mean": float(r2_avg), "train_info": train_info}
    path = os.path.join(args.out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"All results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
