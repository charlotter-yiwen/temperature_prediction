"""
train_setfno_30w.py
===================
使用 SetTransformer + FNO 架构训练 30W 数据集。

数据格式适配：
- 输入：JSON 文件，包含 components (center_mm, power_W) 和 temperature_data
- 坐标转换：毫米坐标 (0-100mm) → 归一化坐标 (0-1)
- 输出：100x100 温度场

用法:
  # Phase 1: data-only
  python train_setfno_30w.py \
      --data-dir ../../training_data_30W_test \
      --physics-norm \
      --t-ambient 25.0 \
      --epochs 2000 \
      --batch-size 32 \
      --lr 1e-4 \
      --early-stopping \
      --patience 200 \
      --out-dir ./results_setfno_phase1 \
      --model-out setfno_30w_phase1.pth

  # Phase 2: + physics loss
  python train_setfno_30w.py \
      --data-dir ../../training_data_30W_test \
      --physics-norm \
      --t-ambient 25.0 \
      --lambda-bc 0.01 \
      --lambda-pde 0.001 \
      --epochs 2000 \
      --batch-size 32 \
      --lr 5e-5 \
      --early-stopping \
      --patience 200 \
      --out-dir ./results_setfno_phase2 \
      --model-out setfno_30w_phase2.pth
"""

import os
import sys
import json
import argparse
from datetime import datetime
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

# 添加父目录到路径，以便导入 models
TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TP_DIR)

from models.set_fno_thermal import SetFNOModel


# ============================================================================
# 1. Data Loading for 30W JSON Format
# ============================================================================

def load_json_sample(json_path, max_components=9):
    """
    Load a single JSON sample and return (params, temps_2d, total_power)

    坐标转换：
    - center_mm: 毫米坐标 (0-100mm) → 归一化坐标 (0-1)
    - 温度场: 10000 个点 → 100x100 网格

    返回:
        params: (max_components, 3) - [x_norm, y_norm, power]
        temps: (100, 100) - 温度场
        total_power: 总功率
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    components = data['components']
    grid_size = data['simulation_params']['grid_size']
    ambient_temp = data['simulation_params']['ambient_temp_C']

    # 构建 params 数组 [x_norm, y_norm, power]
    # 使用 0 填充（不用 nan），因为 SetTransformer 可以处理 0 padding
    params = np.zeros((max_components, 3), dtype=np.float32)
    total_power = 0.0

    for i, comp in enumerate(components):
        # 坐标归一化：毫米 (0-100) → (0-1)
        cx = comp['center_mm'][0] / 100.0
        cy = comp['center_mm'][1] / 100.0
        p = comp['power_W']
        params[i] = [cx, cy, p]
        total_power += p

    # 构建 100x100 温度网格
    temp_data = data['temperature_data']
    T = np.zeros((grid_size, grid_size), dtype=np.float32)

    for td in temp_data:
        x = td['x']  # 0-10 范围
        y = td['y']  # 0-10 范围
        temp = td['temperature']

        # 坐标转换：x 是 0-10 范围，映射到 0-99 索引
        xi = int(round(x * 99 / 10))
        yi = int(round(y * 99 / 10))
        xi = max(0, min(grid_size - 1, xi))
        yi = max(0, min(grid_size - 1, yi))
        T[xi, yi] = temp

    # 填充零值格子（如果有）
    zero_count = np.sum(T == 0)
    if zero_count > 0:
        T[T == 0] = ambient_temp

    return params, T, total_power


def load_30w_dataset(data_dir, max_components=9):
    """
    Load all JSON files from the 30W dataset directory.

    返回:
        params: (N, max_components, 3) - 归一化坐标 + 归一化功率
        temps: (N, 100, 100) - 温度场
        powers: (N,) - 总功率 (原始值，用于物理归一化)
        max_power: 单个组件最大功率 (用于归一化)
    """
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    print(f"Found {len(json_files)} JSON files")

    all_params = []
    all_temps = []
    all_powers = []
    all_comp_powers = []  # 收集所有组件功率用于找最大值

    for i, fname in enumerate(json_files):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(json_files)}...")
        json_path = os.path.join(data_dir, fname)
        params, temps, total_power = load_json_sample(json_path, max_components)
        all_params.append(params)
        all_temps.append(temps)
        all_powers.append(total_power)
        # 收集非零功率
        for p in params:
            if p[2] > 0:
                all_comp_powers.append(p[2])

    # 找到最大功率用于归一化
    max_power = max(all_comp_powers) if all_comp_powers else 1.0
    print(f"Max component power: {max_power:.2f}W")

    # 归一化功率 (坐标已经在 load_json_sample 中归一化了)
    params_array = np.array(all_params, dtype=np.float32)
    params_array[:, :, 2] /= max_power  # 功率归一化到 0-1

    return (params_array,
            np.array(all_temps, dtype=np.float32),
            np.array(all_powers, dtype=np.float32),
            max_power)


def normalize_data(params, temps, powers, physics_norm=True, T_ambient=25.0):
    """
    数据归一化。

    Step 1: 物理归一化 (可选)
        temps_phys = (temps - T_ambient) / power

    Step 2: StandardScaler 标准化
        temps_scaled = (temps_phys - mean) / std

    返回:
        params: 保持不变 (已经归一化到 0-1)
        temps_scaled: 标准化后的温度场
        temps_raw: 原始温度场 (用于评估)
        scaler: StandardScaler 对象
        norm_info: 归一化信息
    """
    from sklearn.preprocessing import StandardScaler

    norm_info = {
        'physics_norm': physics_norm,
        'T_ambient': T_ambient,
    }

    # 保存原始温度场
    temps_raw = temps.copy()

    # Step 1: 物理归一化
    if physics_norm:
        # 物理归一化：T_norm = (T - T_amb) / P
        # 对每个样本除以其总功率，得到热阻场 (°C/W)
        temps_phys = np.zeros_like(temps)
        for i in range(len(temps)):
            if powers[i] > 0:
                temps_phys[i] = (temps[i] - T_ambient) / powers[i]
            else:
                temps_phys[i] = temps[i] - T_ambient
        print(f"Physics norm: thermal impedance range = {temps_phys.min():.4f} - {temps_phys.max():.4f} °C/W")
    else:
        # 简单归一化：减去环境温度
        temps_phys = temps - T_ambient

    # Step 2: StandardScaler 标准化
    scaler = StandardScaler()
    n_samples = temps_phys.shape[0]
    grid_size = temps_phys.shape[1]
    # 展平 -> fit_transform -> reshape
    temps_flat = temps_phys.reshape(n_samples, -1)
    temps_scaled = scaler.fit_transform(temps_flat).reshape(n_samples, grid_size, grid_size)
    temps_scaled = temps_scaled.astype(np.float32)

    print(f"StandardScaler: mean={scaler.mean_.mean():.6f}, std={scaler.scale_.mean():.6f}")
    print(f"Scaled temp range: {temps_scaled.min():.4f} - {temps_scaled.max():.4f}")

    norm_info['scaler_mean'] = scaler.mean_.tolist()
    norm_info['scaler_scale'] = scaler.scale_.tolist()

    return params, temps_scaled, temps_raw, scaler, norm_info


# ============================================================================
# 2. Dataset Class
# ============================================================================

class ThermalDataset30W(Dataset):
    """返回 (params, temps_norm, total_power)"""
    def __init__(self, params, temps, total_power):
        self.params = torch.from_numpy(params).float()
        self.temps = torch.from_numpy(temps[:, None, :, :]).float()  # (N, 1, 100, 100)
        self.total_power = torch.from_numpy(total_power).float()

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.temps[idx], self.total_power[idx]


# ============================================================================
# 3. Physics Loss
# ============================================================================

def compute_physics_loss_on_batch(model, params_b, pb_b, device,
                                   k_fr4=0.35, h_conv=30.0, eps=1e-6):
    """
    对一批归一化后的温度预测计算 BC loss 和 PDE loss。

    BC (修正后): residual = T_norm_edge - c_adj * T_norm_adj = 0
    其中 c_adj = k/dx² / (k/dx² + h/dx) ≈ 0.536
    """
    B = params_b.shape[0]
    grid = 100
    dx_norm = 1.0 / (grid - 1)  # = 1/99

    # 前向传播获取归一化温度
    xb = torch.from_numpy(params_b).float().to(device)
    T_pred = model(xb).squeeze(1)  # (B, 100, 100)

    # BC 系数
    k_dx2 = k_fr4 / (dx_norm ** 2)  # k/dx²
    h_dx = h_conv / dx_norm         # h/dx
    c_adj = k_dx2 / (k_dx2 + h_dx)  # ≈ 0.536

    # BC Loss — 四边
    bc_top = ((T_pred[:, 0, :] - c_adj * T_pred[:, 1, :]) ** 2).mean()
    bc_bottom = ((T_pred[:, -1, :] - c_adj * T_pred[:, -2, :]) ** 2).mean()
    bc_left = ((T_pred[:, :, 0] - c_adj * T_pred[:, :, 1]) ** 2).mean()
    bc_right = ((T_pred[:, :, -1] - c_adj * T_pred[:, :, -2]) ** 2).mean()
    L_bc = bc_top + bc_bottom + bc_left + bc_right

    # 热源掩码
    hs_x_t = torch.from_numpy(params_b[:, :, 0:1]).float().to(device)  # (B, N, 1)
    hs_y_t = torch.from_numpy(params_b[:, :, 1:2]).float().to(device)  # (B, N, 1)

    xs = torch.linspace(0, 1, grid, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    gx = xx.unsqueeze(0).expand(B, -1, -1)   # (B, 100, 100)
    gy = yy.unsqueeze(0).expand(B, -1, -1)

    # 正确的广播：hs_x_t (B, N, 1) -> (B, N, 1, 1), gx (B, 100, 100) -> (B, 1, 100, 100)
    dx = hs_x_t.unsqueeze(-1) - gx.unsqueeze(1)  # (B, N, 100, 100)
    dy = hs_y_t.unsqueeze(-1) - gy.unsqueeze(1)
    dist_sq = dx**2 + dy**2
    min_dist = dist_sq.min(dim=1)[0]  # (B, 100, 100)
    is_source = (min_dist < 0.06).float()

    # PDE Loss — interior non-source 点
    lap_kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32, device=device)
    lap_kernel = lap_kernel.view(1, 1, 3, 3)
    T_bc = T_pred.unsqueeze(1)  # (B, 1, 100, 100)
    lap = F.conv2d(T_bc, lap_kernel, padding=0, stride=1)  # (B, 1, 98, 98)

    interior_mask = torch.zeros(B, 1, grid, grid, device=device)
    interior_mask[:, :, 1:-1, 1:-1] = 1.0
    interior_mask = interior_mask[:, :, 1:-1, 1:-1]
    non_source_interior = interior_mask * (1.0 - is_source[:, 1:-1, 1:-1])
    L_pde = ((lap ** 2) * non_source_interior).sum() / (non_source_interior.sum() + eps)

    return L_pde, L_bc


# ============================================================================
# 4. Training Function
# ============================================================================

def train_model(model, train_loader, val_loader,
                epochs, lr, weight_decay, device,
                lambda_pde=0.0, lambda_bc=0.0,
                log_every=50, early_stopping=False,
                patience=200, min_delta=1e-6, out_dir='.'):
    """训练模型：Loss = MSE + λ_pde * L_pde + λ_bc * L_bc"""
    mse = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)

    train_losses, val_losses = [], []
    best_state, best_val, patience_count, stopped_epoch = None, float("inf"), 0, epochs

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, running_data, running_pde, running_bc = 0.0, 0.0, 0.0, 0.0

        for xb, yb, pb in train_loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            opt.zero_grad()

            pred = model(xb)
            L_data = mse(pred, yb)

            if lambda_pde > 0 or lambda_bc > 0:
                params_np = xb.detach().cpu().numpy()
                pb_np = pb.detach().cpu().numpy()
                L_pde, L_bc = compute_physics_loss_on_batch(
                    model, params_np, pb_np, device)
            else:
                L_pde = torch.tensor(0.0, device=device)
                L_bc = torch.tensor(0.0, device=device)

            loss = L_data + lambda_pde * L_pde + lambda_bc * L_bc
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss += loss.item() * xb.size(0)
            running_data += L_data.item() * xb.size(0)
            running_pde += L_pde.item() * xb.size(0)
            running_bc += L_bc.item() * xb.size(0)

        sched.step()

        n_train = len(train_loader.dataset)
        train_losses.append(running_loss / n_train)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            running_v = 0.0
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    running_v += mse(model(xb), yb).item() * xb.size(0)
            val_loss = running_v / len(val_loader.dataset)
            val_losses.append(val_loss)

            if early_stopping:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    patience_count = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_count += 1

        if ep % log_every == 0 or ep == 1:
            lr_now = sched.get_last_lr()[0]
            msg = (f"Ep {ep:>4}/{epochs} Loss={running_loss/n_train:.6f} "
                   f"Data={running_data/n_train:.6f} PDE={running_pde/n_train:.6f} "
                   f"BC={running_bc/n_train:.6f}")
            if val_loader is not None:
                msg += f" Val={val_loss:.6f}"
            msg += f" LR={lr_now:.2e}"
            print(msg, flush=True)

        if early_stopping and patience_count >= patience:
            print(f"Early stopping at epoch {ep} (best val={best_val:.6f})", flush=True)
            stopped_epoch = ep
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, {
        "best_val": float(best_val),
        "stopped_epoch": int(stopped_epoch if stopped_epoch else epochs)
    }


# ============================================================================
# 5. Evaluation & Visualization
# ============================================================================

def compute_r2(preds, truths):
    """Compute per-sample R²."""
    r2s = []
    for i in range(len(preds)):
        p = preds[i].flatten()
        t = truths[i].flatten()
        r2s.append(r2_score(t, p))
    return np.array(r2s), float(np.mean(r2s))


def plot_loss_curves(train_losses, val_losses, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"[Saved] {out_dir}/loss_curves.png")


def plot_r2_bar(r2_vals, out_dir):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(r2_vals)), r2_vals, color='steelblue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Sample")
    plt.ylabel("R²")
    plt.title(f"Per-sample R² (mean={np.mean(r2_vals):.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r2_scores.png"), dpi=150)
    plt.close()
    print(f"[Saved] {out_dir}/r2_scores.png")


def plot_thermal_comparisons(preds, truths, out_dir, n_samples=6, T_ambient=25.0):
    n = min(n_samples, len(preds))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        vmin = truths[i].min()
        vmax = truths[i].max()

        im0 = axes[i, 0].imshow(truths[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"True {i+1}")
        plt.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(preds[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Pred {i+1}")
        plt.colorbar(im1, ax=axes[i, 1])

        diff = preds[i] - truths[i]
        im2 = axes[i, 2].imshow(diff, cmap='RdBu', origin='lower', vmin=-5, vmax=5)
        axes[i, 2].set_title(f"Error {i+1}")
        plt.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thermal_comparisons.png"), dpi=150)
    plt.close()
    print(f"[Saved] {out_dir}/thermal_comparisons.png")


def plot_scatter(preds, truths, out_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(truths.flatten(), preds.flatten(), alpha=0.3, s=1)
    min_val = min(truths.min(), preds.min())
    max_val = max(truths.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel("True Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Prediction vs Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter.png"), dpi=150)
    plt.close()
    print(f"[Saved] {out_dir}/scatter.png")


def predict_all(model, params, device, batch_size=32):
    """Predict all samples in batches."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(params), batch_size):
            xb = torch.from_numpy(params[i:i+batch_size]).float().to(device)
            pred = model(xb).squeeze(1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def inverse_transform_temps(preds_scaled, scaler, total_power=None, T_ambient=25.0):
    """
    反变换预测结果：scaled → physical → raw temperature

    Step 1: scaler.inverse_transform (标准化 → 物理归一化)
    Step 2: 如果使用物理归一化，T = T_norm * P + T_amb

    Args:
        preds_scaled: (N, H, W) 标准化后的预测
        scaler: StandardScaler 对象
        total_power: (N,) 每个样本的总功率，用于物理归一化反变换
        T_ambient: 环境温度

    Returns:
        temps_raw: (N, H, W) 原始温度预测
    """
    n_samples = preds_scaled.shape[0]
    grid_size = preds_scaled.shape[1]

    # Step 1: 反 StandardScaler
    preds_flat = preds_scaled.reshape(n_samples, -1)
    preds_phys = scaler.inverse_transform(preds_flat).reshape(n_samples, grid_size, grid_size)

    # Step 2: 反物理归一化
    if total_power is not None:
        # T = T_norm * P + T_amb
        temps_raw = preds_phys * total_power[:, None, None] + T_ambient
    else:
        temps_raw = preds_phys + T_ambient

    return temps_raw.astype(np.float32)


# ============================================================================
# 6. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SetFNO for 30W Thermal Data")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default=r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test',
                        help="Directory containing JSON files")
    parser.add_argument("--max-components", type=int, default=9,
                        help="Maximum number of components (padding)")
    parser.add_argument("--physics-norm", action="store_true",
                        help="Use physics-based normalization")
    parser.add_argument("--t-ambient", type=float, default=25.0,
                        help="Ambient temperature")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--n-sab", type=int, default=4)
    parser.add_argument("--fno-ch", type=int, default=64)
    parser.add_argument("--fno-modes", type=int, default=24)
    parser.add_argument("--n-fno", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Physics loss
    parser.add_argument("--lambda-pde", type=float, default=0.0)
    parser.add_argument("--lambda-bc", type=float, default=0.0)

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=0.0)

    # Output
    parser.add_argument("--out-dir", default="./results_setfno")
    parser.add_argument("--model-out", default="setfno_30w.pth")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--n-vis", type=int, default=6)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  SetFNO for 30W Thermal Data", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Architecture: d_model={args.d_model}, heads={args.num_heads}, "
          f"n_sab={args.n_sab}, fno_ch={args.fno_ch}, fno_modes={args.fno_modes}, "
          f"n_fno={args.n_fno}", flush=True)
    print(f"Physics: lambda_pde={args.lambda_pde}, lambda_bc={args.lambda_bc}", flush=True)
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}", flush=True)

    # ========== Load Data ==========
    print(f"\n=== Loading Data ===", flush=True)
    params, temps, powers, max_power = load_30w_dataset(args.data_dir, args.max_components)
    print(f"Loaded: params={params.shape}, temps={temps.shape}, powers={powers.shape}")
    print(f"Total power range: {powers.min():.2f}W - {powers.max():.2f}W")
    print(f"Temperature range: {temps.min():.2f}°C - {temps.max():.2f}°C")
    print(f"Max component power (for normalization): {max_power:.2f}W")

    # ========== Normalize ==========
    params_norm, temps_norm, temps_raw, scaler, norm_info = normalize_data(
        params, temps, powers,
        physics_norm=args.physics_norm,
        T_ambient=args.t_ambient
    )
    norm_info["max_power"] = float(max_power)

    # ========== Split ==========
    n_samples = len(params_norm)
    indices = np.arange(n_samples)

    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(indices,
                                            test_size=args.val_ratio + args.test_ratio,
                                            random_state=42)
    # Second split: val vs test
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_idx, test_idx = train_test_split(temp_idx,
                                          test_size=1 - val_ratio_adjusted,
                                          random_state=42)

    print(f"\nSplit: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    p_train, t_train_scaled, pw_train = params_norm[train_idx], temps_norm[train_idx], powers[train_idx]
    p_val, t_val_scaled, pw_val = params_norm[val_idx], temps_norm[val_idx], powers[val_idx]
    p_test, t_test_scaled, pw_test = params_norm[test_idx], temps_norm[test_idx], powers[test_idx]
    t_test_raw = temps_raw[test_idx]  # 原始温度场用于评估

    # ========== Create Datasets ==========
    train_ds = ThermalDataset30W(p_train, t_train_scaled, pw_train)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_ld = None
    if len(val_idx) > 0:
        val_ds = ThermalDataset30W(p_val, t_val_scaled, pw_val)
        val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ========== Create Model ==========
    print(f"\n=== Creating Model ===", flush=True)
    model = SetFNOModel(
        d_in=3,  # (x, y, power)
        d_model=args.d_model,
        num_heads=args.num_heads,
        n_sab=args.n_sab,
        fno_ch=args.fno_ch,
        fno_modes=args.fno_modes,
        n_fno=args.n_fno,
        dropout=args.dropout,
        out_size=100,  # 100x100 grid
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    # Save run config
    run_config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "args": vars(args),
        "model_params": n_params,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "norm_info": norm_info,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), 'w') as f:
        json.dump(run_config, f, indent=2)

    # ========== Train ==========
    print(f"\n=== Training ===", flush=True)
    train_losses, val_losses, train_info = train_model(
        model, train_ld, val_ld,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        log_every=args.log_every,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        out_dir=args.out_dir,
    )

    # ========== Save Model ==========
    ckpt = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "train_info": train_info,
        "norm_info": norm_info,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    model_path = os.path.join(args.out_dir, args.model_out)
    torch.save(ckpt, model_path)
    print(f"Model saved -> {model_path}", flush=True)

    # ========== Evaluate ==========
    print(f"\n=== Evaluation ===", flush=True)
    preds_scaled = predict_all(model, p_test, device, args.batch_size)

    # Inverse transform 步骤:
    # 1. scaler.inverse_transform: scaled → physics-normalized
    # 2. 如果 physics_norm: T_phys * P + T_amb → 原始温度
    #    否则: T_phys + T_amb → 原始温度

    # Step 1: 反 StandardScaler
    preds_flat = preds_scaled.reshape(len(preds_scaled), -1)
    preds_phys = scaler.inverse_transform(preds_flat).reshape(-1, 100, 100)

    # Step 2: 反物理归一化
    if args.physics_norm:
        preds = preds_phys * pw_test[:, None, None] + args.t_ambient
    else:
        preds = preds_phys + args.t_ambient

    # truths 直接使用原始温度场
    truths = t_test_raw

    r2_vals, r2_avg = compute_r2(preds, truths)
    print(f"Per-sample R²: {np.round(r2_vals, 4)}", flush=True)
    print(f"Average R²: {r2_avg:.4f}", flush=True)
    print(f"Min R²: {r2_vals.min():.4f}, Max R²: {r2_vals.max():.4f}", flush=True)

    # ========== Visualization ==========
    plot_loss_curves(train_losses, val_losses, args.out_dir)
    plot_thermal_comparisons(preds, truths, args.out_dir, n_samples=args.n_vis, T_ambient=args.t_ambient)
    plot_r2_bar(r2_vals, args.out_dir)
    plot_scatter(preds, truths, args.out_dir)

    # Save summary
    summary = {
        "r2_mean": float(r2_avg),
        "r2_std": float(np.std(r2_vals)),
        "r2_min": float(r2_vals.min()),
        "r2_max": float(r2_vals.max()),
        "train_info": train_info,
    }
    with open(os.path.join(args.out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {args.out_dir}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
