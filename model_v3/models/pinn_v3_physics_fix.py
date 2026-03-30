"""
pinn_v3_physics_fix.py
======================
修正物理方程的连续坐标 PINN。

【物理方程修正 — 基于 thermal_prediction.py】

1. 边界条件 (BC)
   SOR 离散方程：count · T_edge = k/dx² · T_adj + h/dx · T_amb
   残差：L_edge = (k/dx²+h/dx)·T_edge - k/dx²·T_adj - h/dx·T_amb = 0

   参数：
     k      = 0.35 W/(m·K)   (FR4 基材，不是铝)
     h      = 30   W/(m²·K)  (对流系数)
     dx     = 1/99 ≈ 0.0101  (归一化坐标，grid=100, board=100mm)

   系数：
     a_edge = k/dx² + h/dx ≈ 3433.5 + 2970 = 6403.5
     a_adj  = k/dx²       ≈ 3433.5
     b_amb  = h/dx         ≈ 2970

   四边均相同（对称）:
     L_bc = Σ_edges [a_edge·T_edge - a_adj·T_adj - b_amb·T_amb]²

2. PDE (稳态)
   热源区外 interior：∇²T = 0
   残差：L_pde = Σ_interior_non_source [(T[i+1]+T[i-1]+T[j+1]+T[j-1] - 4·T[i,j]) / 4]²

   注意：不对热源区约束（那里的温度由 Q 决定），也不用错误的热扩散系数。

3. 热源掩码
   组件占据约 8×8 个网格点（10mm×10mm / 1mm×1mm）
   用距离 < 0.05 (归一化) 来判断热源位置。

架构：
  - Fourier Feature Encoding (坐标 → 高维特征)
  - MLP (共享参数，对所有坐标点)
  - 输出：T(x,y) = R_th(x,y) * P + T_ambient
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fourier Feature Encoding (与 v3 相同)
# ══════════════════════════════════════════════════════════════════════════════

class FourierFeatures(nn.Module):
    """
    Maps (x, y) → Fourier-encoded features.
    Uses fixed frequencies (not learned) — powers of 2 up to num_freqs.
    """
    def __init__(self, in_dim=2, num_freqs=64):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        B = freqs.unsqueeze(0).repeat(in_dim, 1) * 2 * torch.pi
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  修正物理方程的 PINN 模型
# ══════════════════════════════════════════════════════════════════════════════

class ThermalPINNPhysicsFix(nn.Module):
    """
    连续坐标 PINN，修正了物理方程。

    Forward:
      coords (x,y) + heat_sources (N×3) + total_power
        → Fourier Encoding → MLP → R_th(x,y)
        → T(x,y) = R_th(x,y) * P + T_ambient
    """
    def __init__(self, d_hidden=128, n_layers=4, n_freqs=64, n_sources=9,
                 dropout=0.0):
        super().__init__()
        self.n_sources = n_sources

        self.fourier = FourierFeatures(in_dim=2, num_freqs=n_freqs)
        d_coord = n_freqs * 2

        self.source_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        d_source = 64

        self.power_embed = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
        )
        d_total = 32

        d_in = d_coord + d_source + d_total

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_hidden)
        self.fc4 = nn.Linear(d_hidden, d_hidden)
        self.head = nn.Linear(d_hidden, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = F.gelu

        # 预计算网格坐标
        grid_size = 100
        xs = torch.linspace(0, 1, grid_size, dtype=torch.float32)
        ys = torch.linspace(0, 1, grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

        # ── 修正物理常数 ───────────────────────────────────────────────
        # 来自 thermal_prediction.py
        self.k_fr4 = 0.35        # W/(m·K) FR4 基材
        self.k_al  = 180.0      # W/(m·K) 铝（组件）
        self.h_conv = 30.0       # W/(m²·K) 对流系数
        self.T_ambient = 25.0    # °C

        # 归一化坐标：board=100mm, grid=100 → dx=1mm
        # 归一化后：dx_norm = 1/99 ≈ 0.0101 (grid points 0..99)
        self.dx_norm = 1.0 / (grid_size - 1)  # = 1/99 ≈ 0.0101

        # BC 系数（修正后的正确形式）
        # count = k/dx² + h/dx
        # residual = count * T_edge - k/dx² * T_adj - h/dx * T_amb
        self.k_edge = self.k_fr4  # 边界是 FR4
        a_adj  = self.k_edge / (self.dx_norm ** 2)    # k/dx²
        b_amb  = self.h_conv / self.dx_norm            # h/dx
        self.a_edge = a_adj + b_amb                     # k/dx² + h/dx

        print(f"[Physics Fix] k_edge={self.k_edge}, h={self.h_conv}, dx={self.dx_norm:.4f}", flush=True)
        print(f"[Physics Fix] BC: a_edge={self.a_edge:.2f}, a_adj={a_adj:.2f}, b_amb={b_amb:.2f}", flush=True)

    def encode_sources(self, x):
        return self.source_encoder(x)

    def forward(self, coords, heat_sources, total_power):
        B, N = coords.shape[:2]

        coord_feats = self.fourier(coords)

        src_feats = self.encode_sources(heat_sources)
        src_pool = src_feats.mean(dim=1) + 0.1 * src_feats.max(dim=1)[0]

        if isinstance(total_power, float):
            total_power = torch.full((B,), total_power, device=coords.device)
        p_emb = self.power_embed(total_power.unsqueeze(-1))

        x = torch.cat([
            coord_feats,
            src_pool.unsqueeze(1).expand(-1, N, -1),
            p_emb.unsqueeze(1).expand(-1, N, -1)
        ], dim=-1)

        h = self.activation(self.fc1(x))
        if self.dropout:
            h = self.dropout(h)
        h = h + self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        if self.dropout:
            h = self.dropout(h)
        h = h + self.activation(self.fc4(h))
        R_th = self.head(h)

        T = R_th * total_power.unsqueeze(-1).unsqueeze(-1) + self.T_ambient
        return T

    def predict_grid(self, heat_sources, total_power):
        B = heat_sources.shape[0]
        grid = self.grid_x.shape[0]

        coords = torch.stack([self.grid_x.reshape(-1), self.grid_y.reshape(-1)], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1).to(heat_sources.device)

        T_flat = self.forward(coords, heat_sources, total_power).squeeze(-1)
        T = T_flat.reshape(B, grid, grid)
        return T

    def compute_physics_loss(self, heat_sources, total_power, eps=1e-6):
        """
        修正物理方程的 PDE 和 BC loss。

        BC (修正后): residual = a_edge·T_edge - a_adj·T_adj - b_amb·T_amb
        PDE (interior, non-source): Laplacian → 0
        """
        B = heat_sources.shape[0]
        device = heat_sources.device
        grid = self.grid_x.shape[0]

        coords = torch.stack([self.grid_x.reshape(-1), self.grid_y.reshape(-1)], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1).to(device)
        coords.requires_grad_(True)

        T_flat = self.forward(coords, heat_sources, total_power).squeeze(-1)
        T_grid = T_flat.reshape(B, grid, grid)

        # ── 热源掩码 ───────────────────────────────────────────────────
        # 组件约 10mm×10mm，归一化后 ≈ 0.1×0.1
        # 距离热源中心 < 0.05 的点视为热源区
        hs_x = heat_sources[:, :, 0:1]
        hs_y = heat_sources[:, :, 1:2]

        gx = self.grid_x.reshape(-1).to(device)
        gy = self.grid_y.reshape(-1).to(device)

        dx = gx.unsqueeze(0).unsqueeze(0) - hs_x.unsqueeze(-1)
        dy = gy.unsqueeze(0).unsqueeze(0) - hs_y.unsqueeze(-1)
        dist_sq = dx**2 + dy**2
        min_dist = dist_sq.min(dim=1)[0]
        # 热源掩码：距离任意组件中心 < 0.06 的点
        is_source = (min_dist < 0.06).float().reshape(B, 1, grid, grid)

        # ── BC Loss (修正后，归一化) ─────────────────────────────────
        # residual = a_edge * T_edge - a_adj * T_adj - b_amb * T_amb
        # 归一化后: (T_edge - (a_adj/a_edge)*T_adj - (b_amb/a_edge)*T_amb)²
        # 这样 loss 的量级就是温度误差的平方 (~1-100) 而不是 ~10^9
        a_edge = self.a_edge
        a_adj  = self.k_edge / (self.dx_norm ** 2)   # k/dx²
        b_amb  = self.h_conv / self.dx_norm            # h/dx
        c_adj  = a_adj / a_edge                        # ≈ 0.536
        c_amb  = b_amb / a_edge                        # ≈ 0.464

        # 顶部边界 (j=0, y=0)
        T_edge_top = T_grid[:, 0, :]       # (B, 100)
        T_adj_top  = T_grid[:, 1, :]        # (B, 100)
        bc_top = ((T_edge_top - c_adj * T_adj_top - c_amb * self.T_ambient) ** 2).mean()

        # 底部边界 (j=99, y=1)
        T_edge_bot = T_grid[:, -1, :]
        T_adj_bot  = T_grid[:, -2, :]
        bc_bottom = ((T_edge_bot - c_adj * T_adj_bot - c_amb * self.T_ambient) ** 2).mean()

        # 左边界 (i=0, x=0)
        T_edge_left = T_grid[:, :, 0]
        T_adj_left  = T_grid[:, :, 1]
        bc_left = ((T_edge_left - c_adj * T_adj_left - c_amb * self.T_ambient) ** 2).mean()

        # 右边界 (i=99, x=1)
        T_edge_right = T_grid[:, :, -1]
        T_adj_right  = T_grid[:, :, -2]
        bc_right = ((T_edge_right - c_adj * T_adj_right - c_amb * self.T_ambient) ** 2).mean()

        L_bc = bc_top + bc_bottom + bc_left + bc_right

        # ── PDE Loss ─────────────────────────────────────────────────
        # interior non-source 点：∇²T ≈ (T[i+1]+T[i-1]+T[j+1]+T[j-1] - 4T[i,j]) / 4 = 0
        # 注意：这里不乘以系数，只约束拉普拉斯算子为零
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]], dtype=torch.float32, device=device)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)

        T_bc = T_grid.unsqueeze(1)
        lap = F.conv2d(T_bc, lap_kernel, padding=0, stride=1)  # (B, 1, 98, 98)

        # interior mask
        interior_mask = torch.zeros(B, 1, grid, grid, device=device)
        interior_mask[:, :, 1:-1, 1:-1] = 1.0
        interior_mask = interior_mask[:, :, 1:-1, 1:-1]

        # 只对非热源的 interior 点约束 PDE
        non_source_interior = interior_mask * (1.0 - is_source[:, :, 1:-1, 1:-1])
        L_pde = ((lap ** 2) * non_source_interior).sum() / (non_source_interior.sum() + eps)

        return L_pde, L_bc


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PINNDataset(Dataset):
    def __init__(self, params, temps, total_power):
        self.params = torch.from_numpy(params).float()
        self.temps  = torch.from_numpy(temps).float()
        self.total_power = torch.from_numpy(total_power).float()

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.temps[idx], self.total_power[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_pinn_physics(model, train_loader, val_loader, epochs, lr, weight_decay,
                       lambda_pde=0.001, lambda_bc=0.0001,
                       log_every=50, device='cpu', early_stopping=False,
                       patience=200, min_delta=1e-6, out_dir='.'):
    """
    训练 PINN：Loss = MSE + λ_pde * L_pde + λ_bc * L_bc
    """
    mse = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    train_losses, val_losses = [], []
    best_state, best_val, patience_count, stopped_epoch = None, float("inf"), 0, epochs

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, running_data, running_pde, running_bc = 0.0, 0.0, 0.0, 0.0

        for xb, yb, pb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pb = pb.to(device)

            opt.zero_grad()

            T_pred = model.predict_grid(xb, pb)
            L_data = mse(T_pred, yb)

            L_pde, L_bc = model.compute_physics_loss(xb, pb)

            loss = L_data + lambda_pde * L_pde + lambda_bc * L_bc

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss  += loss.item() * xb.size(0)
            running_data  += L_data.item() * xb.size(0)
            running_pde   += L_pde.item() * xb.size(0)
            running_bc    += L_bc.item() * xb.size(0)

        sched.step()

        n_train = len(train_loader.dataset)
        train_loss = running_loss / n_train
        train_data = running_data / n_train
        train_pde  = running_pde / n_train
        train_bc   = running_bc / n_train
        train_losses.append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            running_v = 0.0
            with torch.no_grad():
                for xb, yb, pb in val_loader:
                    xb = xb.to(device); yb = yb.to(device); pb = pb.to(device)
                    T_pred = model.predict_grid(xb, pb)
                    running_v += mse(T_pred, yb).item() * xb.size(0)
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
            msg = f"Ep {ep:>4}/{epochs} Loss={train_loss:.6f} Data={train_data:.6f} PDE={train_pde:.6f} BC={train_bc:.6f}"
            if val_loader is not None:
                msg += f" Val={val_loss:.6f}"
            msg += f" LR={lr_now:.2e}"
            print(msg, flush=True)

        if early_stopping and patience_count >= patience:
            stopped_epoch = ep
            print(f"Early stopping at epoch {ep} (best val={best_val:.6f})", flush=True)
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, {
        "best_val": float(best_val),
        "stopped_epoch": int(stopped_epoch)
    }


def compute_r2(preds, true):
    """Compute per-sample R²."""
    r2s = []
    for i in range(len(preds)):
        p, t = preds[i].ravel(), true[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2s.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2s = np.array(r2s)
    finite = np.isfinite(r2s)
    return r2s, float(np.mean(r2s[finite])) if finite.any() else np.nan


def predict_all_pinn(model, params_scaled, total_power, device, batch_size=8):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(params_scaled), batch_size):
            xb = torch.from_numpy(params_scaled[i:i+batch_size]).float().to(device)
            pb = torch.from_numpy(total_power[i:i+batch_size]).float().to(device)
            preds.append(model.predict_grid(xb, pb).squeeze(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def plot_loss_curves(train_losses, val_losses, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.legend()
    plt.title("PINN Physics-Fix Training / Validation Loss")
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


def plot_thermal_comparisons(preds, truths, out_dir, n_samples=6):
    n = min(n_samples, len(preds))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        im0 = axes[i, 0].imshow(truths[i], cmap='jet', origin='lower')
        axes[i, 0].set_title(f"True {i+1}")
        plt.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(preds[i], cmap='jet', origin='lower')
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
