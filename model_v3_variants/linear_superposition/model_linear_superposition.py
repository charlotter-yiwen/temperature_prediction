"""
model_linear_superposition.py
=============================
Variant ②: Attention-weighted Per-Component Superposition

核心思路（简化版，避免内存爆炸）：
- 保留原始全局 SetTransformerEncoder（捕获组件交互）
- 在解码器之前：学习 N 组解码器权重（不是 N×完整解码器）
- 全局条件向量通过每组件权重生成贡献图
- 最终输出 = sum of weighted contributions + residual

参数量与原架构相近，但能建模每组件贡献。
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

TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # model_v3_variants
sys.path.insert(0, os.path.dirname(TP_DIR))  # temperature_prediction/

from models.set_fno_thermal import (
    SetTransformerEncoder, FNODecoder, SetFNOModel,
    load_count_sweep_data,
    compute_r2 as compute_r2_base,
    plot_loss_curves,
    plot_r2_bar,
    plot_thermal_comparisons,
    plot_scatter,
    inverse_transform_temps,
)


class ComponentContributionHead(nn.Module):
    """
    轻量级组件贡献头：
    - 输入: 全局条件向量 (B, d_model)
    - 输出: 每组件贡献权重 (B, N, d_model)
    - 然后通过共享解码器生成贡献图
    """
    def __init__(self, d_model: int, max_components: int = 10):
        super().__init__()
        self.max_components = max_components
        # 将全局向量扩展为 per-component 查询
        self.query_proj = nn.Linear(d_model, d_model * max_components)
        self.key_proj   = nn.Linear(d_model, d_model * max_components)
        # 每组件贡献权重
        self.contribution_gate = nn.Linear(d_model, max_components)

    def forward(self, global_cond: torch.Tensor, n_active: int) -> torch.Tensor:
        """
        global_cond: (B, d_model) 全局条件向量
        n_active: 活跃组件数量
        返回: (B, n_active) 每组件的贡献权重
        """
        # gates: (B, max_components)
        gates = torch.sigmoid(self.contribution_gate(global_cond))  # (B, max_components)
        # 取前 n_active 个
        weights = gates[:, :n_active]  # (B, n_active)
        # 归一化（使得权重和为1，但不强制）
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return weights


class LinearSuperpositionModel(nn.Module):
    """
    注意力加权组件叠加模型。

    流程：
      X (B, N, d_in)
        → SetTransformerEncoder → (B, d_model) 全局向量
        → ComponentContributionHead → (B, N) 权重
        → 条件向量 × 权重 → (B, N, d_model) 展开
        → FNODecoder → (B, 1, H, W)

    参数量: ~43M（与原架构相同）
    """
    def __init__(self, d_in: int = 3, d_model: int = 256, num_heads: int = 8,
                 n_sab: int = 4, fno_ch: int = 64, fno_modes: int = 24,
                 n_fno: int = 6, dropout: float = 0.0, out_size: int = 100,
                 max_components: int = 10):
        super().__init__()
        self.max_components = max_components

        # 原始全局编码器
        self.encoder = SetTransformerEncoder(d_in, d_model, num_heads, n_sab, dropout)

        # 组件贡献头
        self.contribution_head = ComponentContributionHead(d_model, max_components)

        # 条件展开：d_model → N*d_model 然后 reshape
        self.cond_expand = nn.Linear(d_model, d_model * max_components)

        # 共享解码器
        self.decoder = FNODecoder(d_model, fno_ch, fno_modes, n_fno, out_size=out_size)

        # 残差
        self.residual = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (B, N, d_in)
        返回: (B, 1, H, W)
        """
        B, N, _ = X.shape

        # 全局编码
        global_cond = self.encoder(X)  # (B, d_model)

        # 贡献权重
        weights = self.contribution_head(global_cond, N)  # (B, N)

        # 展开条件向量
        expanded = self.cond_expand(global_cond)  # (B, N*d_model)
        expanded = expanded.view(B, N, -1)  # (B, N, d_model)

        # 加权
        weighted = expanded * weights.unsqueeze(-1)  # (B, N, d_model)

        # 通过解码器（每次迭代 detach 释放激活，节省内存）
        T_sum = torch.zeros(B, 1, 100, 100, device=X.device, dtype=X.dtype)

        for i in range(N):
            w = weights[:, i:i+1].unsqueeze(-1)  # (B, 1, 1)
            Ti = self.decoder(weighted[:, i, :].detach())  # detach 切断计算图
            T_sum = T_sum + w * Ti

        # 残差修正
        return T_sum + self.residual(T_sum)


# ── 包装器 ──────────────────────────────────────────────────────────────────

class PlanAPlusPhysics(nn.Module):
    def __init__(self, base_model, k_fr4=0.35, h_conv=30.0, dx_norm=1.0/99.0):
        super().__init__()
        self.base = base_model
        self.k_fr4 = k_fr4
        self.h_conv = h_conv
        self.dx_norm = dx_norm

        grid = 100
        xs = torch.linspace(0, 1, grid, dtype=torch.float32)
        ys = torch.linspace(0, 1, grid, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

        k_dx2 = k_fr4 / (dx_norm ** 2)
        h_dx  = h_conv / dx_norm
        self.c_adj = k_dx2 / (k_dx2 + h_dx)

        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap_kernel)

        m = torch.zeros(1, 1, grid, grid, dtype=torch.float32)
        m[:, :, 1:-1, 1:-1] = 1.0
        self.register_buffer('interior_mask', m)

    def forward(self, x):
        return self.base(x)

    def compute_physics_loss(self, xb, eps=1e-6):
        B, n_comp, _ = xb.shape
        device = xb.device

        T_pred = self.base(xb).squeeze(1)

        c = self.c_adj
        bc_top    = ((T_pred[:, 0, :]    - c * T_pred[:, 1, :]   ) ** 2).mean()
        bc_bottom = ((T_pred[:, -1, :]  - c * T_pred[:, -2, :]  ) ** 2).mean()
        bc_left   = ((T_pred[:, :, 0]   - c * T_pred[:, :, 1]   ) ** 2).mean()
        bc_right  = ((T_pred[:, :, -1]   - c * T_pred[:, :, -2]  ) ** 2).mean()
        L_bc = bc_top + bc_bottom + bc_left + bc_right

        hs_x = xb[:, :, 0:1]
        hs_y = xb[:, :, 1:2]
        gx = self.grid_x.unsqueeze(0)
        gy = self.grid_y.unsqueeze(0)
        dx = hs_x.unsqueeze(-1) - gx.unsqueeze(1)
        dy = hs_y.unsqueeze(-1) - gy.unsqueeze(1)
        dist_sq = dx * dx + dy * dy
        min_dist, _ = dist_sq.min(dim=1)
        is_source = (min_dist < 0.06).float()

        lap_kernel = self.lap_kernel.view(1, 1, 3, 3)
        lap = F.conv2d(T_pred.unsqueeze(1), lap_kernel, padding=0, stride=1)
        interior_mask = self.interior_mask[:, :, 1:-1, 1:-1]
        non_source = interior_mask * (1.0 - is_source[:, 1:-1, 1:-1])
        L_pde = ((lap ** 2) * non_source).sum() / (non_source.sum() + eps)

        return L_pde, L_bc


class ThermalDatasetWithPower(Dataset):
    def __init__(self, params, temps, total_power):
        self.params = torch.from_numpy(params).float()
        self.temps  = torch.from_numpy(temps[:, None, :, :]).float()
        self.total_power = torch.from_numpy(total_power).float()
    def __len__(self):
        return self.params.shape[0]
    def __getitem__(self, idx):
        return self.params[idx], self.temps[idx], self.total_power[idx]


def train_plan_a_physics(model, train_loader, val_loader, scaler_y,
                         epochs, lr, weight_decay, device,
                         lambda_pde=0.0, lambda_bc=0.0,
                         log_every=50, early_stopping=False,
                         patience=200, min_delta=1e-6, out_dir='.'):
    mse = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)

    train_losses, val_losses = [], []
    best_state, best_val, patience_count = None, float("inf"), 0
    stopped_epoch = epochs

    for ep in range(1, epochs + 1):
        model.train()
        rt, rd, rp, rb = 0.0, 0.0, 0.0, 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            L_data = mse(pred, yb)
            if lambda_pde > 0 or lambda_bc > 0:
                L_pde, L_bc = model.compute_physics_loss(xb)
            else:
                L_pde = L_bc = torch.tensor(0.0, device=device)
            loss = L_data + lambda_pde * L_pde + lambda_bc * L_bc
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            rt += loss.item() * xb.size(0)
            rd += L_data.item() * xb.size(0)
            rp += L_pde.item() * xb.size(0)
            rb += L_bc.item() * xb.size(0)
        sched.step()

        n = len(train_loader.dataset)
        train_losses.append(rt / n)
        val_loss = float("nan")
        if val_loader:
            model.eval()
            rv = 0.0
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    rv += mse(model(xb), yb).item() * xb.size(0)
            val_loss = rv / len(val_loader.dataset)
            val_losses.append(val_loss)
            if early_stopping:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    patience_count = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_count += 1

        if ep % log_every == 0 or ep == 1:
            lr_n = sched.get_last_lr()[0]
            msg = f"Ep {ep:>4}/{epochs} Loss={rt/n:.6f} Data={rd/n:.6f} PDE={rp/n:.6f} BC={rb/n:.6f}"
            if val_loader: msg += f" Val={val_loss:.6f}"
            msg += f" LR={lr_n:.2e}"
            print(msg, flush=True)

        if early_stopping and patience_count >= patience:
            stopped_epoch = ep
            print(f"Early stop @ {ep} (best={best_val:.6f})", flush=True)
            break

    if early_stopping and best_state:
        model.load_state_dict(best_state)

    return train_losses, val_losses, {"best_val": float(best_val), "stopped_epoch": int(stopped_epoch)}


def predict_all(model, params, device, batch_size):
    model.eval()
    preds = []
    n = params.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(params[i:i+batch_size]).float().to(device)
            pred = model(xb).squeeze(1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-sweep-params", required=True)
    parser.add_argument("--count-sweep-temps",  required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--d-per-comp",   type=int,   default=3)
    parser.add_argument("--physics-norm", action="store_true")
    parser.add_argument("--t-ambient", type=float, default=25.0)
    parser.add_argument("--d-model",   type=int,   default=256)
    parser.add_argument("--num-heads", type=int,   default=8)
    parser.add_argument("--n-sab",     type=int,   default=4)
    parser.add_argument("--fno-ch",    type=int,   default=64)
    parser.add_argument("--fno-modes", type=int,   default=24)
    parser.add_argument("--n-fno",     type=int,   default=6)
    parser.add_argument("--dropout",   type=float, default=0.0)
    parser.add_argument("--lambda-pde", type=float, default=0.001)
    parser.add_argument("--lambda-bc",  type=float, default=0.0005)
    parser.add_argument("--epochs",       type=int,   default=10000)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--log-every",   type=int,   default=50)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience",     type=int,   default=500)
    parser.add_argument("--min-delta",    type=float, default=0.0)
    parser.add_argument("--out-dir",   default="results_linear_superposition")
    parser.add_argument("--model-out", default="linear_superposition_phase2.pth")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--n-vis",     type=int, default=6)
    parser.add_argument("--max-components", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Variant 2: Attention-Weighted Component Superposition", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Architecture: d_model={args.d_model}, heads={args.num_heads}, "
          f"n_sab={args.n_sab}, fno_ch={args.fno_ch}, fno_modes={args.fno_modes}, n_fno={args.n_fno}", flush=True)
    print(f"Physics: lambda_pde={args.lambda_pde}, lambda_bc={args.lambda_bc}", flush=True)
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}", flush=True)

    (p_train, t_tr_s, p_val, t_val_s,
     p_test, t_te_raw, scaler_y, grid_size, norm_info) = load_count_sweep_data(
        args.count_sweep_params, args.count_sweep_temps,
        max_components=args.n_components, d_per_comp=args.d_per_comp,
        test_ratio=args.test_ratio, val_ratio=args.val_ratio,
        split_seed=42,
        physics_norm=args.physics_norm, T_ambient=args.t_ambient)

    tp_train = np.nansum(p_train[:, :, 2], axis=1).astype(np.float32)
    tp_val   = np.nansum(p_val[:, :, 2], axis=1).astype(np.float32)   if p_val   is not None else None
    tp_test  = np.nansum(p_test[:, :, 2], axis=1).astype(np.float32)

    print(f"Train: {p_train.shape[0]}, Val: {p_val.shape[0] if p_val is not None else 0}, "
          f"Test: {p_test.shape[0]}", flush=True)

    train_ds = ThermalDatasetWithPower(p_train, t_tr_s, tp_train)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_ld = None
    if p_val is not None:
        val_ds = ThermalDatasetWithPower(p_val, t_val_s, tp_val)
        val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    base_model = LinearSuperpositionModel(
        d_in=args.d_per_comp, d_model=args.d_model,
        num_heads=args.num_heads, n_sab=args.n_sab,
        fno_ch=args.fno_ch, fno_modes=args.fno_modes,
        n_fno=args.n_fno, dropout=args.dropout, out_size=grid_size,
        max_components=args.max_components,
    ).to(device)

    model = PlanAPlusPhysics(base_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        has_base_prefix = any(k.startswith("base.") for k in sd.keys())
        if has_base_prefix:
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("base."):
                    new_sd[k[5:]] = v
                elif k not in ("grid_x", "grid_y"):
                    new_sd[k] = v
            sd = new_sd
        model.load_state_dict(sd, strict=False)
        print(f"[Resumed from] {args.resume_from}", flush=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    payload = {"timestamp": ts, "run_args": vars(args), "model_params": int(n_params)}
    for suffix in [f"_{ts}.json", "_latest.json"]:
        with open(os.path.join(args.out_dir, "run_config" + suffix), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    train_losses, val_losses, train_info = train_plan_a_physics(
        model, train_ld, val_ld, scaler_y,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        device=device,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc,
        log_every=args.log_every,
        early_stopping=args.early_stopping,
        patience=args.patience, min_delta=args.min_delta, out_dir=args.out_dir,
    )

    ckpt = {
        "state_dict":      model.state_dict(),
        "scaler_y_mean":  scaler_y.mean_,
        "scaler_y_scale":  scaler_y.scale_,
        "args":            vars(args),
        "train_info":      train_info,
        "grid_size":       grid_size,
        "norm_info":       dict(norm_info) if norm_info else {},
    }
    torch.save(ckpt, os.path.join(args.out_dir, args.model_out))
    print(f"Model saved -> {args.out_dir}/{args.model_out}", flush=True)

    preds_scaled = predict_all(model.base, p_test, device, args.batch_size)
    t_te_grid = t_te_raw.reshape(-1, grid_size, grid_size)
    tp = tp_test if args.physics_norm else None
    preds_inv = inverse_transform_temps(preds_scaled, scaler_y, total_power=tp, T_ambient=args.t_ambient)

    r2_vals, r2_avg = compute_r2_base(preds_inv, t_te_grid)
    print(f"\nPer-sample R2: {np.round(r2_vals, 4)}", flush=True)
    print(f"Average  R2  : {r2_avg:.4f}", flush=True)

    plot_loss_curves(train_losses, val_losses, args.out_dir)
    plot_thermal_comparisons(preds_inv, t_te_grid, scaler_y, args.out_dir,
                            n_samples=args.n_vis, total_power=tp, T_ambient=args.t_ambient)
    plot_r2_bar(r2_vals, args.out_dir)
    plot_scatter(preds_inv, t_te_grid, scaler_y, args.out_dir,
                 total_power=tp, T_ambient=args.t_ambient)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"r2_mean": float(r2_avg), "train_info": train_info}, f, indent=2)
    print(f"All results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
