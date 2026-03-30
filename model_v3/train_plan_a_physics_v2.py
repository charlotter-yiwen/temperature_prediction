"""
train_plan_a_physics_v2.py
==========================
Plan A + 物理约束，用简化的方式实现：
把物理 loss 集成到模型内部，避免在训练循环中传参数。

所有物理 loss 计算在模型的 compute_physics_loss() 方法内进行。
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TP_DIR)

from models.set_fno_thermal import (
    SetFNOModel,
    load_count_sweep_data,
    compute_r2 as compute_r2_base,
    plot_loss_curves,
    plot_r2_bar,
    plot_thermal_comparisons,
    plot_scatter,
    inverse_transform_temps,
    predict_all,
)


class PlanAPlusPhysics(nn.Module):
    """
    包装 Plan A 模型，增加 compute_physics_loss_on_grid 方法。
    在 forward 之后直接计算 BC + PDE loss，无需传额外参数。
    """
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
        self.register_buffer('grid_x', xx)  # (100, 100)
        self.register_buffer('grid_y', yy)

        # BC 系数
        k_dx2 = k_fr4 / (dx_norm ** 2)
        h_dx  = h_conv / dx_norm
        self.c_adj = k_dx2 / (k_dx2 + h_dx)  # ≈ 0.536
        print(f"[PlanA+Physics] c_adj={self.c_adj:.4f}, k_dx2={k_dx2:.2f}, h_dx={h_dx:.2f}", flush=True)

    def forward(self, x):
        return self.base(x)

    def compute_physics_loss(self, xb, eps=1e-6):
        """
        xb: (B, n_comp, 3) 归一化参数
        返回 (L_pde, L_bc)
        """
        B, n_comp, _ = xb.shape
        device = xb.device
        grid = 100

        # 前向传播
        T_pred = self.base(xb).squeeze(1)  # (B, 100, 100) normalized

        # ── BC Loss ─────────────────────────────────────────────
        c = self.c_adj
        bc_top    = ((T_pred[:, 0, :]    - c * T_pred[:, 1, :]   ) ** 2).mean()
        bc_bottom = ((T_pred[:, -1, :]  - c * T_pred[:, -2, :]  ) ** 2).mean()
        bc_left   = ((T_pred[:, :, 0]   - c * T_pred[:, :, 1]   ) ** 2).mean()
        bc_right  = ((T_pred[:, :, -1]   - c * T_pred[:, :, -2]  ) ** 2).mean()
        L_bc = bc_top + bc_bottom + bc_left + bc_right

        # ── 热源掩码 ───────────────────────────────────────────
        hs_x = xb[:, :, 0:1]   # (B, n_comp, 1)
        hs_y = xb[:, :, 1:2]

        # gx[b, i, j] = i/99
        gx = self.grid_x.unsqueeze(0).expand(B, -1, -1).to(device)   # (B, 100, 100)
        gy = self.grid_y.unsqueeze(0).expand(B, -1, -1).to(device)

        dx = hs_x.unsqueeze(-1) - gx.unsqueeze(1)   # (B, n_comp, 100, 100)
        dy = hs_y.unsqueeze(-1) - gy.unsqueeze(1)
        dist_sq = dx**2 + dy**2
        min_dist, _ = dist_sq.min(dim=1)              # (B, 100, 100)
        is_source = (min_dist < 0.06).float()         # (B, 100, 100)

        # ── PDE Loss ──────────────────────────────────────────
        lap_kernel = torch.tensor([[0., 1., 0.],
                                  [1., -4., 1.],
                                  [0., 1., 0.]], dtype=torch.float32, device=device)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = F.conv2d(T_pred.unsqueeze(1), lap_kernel, padding=0, stride=1)  # (B,1,98,98)

        interior_mask = torch.zeros(B, 1, grid, grid, device=device)
        interior_mask[:, :, 1:-1, 1:-1] = 1.0
        interior_mask = interior_mask[:, :, 1:-1, 1:-1]
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
    parser.add_argument("--lambda-pde", type=float, default=0.0)
    parser.add_argument("--lambda-bc",  type=float, default=0.0)
    parser.add_argument("--epochs",       type=int,   default=2000)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--log-every",   type=int,   default=50)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience",     type=int,   default=200)
    parser.add_argument("--min-delta",    type=float, default=0.0)
    parser.add_argument("--out-dir",   default="results_plan_a_physics")
    parser.add_argument("--model-out", default="plan_a_physics_model.pth")
    parser.add_argument("--n-vis",     type=int, default=6)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Plan A + Corrected Physics Loss v2", flush=True)
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

    base_model = SetFNOModel(
        d_in=args.d_per_comp, d_model=args.d_model,
        num_heads=args.num_heads, n_sab=args.n_sab,
        fno_ch=args.fno_ch, fno_modes=args.fno_modes,
        n_fno=args.n_fno, dropout=args.dropout, out_size=grid_size,
    ).to(device)

    model = PlanAPlusPhysics(base_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    # 保存配置
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

    # 保存
    ckpt = {
        "state_dict":      model.state_dict(),
        "scaler_y_mean":  scaler_y.mean_,
        "scaler_y_scale": scaler_y.scale_,
        "args":            vars(args),
        "train_info":      train_info,
        "grid_size":       grid_size,
        "norm_info":       dict(norm_info) if norm_info else {},
    }
    torch.save(ckpt, os.path.join(args.out_dir, args.model_out))
    print(f"Model saved -> {args.out_dir}/{args.model_out}", flush=True)

    # 评估
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
