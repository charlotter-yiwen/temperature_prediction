"""
set_fno_thermal.py
==================
Variable-component thermal field prediction using:
  - Set Transformer  (permutation-invariant encoder, handles any N components)
  - FNO 2D           (Fourier Neural Operator decoder, predicts 200x200 field)

Usage example:
  python set_fno_thermal.py \
      --train-params "thermal_analysis_output/training data/params_training.npy" \
      --train-temps  "thermal_analysis_output/training data/temps_training.npy"  \
      --test-params  "thermal_analysis_output/test data/params_testing.npy"      \
      --test-temps   "thermal_analysis_output/test data/temps_testing.npy"       \
      --n-components 4 --d-per-comp 1
"""

import argparse
import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE_HPARAMS = {
    "architecture": {
        "d_model": 128,
        "num_heads": 4,
        "n_sab": 2,
        "fno_ch": 32,
        "fno_modes": 12,
        "n_fno": 4,
        "dropout": 0.0,
    },
    "training": {
        "epochs": 2000,
        "batch_size": 8,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "val_ratio": 0.1,
        "log_every": 50,
    },
    "data": {
        "n_components": 4,
        "d_per_comp": 1,
        "xi": 200,
        "yi": 200,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Set Transformer
# ══════════════════════════════════════════════════════════════════════════════

class MAB(nn.Module):
    """Multi-head Attention Block: Q attends to K / V."""
    def __init__(self, d: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(Q, K, K)
        x = self.norm1(Q + attn_out)
        return self.norm2(x + self.ff(x))


class SAB(nn.Module):
    """Self-Attention Block."""
    def __init__(self, d: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(d, num_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class PMA(nn.Module):
    """Pooling by Multihead Attention — collapses N elements → k seeds."""
    def __init__(self, d: int, num_heads: int, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S   = nn.Parameter(torch.randn(1, k, d))
        self.rff = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.mab = MAB(d, num_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        S = self.S.expand(X.size(0), -1, -1)
        return self.mab(S, self.rff(X))          # (B, k, d)


class SetTransformerEncoder(nn.Module):
    """
    Encodes a *set* of N components (each d_in-dimensional) into a single
    d_model vector — invariant to component ordering and works for any N.
    """
    def __init__(self, d_in: int, d_model: int = 128,
                 num_heads: int = 4, n_sab: int = 2, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.sabs = nn.ModuleList([SAB(d_model, num_heads, dropout) for _ in range(n_sab)])
        self.pma  = PMA(d_model, num_heads, k=1, dropout=dropout)
        self.out  = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X : (B, N, d_in)  →  (B, d_model)"""
        h = self.proj(X)
        for sab in self.sabs:
            h = sab(h)
        h = self.pma(h).squeeze(1)   # (B, d_model)
        return self.out(h)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FNO 2-D  (Fourier Neural Operator)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.W1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.W2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xft   = torch.fft.rfft2(x)
        half  = W // 2 + 1
        out   = torch.zeros(B, self.out_ch, H, half, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1,  :m2] = self._mul(xft[:, :, :m1,  :m2], self.W1)
        out[:, :, -m1:, :m2] = self._mul(xft[:, :, -m1:, :m2], self.W2)
        return torch.fft.irfft2(out, s=(H, W))


class FNOBlock2d(nn.Module):
    def __init__(self, ch: int, modes1: int, modes2: int):
        super().__init__()
        self.spec   = SpectralConv2d(ch, ch, modes1, modes2)
        self.bypass = nn.Conv2d(ch, ch, 1)
        self.norm   = nn.InstanceNorm2d(ch, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spec(x) + self.bypass(x)))


class FNODecoder(nn.Module):
    """
    Takes a condition vector (B, d_cond) and produces a spatial field (B,1,H,W).
    Steps:  project → reshape (B,ch,H/4,W/4) → 2× upsample → N FNO blocks → 1×1 conv.
    """
    def __init__(self, d_cond: int = 128, ch: int = 32,
                 modes: int = 12, n_layers: int = 4, out_size: int = 200):
        super().__init__()
        self.out_size  = out_size
        self.init_size = out_size // 4          # 50

        self.cond_proj = nn.Linear(d_cond, ch * self.init_size * self.init_size)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1), nn.GELU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1), nn.GELU())
        self.fno_blocks = nn.ModuleList(
            [FNOBlock2d(ch, modes, modes) for _ in range(n_layers)])
        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.size(0)
        h = self.cond_proj(cond).view(B, -1, self.init_size, self.init_size)
        h = self.up1(h)
        h = self.up2(h)
        for blk in self.fno_blocks:
            h = blk(h)
        return self.out_conv(h)                 # (B, 1, 200, 200)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Combined model
# ══════════════════════════════════════════════════════════════════════════════

class SetFNOModel(nn.Module):
    """
    Full model: Set Transformer encoder  +  FNO decoder.

    Input  X : (B, N, d_in)   — N components, each with d_in features.
    Output   : (B, 1, 200, 200) temperature field.
    """
    def __init__(self, d_in: int = 1, d_model: int = 128, num_heads: int = 4,
                 n_sab: int = 2, fno_ch: int = 32, fno_modes: int = 12,
                 n_fno: int = 4, dropout: float = 0.0):
        super().__init__()
        self.encoder = SetTransformerEncoder(d_in, d_model, num_heads, n_sab, dropout)
        self.decoder = FNODecoder(d_model, fno_ch, fno_modes, n_fno)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        cond = self.encoder(X)                  # (B, d_model)
        return self.decoder(cond)               # (B, 1, 200, 200)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def fill_nan_per_row(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    for i in range(flat.shape[0]):
        row  = flat[i]
        mask = np.isnan(row)
        if mask.any():
            mean_val = np.nanmean(row)
            row[mask] = 0.0 if np.isnan(mean_val) else mean_val
    return flat.reshape(arr.shape)


def load_data(train_p, train_t, test_p, test_t, val_ratio, n_comp, d_per_comp, split_seed=42):
    p_tr = np.load(train_p)
    t_tr = np.load(train_t)
    p_te = np.load(test_p)
    t_te = np.load(test_t)

    t_tr = fill_nan_per_row(t_tr)
    t_te = fill_nan_per_row(t_te)

    # Reshape params to (N_samples, n_comp, d_per_comp) if needed
    def reshape_params(p):
        if p.ndim == 2:
            total = p.shape[1]
            assert total == n_comp * d_per_comp, (
                f"params dim {total} ≠ n_comp({n_comp}) × d_per_comp({d_per_comp})")
            return p.reshape(p.shape[0], n_comp, d_per_comp)
        return p

    p_tr = reshape_params(p_tr)
    p_te = reshape_params(p_te)

    # Standardise params per feature dimension
    orig_shape_tr = p_tr.shape
    orig_shape_te = p_te.shape
    scaler_x = StandardScaler()
    p_tr_s = scaler_x.fit_transform(p_tr.reshape(p_tr.shape[0], -1)).reshape(orig_shape_tr).astype(np.float32)
    p_te_s = scaler_x.transform    (p_te.reshape(p_te.shape[0], -1)).reshape(orig_shape_te).astype(np.float32)

    # Standardise temperatures
    scaler_y = StandardScaler()
    h, w = 200, 200
    t_tr_f = t_tr.reshape(t_tr.shape[0], -1)
    t_te_f = t_te.reshape(t_te.shape[0], -1)
    t_tr_s = scaler_y.fit_transform(t_tr_f).reshape(-1, h, w).astype(np.float32)
    t_te_s = scaler_y.transform    (t_te_f).reshape(-1, h, w).astype(np.float32)

    if val_ratio > 0.0 and t_tr_s.shape[0] > 1:
        p_tr_s, p_val_s, t_tr_s, t_val_s = train_test_split(
            p_tr_s, t_tr_s, test_size=val_ratio, random_state=split_seed)
    else:
        p_val_s, t_val_s = None, None

    return p_tr_s, t_tr_s, p_val_s, t_val_s, p_te_s, t_te, scaler_y


class ThermalDataset(Dataset):
    def __init__(self, params, temps):
        # params: (N, n_comp, d_per_comp), temps: (N, 200, 200)
        self.params = torch.from_numpy(params)
        self.temps  = torch.from_numpy(temps[:, None, :, :])  # (N,1,200,200)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.temps[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Training & evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, epochs, lr, weight_decay, log_every, device,
          early_stopping=False, patience=200, min_delta=0.0):
    mse  = nn.MSELoss()
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    train_losses, val_losses = [], []
    best_state = None
    best_val = float("inf")
    patience_count = 0
    stopped_epoch = epochs

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            running_v = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    running_v += mse(pred, yb).item() * xb.size(0)
            val_loss = running_v / len(val_loader.dataset)
            val_losses.append(val_loss)

            if early_stopping:
                if val_loss < (best_val - min_delta):
                    best_val = val_loss
                    patience_count = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_count += 1

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            lr_now = sched.get_last_lr()[0]
            if val_loader is not None:
                print(f"Epoch {ep:>5}/{epochs}  Train={train_loss:.6f}  "
                      f"Val={val_loss:.6f}  LR={lr_now:.2e}", flush=True)
            else:
                print(f"Epoch {ep:>5}/{epochs}  Train={train_loss:.6f}  LR={lr_now:.2e}", flush=True)

        if early_stopping and val_loader is not None and patience_count >= patience:
            stopped_epoch = ep
            print(f"Early stopping triggered at epoch {ep} (best val={best_val:.6f})", flush=True)
            break

    if early_stopping and val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model weights with val={best_val:.6f}", flush=True)

    info = {
        "best_val": float(best_val) if np.isfinite(best_val) else None,
        "stopped_epoch": int(stopped_epoch),
        "early_stopping": bool(early_stopping),
    }

    return train_losses, val_losses, info


def predict_all(model, params_scaled, device, batch_size=8):
    model.eval()
    preds = []
    ds = torch.from_numpy(params_scaled)
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            xb   = ds[i:i + batch_size].to(device)
            pred = model(xb).squeeze(1).cpu().numpy()   # (B, 200, 200)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def compute_r2(preds, true):
    """Per-sample R² on finite pixels."""
    r2s = []
    for i in range(len(preds)):
        p = preds[i].ravel()
        t = true[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2s.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2s = np.array(r2s)
    finite = np.isfinite(r2s)
    return r2s, float(np.mean(r2s[finite])) if finite.any() else np.nan


def get_param_breakdown(model):
    details = []
    total = 0
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        details.append({"module": name, "params": int(count)})
        total += count
    details.append({"module": "total", "params": int(total)})
    return details


def save_run_metadata(args, out_dir, model, n_params):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    payload = {
        "timestamp": timestamp,
        "baseline_hparams": BASELINE_HPARAMS,
        "run_args": vars(args),
        "model_params": {
            "total": int(n_params),
            "by_module": get_param_breakdown(model),
        },
    }

    run_path = os.path.join(out_dir, f"run_config_{timestamp}.json")
    latest_path = os.path.join(out_dir, "run_config_latest.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved run metadata: {run_path}", flush=True)
    return run_path


def save_summary(out_dir, summary):
    path = os.path.join(out_dir, "multi_run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved multi-run summary: {path}", flush=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (standardised)")
    plt.title("Training / Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_thermal_comparisons(preds, true, scaler_y, out_dir, n_samples=6):
    """For each of the first n_samples test cases: GT | Pred | |Error|."""
    n = min(n_samples, len(preds))
    for i in range(n):
        pred_img = scaler_y.inverse_transform(preds[i].reshape(1, -1)).reshape(200, 200)
        true_img = true[i].reshape(200, 200) if true[i].ndim == 1 else true[i]

        err = np.abs(pred_img - true_img)
        vmin = min(pred_img.min(), true_img.min())
        vmax = max(pred_img.max(), true_img.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ["Ground Truth (°C)", "Predicted (°C)", "Absolute Error (°C)"]
        imgs   = [true_img, pred_img, err]
        cmaps  = ["hot", "hot", "Reds"]

        for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
            im = ax.imshow(img, cmap=cmap,
                           vmin=(vmin if cmap == "hot" else None),
                           vmax=(vmax if cmap == "hot" else None),
                           origin="lower")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        r2 = r2_score(true_img.ravel(), pred_img.ravel())
        mae = np.mean(err)
        fig.suptitle(f"Test sample {i+1}   R²={r2:.4f}   MAE={mae:.3f} °C", fontsize=12)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sample_{i+1:02d}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Saved] {path}")


def plot_r2_bar(r2_vals, out_dir):
    finite = np.isfinite(r2_vals)
    r2_plot = np.where(finite, r2_vals, 0.0)
    n = len(r2_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Bar chart
    colours = ["steelblue" if v >= 0 else "tomato" for v in r2_plot]
    axes[0].bar(range(n), r2_plot, color=colours, edgecolor="k", linewidth=0.5)
    axes[0].axhline(np.nanmean(r2_vals), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean R²={np.nanmean(r2_vals):.4f}")
    axes[0].set_xlabel("Test sample index")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Per-sample R² on test set")
    axes[0].legend()

    # Histogram
    axes[1].hist(r2_vals[finite], bins=10, color="steelblue", edgecolor="k")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("Count")
    axes[1].set_title("R² Distribution")

    plt.tight_layout()
    path = os.path.join(out_dir, "r2_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_scatter(preds, true, scaler_y, out_dir, max_pts=50000):
    """Predicted vs true scatter for all test pixels."""
    all_pred, all_true = [], []
    for i in range(len(preds)):
        p = scaler_y.inverse_transform(preds[i].reshape(1, -1)).ravel()
        t = true[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        all_pred.append(p[mask])
        all_true.append(t[mask])
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # subsample for speed
    if len(all_pred) > max_pts:
        idx = np.random.choice(len(all_pred), max_pts, replace=False)
        all_pred, all_true = all_pred[idx], all_true[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_pred, s=2, alpha=0.3, rasterized=True)
    mn, mx = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Ground Truth (°C)")
    ax.set_ylabel("Predicted (°C)")
    ax.set_title("Predicted vs Ground Truth (all test pixels)")
    r2_all = r2_score(all_true, all_pred)
    ax.legend(title=f"Global R²={r2_all:.4f}")
    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_pred_vs_true.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Set Transformer + FNO for variable-component thermal prediction")

    # Data
    parser.add_argument("--train-params", default="thermal_analysis_output/training data/params_training.npy")
    parser.add_argument("--train-temps",  default="thermal_analysis_output/training data/temps_training.npy")
    parser.add_argument("--test-params",  default="thermal_analysis_output/test data/params_testing.npy")
    parser.add_argument("--test-temps",   default="thermal_analysis_output/test data/temps_testing.npy")

    # Component layout  (n_comp × d_per_comp must equal params feature dim)
    parser.add_argument("--n-components", type=int, default=4,
                        help="Number of components per sample (N in Set Transformer input)")
    parser.add_argument("--d-per-comp",   type=int, default=1,
                        help="Feature dimension per component")

    # Architecture
    parser.add_argument("--d-model",   type=int,   default=64)
    parser.add_argument("--num-heads", type=int,   default=4)
    parser.add_argument("--n-sab",     type=int,   default=1)
    parser.add_argument("--fno-ch",    type=int,   default=16)
    parser.add_argument("--fno-modes", type=int,   default=8)
    parser.add_argument("--n-fno",     type=int,   default=2)
    parser.add_argument("--dropout",   type=float, default=0.0)

    # Training
    parser.add_argument("--epochs",       type=int,   default=2000)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--log-every",    type=int,   default=50)
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping based on validation loss")
    parser.add_argument("--patience",     type=int,   default=200,
                        help="Early stopping patience in epochs")
    parser.add_argument("--min-delta",    type=float, default=0.0,
                        help="Minimum validation-loss improvement to reset patience")

    # Multi-run
    parser.add_argument("--n-runs",       type=int,   default=1,
                        help="Number of runs with different random split seeds")
    parser.add_argument("--seed-base",    type=int,   default=42,
                        help="Base random seed used for train/val split across runs")

    # Output
    parser.add_argument("--out-dir",    default="set_fno_results")
    parser.add_argument("--model-out",  default="set_fno_model.pth")
    parser.add_argument("--n-vis",      type=int, default=6,
                        help="Number of test samples to visualise")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    run_r2 = []
    run_details = []

    for run_idx in range(args.n_runs):
        split_seed = args.seed_base + run_idx
        run_tag = f"run_{run_idx + 1:02d}"
        run_dir = os.path.join(args.out_dir, run_tag) if args.n_runs > 1 else args.out_dir
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n===== {run_tag} / split_seed={split_seed} =====", flush=True)

        p_tr, t_tr, p_val, t_val, p_te, t_te_raw, scaler_y = load_data(
            args.train_params, args.train_temps,
            args.test_params,  args.test_temps,
            args.val_ratio, args.n_components, args.d_per_comp,
            split_seed=split_seed)

        print(f"Train samples  : {p_tr.shape[0]}", flush=True)
        print(f"Val   samples  : {p_val.shape[0] if p_val is not None else 0}", flush=True)
        print(f"Test  samples  : {p_te.shape[0]}", flush=True)
        print(f"Param shape    : {p_tr.shape}  (N_samples x n_comp x d_per_comp)", flush=True)

        train_ds = ThermalDataset(p_tr, t_tr)
        train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
        val_ld = None
        if p_val is not None:
            val_ds = ThermalDataset(p_val, t_val)
            val_ld = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, drop_last=False)

        model = SetFNOModel(
            d_in=args.d_per_comp,
            d_model=args.d_model,
            num_heads=args.num_heads,
            n_sab=args.n_sab,
            fno_ch=args.fno_ch,
            fno_modes=args.fno_modes,
            n_fno=args.n_fno,
            dropout=args.dropout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}", flush=True)
        save_run_metadata(args, run_dir, model, n_params)

        train_losses, val_losses, train_info = train(
            model, train_ld, val_ld,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            log_every=args.log_every, device=device,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta)

        model_out_path = args.model_out if args.n_runs == 1 else os.path.join(run_dir, args.model_out)
        ckpt = {
            "state_dict":      model.state_dict(),
            "scaler_y_mean":   scaler_y.mean_,
            "scaler_y_scale":  scaler_y.scale_,
            "args":            vars(args),
            "split_seed":      split_seed,
            "train_info":      train_info,
        }
        torch.save(ckpt, model_out_path)
        print(f"Model saved -> {model_out_path}", flush=True)

        preds_scaled = predict_all(model, p_te, device, args.batch_size)
        preds_inv = scaler_y.inverse_transform(preds_scaled.reshape(len(preds_scaled), -1)).reshape(-1, 200, 200)
        t_te_grid = t_te_raw.reshape(-1, 200, 200) if t_te_raw.ndim == 2 else t_te_raw

        r2_vals, r2_avg = compute_r2(preds_inv, t_te_grid)
        print(f"\nPer-sample R²: {np.round(r2_vals, 4)}", flush=True)
        print(f"Average  R²  : {r2_avg:.4f}", flush=True)

        print("\nGenerating visualisations ...", flush=True)
        plot_loss_curves(train_losses, val_losses, run_dir)
        plot_thermal_comparisons(preds_scaled, t_te_grid, scaler_y, run_dir, n_samples=args.n_vis)
        plot_r2_bar(r2_vals, run_dir)
        plot_scatter(preds_scaled, t_te_grid, scaler_y, run_dir)

        run_r2.append(r2_avg)
        run_details.append({
            "run": run_idx + 1,
            "split_seed": split_seed,
            "r2_avg": float(r2_avg),
            "train_info": train_info,
            "model_path": model_out_path,
            "result_dir": run_dir,
        })

    run_r2_arr = np.array(run_r2, dtype=np.float64)
    mean_r2 = float(np.nanmean(run_r2_arr))
    std_r2 = float(np.nanstd(run_r2_arr))
    print(f"\nMulti-run R² mean={mean_r2:.4f}, std={std_r2:.4f}, n_runs={args.n_runs}", flush=True)

    summary = {
        "n_runs": int(args.n_runs),
        "r2_mean": mean_r2,
        "r2_std": std_r2,
        "runs": run_details,
    }
    save_summary(args.out_dir, summary)
    print(f"All results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
