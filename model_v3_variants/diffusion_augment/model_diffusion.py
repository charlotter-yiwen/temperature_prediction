"""
model_diffusion.py
==================
Variant ④: Conditional Diffusion 数据增强

只用 1-5C 真实数据训练 DDPM 扩散模型，学习温度场分布。
然后用扩散模型生成 6-9C 的温度场样本，扩充主模型训练数据。

架构：标准 DDPM UNet + FiLM 条件注入
- encoder: (base_ch) → (base_ch*2) → (base_ch*4)
- decoder: (base_ch*4) → (base_ch*2) → (base_ch)
- 每层 num_res_blocks 个 ResBlock + attention
- FiLM 条件注入: gamma, beta from condition
"""

import os, sys, json, argparse
from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TP_DIR))

from models.set_fno_thermal import (
    load_count_sweep_data,
    compute_r2 as compute_r2_base,
    plot_loss_curves,
    plot_scatter,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Sinusoidal Embedding
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        embeddings = math.log(10000.0) / (half - 1)
        embeddings = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -embeddings)
        embeddings = t.float()[:, None] * embeddings[None, :]
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  ResBlock + Attention
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        gn_groups1 = min(8, in_ch)
        gn_groups2 = min(8, out_ch)
        self.norm1 = nn.GroupNorm(gn_groups1, in_ch)
        self.norm2 = nn.GroupNorm(gn_groups2, out_ch)
        self.dropout = nn.Dropout2d(dropout)
        # FiLM: emb → gamma + beta
        self.emb_proj = nn.Linear(emb_ch, out_ch * 2)
        # Skip projection when in_ch != out_ch
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        identity = self.skip_conv(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM conditioning
        gamma_beta = self.emb_proj(emb)  # (B, out_ch*2)
        gamma, beta = gamma_beta[:, :gamma_beta.size(1)//2], gamma_beta[:, gamma_beta.size(1)//2:]
        h = h * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + identity


class SelfAttention(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        gn_groups = min(8, ch)
        self.norm = nn.GroupNorm(gn_groups, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ══════════════════════════════════════════════════════════════════════════════
#  Conditional UNet
# ══════════════════════════════════════════════════════════════════════════════

class ConditionalUNet(nn.Module):
    """
    条件 UNet 去噪器。
    Encoder: levels × (ResBlock×2 + Downsample)
    Decoder: levels × (ResBlock×2 + Upsample + Skip)
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 128,
                 ch_mults: tuple = (1, 2, 4), num_res_blocks: int = 2,
                 cond_dim: int = 128, time_emb_dim: int = 256,
                 dropout: float = 0.1, use_attention: tuple = (True, True, True)):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.ch_mults = ch_mults
        n_levels = len(ch_mults)

        # Time embedding
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        # Condition projection: cond_dim → time_emb_dim
        self.cond_proj = nn.Linear(cond_dim, time_emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_chs = []  # (base_ch, base_ch*2, base_ch*4, ...)

        ch = base_ch
        for i in range(n_levels):
            self.encoder_chs.append(ch)
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(ch, ch, time_emb_dim, dropout))
                if use_attention[i]:
                    self.encoder_attn.append(SelfAttention(ch))
                else:
                    self.encoder_attn.append(nn.Identity())
            if i < n_levels - 1:
                next_ch = base_ch * ch_mults[i + 1]
                self.downsamples.append(nn.Conv2d(ch, next_ch, 3, stride=2, padding=1))
                ch = next_ch

        # Middle
        self.mid_ch = ch
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Decoder: 1 ResBlock per level, concat(skip, up_h) then process
        self.decoder = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(n_levels)):
            dec_ch = base_ch * ch_mults[i]
            skip_ch = self.encoder_chs[i]
            in_ch_first = dec_ch + skip_ch  # concat(skip, up_h) 后的输入通道
            self.decoder.append(ResBlock(in_ch_first, dec_ch, time_emb_dim, dropout))
            if use_attention[i]:
                self.decoder_attn.append(SelfAttention(dec_ch))
            else:
                self.decoder_attn.append(nn.Identity())
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(dec_ch, base_ch * ch_mults[i - 1], 2, stride=2))
            else:
                self.upsamples.append(nn.Identity())

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) 带噪图
        t: (B,) timestep (0 ~ T-1)
        cond: (B, cond_dim) 条件向量
        """
        # Time + condition embedding
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        c_emb = self.cond_proj(cond)  # (B, time_emb_dim)
        emb = t_emb + c_emb  # (B, time_emb_dim)

        # Initial
        h = self.init_conv(x)  # (B, base_ch, H, W)

        # Encoder: 1 skip per level (after all ResBlocks, before downsample)
        skips = []
        for i in range(len(self.ch_mults)):
            for _ in range(self.num_res_blocks):
                block_idx = i * self.num_res_blocks + _
                h = self.encoder[block_idx](h, emb)
                h = self.encoder_attn[block_idx](h)
            skips.append(h)  # store after all ResBlocks at this level
            if i < len(self.ch_mults) - 1:
                h = self.downsamples[i](h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder: iterate forward (decoder[0]=level2, decoder[1]=level1, decoder[2]=level0)
        # skip.pop() order from reverse: skip[5]=level2(skip_2), skip[3]=level1(skip_1), skip[1]=level0(skip_0)
        # Decoder 0 (level 2): middle output + skip_2 (no upsample first)
        # Decoder 1 (level 1): upsample then + skip_1
        # Decoder 2 (level 0): upsample then + skip_0
        n_levels = len(self.ch_mults)
        # Upsample indices: level 2->1: upsample[0], level 1->0: upsample[1]
        # decoder[0] (i=2, coarsest): no upsample first
        # decoder[1] (i=1, mid): upsample[0] then concat
        # decoder[2] (i=0, finest): upsample[1] then concat
        for dec_idx in range(n_levels):
            skip_h = skips.pop()  # pops in reverse: level2_skip, level1_skip, level0_skip
            if dec_idx > 0:  # not the first (coarsest) block
                up_idx = dec_idx - 1  # upsample[0] for dec_idx=1, upsample[1] for dec_idx=2
                h = self.upsamples[up_idx](h)
            h = torch.cat([h, skip_h], dim=1)
            h = self.decoder[dec_idx](h, emb)
            h = self.decoder_attn[dec_idx](h)

        return self.out_conv(h)


# ══════════════════════════════════════════════════════════════════════════════
#  Diffusion Utilities
# ══════════════════════════════════════════════════════════════════════════════

def make_beta_schedule(n_timestep: int = 1000,
                       linear_start: float = 1e-4, linear_end: float = 2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep)
    return betas


def p_losses(denoise_model, x_start: torch.Tensor, t: torch.Tensor,
             cond: torch.Tensor, betas: torch.Tensor, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    T = len(betas)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    t_safe = t.clamp(0, T - 1)
    a = alphas_cumprod[t_safe].view(-1, 1, 1, 1)
    x_noisy = a.sqrt() * x_start + (1 - a).sqrt() * noise
    predicted = denoise_model(x_noisy, t, cond)
    return F.mse_loss(predicted, noise)


@torch.no_grad()
def ddim_sample(model, cond: torch.Tensor, shape: tuple,
                betas: torch.Tensor, n_steps: int = 50, device: str = "cuda"):
    """DDIM 采样"""
    model.eval()
    T = len(betas)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    x = torch.randn(shape, device=device)
    times = torch.linspace(0, T - 1, n_steps, dtype=torch.long, device=device)

    for i, t_i in enumerate(times):
        t_tensor = torch.full((shape[0],), T - 1 - t_i, device=device, dtype=torch.long)
        predicted = model(x, t_tensor, cond)

        alpha_now = alphas_cumprod[T - 1 - t_i]
        if i < n_steps - 1:
            t_next = T - 1 - times[i + 1]
            alpha_next = alphas_cumprod[t_next]
            pred_x0 = (x - predicted * (1 - alpha_now).sqrt()) / alpha_now.sqrt()
            x = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * predicted
        else:
            pred_x0 = (x - predicted * (1 - alpha_now).sqrt()) / alpha_now.sqrt()
            x = pred_x0

    return x.clamp(-3, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ThermalDiffusionDataset(Dataset):
    def __init__(self, temps, params, total_power):
        # temps: (N, 100, 100) 原始温度
        self.temps = torch.from_numpy(temps).float().unsqueeze(1)  # (N, 1, 100, 100)
        max_comp = params.shape[1]
        self.cond = torch.zeros(params.shape[0], max_comp * 3 + 1, dtype=torch.float32)
        self.cond[:, :-1] = torch.from_numpy(params.reshape(params.shape[0], -1)).float()
        self.cond[:, -1] = torch.from_numpy(total_power).float()

    def __len__(self):
        return self.temps.shape[0]

    def __getitem__(self, idx):
        return self.temps[idx], self.cond[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_diffusion(model, train_loader, betas, epochs, lr, device, log_every=100, out_dir="."):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    losses = []

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x0, cond in train_loader:
            x0 = x0.to(device)
            cond = cond.to(device)
            t = torch.randint(0, len(betas), (x0.shape[0],), device=device)
            loss = p_losses(model, x0, t, cond, betas)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x0.shape[0]
        scheduler.step()
        losses.append(epoch_loss / len(train_loader.dataset))

        if ep % log_every == 0 or ep == 1:
            print(f"Ep {ep:>5}/{epochs} Loss={losses[-1]:.6f}", flush=True)

        if ep % 2000 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"diffusion_ckpt_ep{ep}.pt"))

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-sweep-params", required=True)
    parser.add_argument("--count-sweep-temps", required=True)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--d-per-comp", type=int, default=3)
    parser.add_argument("--physics-norm", action="store_true")
    parser.add_argument("--t-ambient", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-timesteps", type=int, default=1000)
    parser.add_argument("--n-sample-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--base-ch", type=int, default=128)
    parser.add_argument("--ch-mults", type=str, default="1,2,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--time-emb-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out-dir", default="results_diffusion")
    parser.add_argument("--model-out", default="diffusion.pt")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Variant 4: Conditional Diffusion (Full UNet)", flush=True)
    print(f"{'='*60}", flush=True)

    # 加载数据
    (p_train, t_tr_s, p_val, t_val_s,
     p_test, t_te_raw, scaler_y, grid_size, norm_info) = load_count_sweep_data(
        args.count_sweep_params, args.count_sweep_temps,
        max_components=args.n_components, d_per_comp=args.d_per_comp,
        test_ratio=0.2, val_ratio=0.0,
        split_seed=42,
        physics_norm=args.physics_norm, T_ambient=args.t_ambient)

    tp_train = np.nansum(p_train[:, :, 2], axis=1).astype(np.float32)

    # 归一化
    t_tr_grid = t_tr_s.reshape(-1, grid_size, grid_size)
    preds_inv = t_tr_grid * scaler_y.scale_ + scaler_y.mean_
    if args.physics_norm:
        preds_inv = preds_inv * tp_train[:, None, None] + args.t_ambient

    print(f"Train samples: {p_train.shape[0]}", flush=True)

    betas = make_beta_schedule(args.n_timesteps).to(device)

    # 条件维度
    max_comp = args.n_components
    cond_dim = max_comp * 3 + 1

    # UNet
    ch_mults = tuple(int(x) for x in args.ch_mults.split(","))
    model = ConditionalUNet(
        in_ch=1,
        base_ch=args.base_ch,
        ch_mults=ch_mults,
        num_res_blocks=args.num_res_blocks,
        cond_dim=cond_dim,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Diffusion model params: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    train_ds = ThermalDiffusionDataset(preds_inv, p_train, tp_train)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    losses = train_diffusion(model, train_ld, betas, args.epochs, args.lr,
                           device, log_every=args.log_every, out_dir=args.out_dir)

    torch.save(model.state_dict(), os.path.join(args.out_dir, args.model_out))
    print(f"Model saved -> {args.out_dir}/{args.model_out}", flush=True)

    # 测试采样
    print("\nSampling test...", flush=True)
    model.eval()
    with torch.no_grad():
        test_cond = torch.zeros(1, cond_dim, dtype=torch.float32, device=device)
        test_cond[0, :6] = torch.tensor([0.5, 0.5, 2.5, 0.6, 0.4, 2.5], device=device)
        test_cond[0, -1] = 10.0
        sample = ddim_sample(model, test_cond, (1, 1, 100, 100),
                           betas, n_steps=args.n_sample_steps, device=device)
        print(f"Generated sample range: [{sample.min():.3f}, {sample.max():.3f}]", flush=True)

    print(f"All results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
