"""
train_setfno_v3.py  (SetFNO V3 架构 — 修正版)
===============================================
改进点（相比原版）：
  1. Heat Source Map (2通道)
       ch0 = H_norm = Σ P_k*Gauss(d_k) / P_total   (形状归一化)
       ch1 = P_total_norm = P_total / P_TOTAL_REF   (量级信息)
     2通道合在一起让模型同时知道「热源在哪」和「总功率多大」

  2. 物理归一化输出
       预测热阻 θ = (T-T_amb)/P_total  →  T = θ*P_total + T_amb
       使不同功率量级的目标量纲一致，改善外推泛化

  3. Pairwise 交互编码（修正版）
       [wp, min_dist, total_p/P_TOTAL_REF, n_active/M]
       total_p 用训练集最大*总*功率归一化，保证测试时尺度不爆炸

  4. PDE Loss（修正版）
       Laplacian ≈ 0 只在非热源区约束
       热源区用 heatmap > threshold 确定，dx 用物理单位 (m)

  5. BC Loss（修正版）
       用真实物理尺寸 dx_m = board/grid (m) 推导 c_adj
       BC 对 θ 空间成立：k*(θ_adj-θ_edge)/dx = h*θ_edge  → c_adj = k/(k+h*dx)

  6. DeepOHeat 风格 Residual Refinement（修正版）
       不是可训练 CNN corrector，而是：
         a. 模型预测粗解 θ_coarse
         b. 计算 PDE residual r = Lap(θ_coarse) - Q_norm
         c. 在 residual 大的区域跑少量 SOR 迭代修正
       仅在推理时可选启用（训练时关闭以节省时间）

数据格式（training_data/）：
  params_count_sweep.npy  : (N, 5, 3)  [x_mm, y_mm, power_W]  NaN=缺失
  temps_count_sweep.npy   : (N, 10000) 展平的 100x100 温度场 (°C)

用法：
  # 训练
  python my_scripts/train_setfno_v3.py \
      --params training_data/params_count_sweep.npy \
      --temps  training_data/temps_count_sweep.npy  \
      --epochs 2000 --batch-size 32 --lr 5e-5 \
      --lambda-bc 0.005 --lambda-pde 0.001 \
      --early-stopping --patience 200 \
      --out-dir my_scripts/results_v3

  # 泛化测试（6-9组件），加 SOR refinement
  python my_scripts/train_setfno_v3.py --test-only --sor-refine \
      --params training_data/gen_test_params.npy \
      --temps  training_data/gen_test_temps.npy  \
      --model-path my_scripts/results_v3/setfno_v3_best.pth \
      --out-dir my_scripts/results_v3/gen_test
"""

import os
import sys
import argparse
import json
import shutil
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

# ──────────────────────────────────────────────────────────────────────────────
# 超参数默认值
# ──────────────────────────────────────────────────────────────────────────────
GRID          = 100     # 温度场分辨率
BOARD_MM      = 100.0   # PCB 尺寸 (mm)
T_AMB         = 25.0    # 环境温度 (°C)
MAX_COMP      = 5       # 最大组件数（训练集）
MAX_COMP_REF  = 9       # 固定参考值，用于归一化 n_active（训练/测试一致）
SIGMA_MM      = 6.0     # Gaussian blob 半径 (mm)
K_FR4         = 0.35    # FR-4 热导率 (W/m/K)
H_CONV        = 30.0    # 对流系数 (W/m²/K)
# dx_m = board_mm/1000/grid = 0.001 m
DX_M          = BOARD_MM / 1000.0 / GRID   # 物理网格间距 (m)
# BC 系数: c_adj = k/(k + h*dx)  (在 θ 空间 Robin BC)
C_ADJ         = K_FR4 / (K_FR4 + H_CONV * DX_M)

# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def make_heatmap(params_np: np.ndarray, p_total_ref: float,
                 grid: int = GRID, board_mm: float = BOARD_MM,
                 sigma_mm: float = SIGMA_MM) -> np.ndarray:
    """
    将 (N, max_comp, 3) 组件参数转为 (N, 2, grid, grid) 热源图。

    ch0: H_shape = Σ P_k*Gauss / P_total   形状归一化（值域 ~[0,1]，与功率无关）
    ch1: P_scale = P_total / p_total_ref    功率量级（让模型知道总功率大小）

    两通道分工：ch0 告诉模型「热源在哪」，ch1 告诉模型「功率有多大」。
    测试时组件数/功率不同，两通道各自保持物理意义不变。

    params_np    : (N, max_comp, 3)  [x_mm, y_mm, power_W]  0=缺失
    p_total_ref  : 训练集最大总功率，用于归一化 ch1
    返回         : (N, 2, grid, grid) float32
    """
    N, M, _ = params_np.shape
    sigma_norm = sigma_mm / board_mm

    lin = np.linspace(0.5 / grid, 1 - 0.5 / grid, grid, dtype=np.float32)
    gx, gy = np.meshgrid(lin, lin, indexing='ij')  # (grid, grid)

    ch0 = np.zeros((N, grid, grid), dtype=np.float32)  # shape
    ch1 = np.zeros((N, 1,    1),    dtype=np.float32)  # scale (broadcast)

    for n in range(N):
        total_p = 0.0
        for k in range(M):
            x_mm, y_mm, p = params_np[n, k]
            if p == 0.0:
                continue
            x_n = x_mm / board_mm
            y_n = y_mm / board_mm
            dist_sq = (gx - x_n) ** 2 + (gy - y_n) ** 2
            ch0[n] += p * np.exp(-dist_sq / (2 * sigma_norm ** 2))
            total_p += p
        if total_p > 0:
            ch0[n] /= total_p            # ch0: 形状，总积分=1
        ch1[n, 0, 0] = total_p / max(p_total_ref, 1e-6)   # ch1: 量级

    ch1_map = np.broadcast_to(ch1, (N, grid, grid)).copy()
    return np.stack([ch0, ch1_map], axis=1)   # (N, 2, grid, grid)


def make_pairwise_features(params_np: np.ndarray,
                            p_total_ref: float,
                            board_mm: float = BOARD_MM) -> np.ndarray:
    """
    为每个组件添加 pairwise 交互特征（修正版）：
      [0] weighted_power  = Σ_{j≠i} P_j/d_ij²  / (P_single_ref/1²)
                            热影响强度，用单组件参考值归一化
      [1] min_dist_norm   = min_j(d_ij) / board_mm
      [2] total_power_norm= P_total / p_total_ref
                            用训练集最大「总」功率归一化，泛化时不会爆炸
      [3] n_active_norm   = 有效组件数 / MAX_COMP

    params_np   : (N, max_comp, 3)  [x_mm, y_mm, power_W]  0=缺失
    p_total_ref : 训练集最大总功率（保存到 norm_info，测试时复用）
    返回        : (N, max_comp, 4)
    """
    N, M, _ = params_np.shape
    feat = np.zeros((N, M, 4), dtype=np.float32)

    # 单组件功率参考值（用于归一化 weighted_power）
    valid_p = params_np[:, :, 2][params_np[:, :, 2] > 0]
    p_single_ref = float(valid_p.max()) if len(valid_p) else 1.0
    wp_ref = p_single_ref / (1.0 ** 2)   # 1mm 距离时的参考值

    for n in range(N):
        active = [k for k in range(M) if params_np[n, k, 2] > 0]
        n_act  = len(active)
        total_p = params_np[n, :, 2].sum()

        for i in active:
            xi, yi, pi = params_np[n, i]
            wp    = 0.0
            min_d = board_mm   # 默认：board 宽度（无邻居时）
            for j in active:
                if j == i:
                    continue
                xj, yj, pj = params_np[n, j]
                d      = max(np.sqrt((xi-xj)**2 + (yi-yj)**2), 1.0)
                wp    += pj / (d ** 2)
                min_d  = min(min_d, d)
            feat[n, i, 0] = wp / wp_ref                        # 热影响强度
            feat[n, i, 1] = min_d / board_mm                   # 最近邻归一化距离
            feat[n, i, 2] = total_p / max(p_total_ref, 1e-6)   # 总功率（与ch1一致）
            feat[n, i, 3] = n_act / MAX_COMP_REF               # 固定参考值，训练/测试一致

    return feat  # (N, max_comp, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 模型组件
# ──────────────────────────────────────────────────────────────────────────────

class MAB(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, Q, K):
        out, _ = self.attn(Q, K, K)
        x = self.norm1(Q + out)
        return self.norm2(x + self.ff(x))


class SAB(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.mab = MAB(d, heads, dropout)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, d, heads, k=1, dropout=0.0):
        super().__init__()
        self.S   = nn.Parameter(torch.randn(1, k, d))
        self.rff = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.mab = MAB(d, heads, dropout)

    def forward(self, X):
        S = self.S.expand(X.size(0), -1, -1)
        return self.mab(S, self.rff(X))


class SetTransformerEncoder(nn.Module):
    """d_in=7 (x,y,p + 4 pairwise) → (B, d_model)"""
    def __init__(self, d_in=7, d_model=256, num_heads=8, n_sab=4, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.sabs = nn.ModuleList([SAB(d_model, num_heads, dropout) for _ in range(n_sab)])
        self.pma  = PMA(d_model, num_heads, k=1, dropout=dropout)
        self.out  = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, X):
        """X: (B, N, d_in) → (B, d_model)"""
        h = self.proj(X)
        for sab in self.sabs:
            h = sab(h)
        h = self.pma(h).squeeze(1)
        return self.out(h)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        scale = 1.0 / (in_ch * out_ch)
        self.W1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        self.W2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        self.modes = modes
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.modes
        xft = torch.fft.rfft2(x)
        half = W // 2 + 1
        out = torch.zeros(B, self.out_ch, H, half, dtype=torch.cfloat, device=x.device)
        out[:, :, :m,  :m] = torch.einsum('bixy,ioxy->boxy', xft[:, :, :m,  :m], self.W1)
        out[:, :, -m:, :m] = torch.einsum('bixy,ioxy->boxy', xft[:, :, -m:, :m], self.W2)
        return torch.fft.irfft2(out, s=(H, W))


class FNOBlock(nn.Module):
    def __init__(self, ch, modes):
        super().__init__()
        self.spec   = SpectralConv2d(ch, ch, modes)
        self.bypass = nn.Conv2d(ch, ch, 1)
        self.norm   = nn.InstanceNorm2d(ch, affine=True)

    def forward(self, x):
        return F.gelu(self.norm(self.spec(x) + self.bypass(x)))


class SetFNOv3(nn.Module):
    """
    V3 架构：
      Branch A: SetTransformer (pairwise features) → global cond (B, d_model)
      Branch B: HeatMap (B, 1, 100, 100) + xy_grid (B, 2, 100, 100) → (B, 3, 100, 100)
      Fusion:   FNO 处理空间输入 + 加入 global cond (通过 AdaIN)
      Output:   热阻场 θ_coarse (B, 1, 100, 100)
      Corrector: 小 CNN (θ_coarse + heatmap) → δθ (B, 1, 100, 100)
      Final:    θ = θ_coarse + δθ  →  T = θ * P_total + T_amb
    """
    def __init__(self, d_model=256, num_heads=8, n_sab=4,
                 fno_ch=64, fno_modes=24, n_fno=6, dropout=0.0,
                 grid=100, use_corrector=True):
        super().__init__()
        self.grid = grid
        self.use_corrector = use_corrector

        # ── Branch A: SetTransformer（输入7维：3原始+4pairwise）
        self.set_enc = SetTransformerEncoder(
            d_in=7, d_model=d_model, num_heads=num_heads,
            n_sab=n_sab, dropout=dropout)

        # ── Branch B: 空间输入嵌入 (4通道 → fno_ch通道)
        # 输入: heatmap_shape(1) + p_scale(1) + x_grid(1) + y_grid(1) = 4通道
        self.spatial_embed = nn.Sequential(
            nn.Conv2d(4, fno_ch, 1),
            nn.GELU(),
            nn.Conv2d(fno_ch, fno_ch, 1))

        # ── 坐标网格（固定，不训练）
        lin = torch.linspace(0.5/grid, 1-0.5/grid, grid)
        gx, gy = torch.meshgrid(lin, lin, indexing='ij')
        self.register_buffer('grid_x', gx.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)
        self.register_buffer('grid_y', gy.unsqueeze(0).unsqueeze(0))

        # ── AdaIN 调制：用 global cond 调制 FNO 特征
        self.adain_layers = nn.ModuleList([
            nn.Linear(d_model, fno_ch * 2) for _ in range(n_fno)])

        # ── FNO blocks
        self.fno_blocks = nn.ModuleList(
            [FNOBlock(fno_ch, fno_modes) for _ in range(n_fno)])

        # ── 输出头 → 热阻场 θ (1通道)
        self.out_head = nn.Sequential(
            nn.Conv2d(fno_ch, fno_ch // 2, 1), nn.GELU(),
            nn.Conv2d(fno_ch // 2, 1, 1))

        # use_corrector 保留接口，但 SOR refinement 在推理阶段外部调用
        # 模型本身不含 corrector 参数（DeepOHeat-v1 hybrid 方式）
        self.use_corrector = use_corrector

    def forward(self, params_7d, heatmap):
        """
        params_7d : (B, N, 7)  [x,y,p,wp,mind,totalp,nact]
        heatmap   : (B, 2, H, W)  ch0=形状, ch1=功率量级
        返回      : θ (B, 1, H, W)  热阻场
        """
        B = params_7d.size(0)

        # Branch A: global condition
        cond = self.set_enc(params_7d)  # (B, d_model)

        # Branch B: 空间输入 (heatmap 2ch + xy grid 2ch = 4ch)
        gx = self.grid_x.expand(B, -1, -1, -1)
        gy = self.grid_y.expand(B, -1, -1, -1)
        spatial = torch.cat([heatmap, gx, gy], dim=1)  # (B, 4, H, W)
        h = self.spatial_embed(spatial)                 # (B, fno_ch, H, W)

        # FNO + AdaIN 调制
        for blk, adain_lin in zip(self.fno_blocks, self.adain_layers):
            h = blk(h)
            gamma_beta = adain_lin(cond)               # (B, fno_ch*2)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.view(B, -1, 1, 1) + 1.0     # 初始化为1（恒等映射）
            beta  = beta.view(B, -1, 1, 1)
            h = gamma * h + beta

        return self.out_head(h)   # (B, 1, H, W) — 热阻场 θ


# ──────────────────────────────────────────────────────────────────────────────
# Physics Loss
# ──────────────────────────────────────────────────────────────────────────────

def physics_loss(theta_pred, heatmap_shape,
                 k_fr4=K_FR4, h_conv=H_CONV, dx_m=DX_M, c_adj=C_ADJ,
                 lambda_pde=0.001, lambda_bc=0.005,
                 source_rel_threshold=0.1):
    """
    theta_pred          : (B, 1, H, W) 预测热阻场
    heatmap_shape       : (B, 2, H, W) heatmap ch0（形状通道）
    source_rel_threshold: 相对阈值。热源掋码定义为
                            h_shape > source_rel_threshold * h_shape.max()
                          比固定 0.05 更稳定：
                          1个元件时 Gaussian 峰值高，阈值也高；
                          9个元件分散后峰值低，阈值成比例下降，始终覆盖各热源。

    BC Loss：θ_edge = c_adj * θ_adj， c_adj = k/(k+h*dx)
    PDE Loss：Laplacian(θ) ≈ 0 只在非热源 interior 点

    返回: (λ_bc * L_bc, λ_pde * L_pde)
    """
    T = theta_pred.squeeze(1)  # (B, H, W)

    # ─── BC Loss ───
    bc_top    = ((T[:, 0, :]  - c_adj * T[:, 1, :])  ** 2).mean()
    bc_bottom = ((T[:, -1, :] - c_adj * T[:, -2, :]) ** 2).mean()
    bc_left   = ((T[:, :, 0]  - c_adj * T[:, :, 1])  ** 2).mean()
    bc_right  = ((T[:, :, -1] - c_adj * T[:, :, -2]) ** 2).mean()
    L_bc = (bc_top + bc_bottom + bc_left + bc_right) / 4.0

    # ─── PDE Loss ───
    lap_kernel = torch.tensor([[[[0., 1., 0.],
                                  [1., -4., 1.],
                                  [0., 1., 0.]]]], dtype=T.dtype, device=T.device)
    lap = F.conv2d(T.unsqueeze(1), lap_kernel, padding=1).squeeze(1)  # (B,H,W)

    # 热源掩码：相对阈值，对任意元件数/功率分布都稳定
    h_shape = heatmap_shape[:, 0, :, :]                          # (B, H, W)
    h_max   = h_shape.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)  # (B,1,1)
    is_source = (h_shape > source_rel_threshold * h_max).float()

    # interior mask
    interior = torch.ones_like(lap)
    interior[:, 0, :] = 0; interior[:, -1, :] = 0
    interior[:, :, 0] = 0; interior[:, :, -1] = 0

    non_source_interior = interior * (1.0 - is_source)
    n_pts = non_source_interior.sum().clamp(min=1.0)
    L_pde = (lap ** 2 * non_source_interior).sum() / n_pts

    return lambda_bc * L_bc, lambda_pde * L_pde


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ThermalDatasetV3(Dataset):
    """
    返回 (params_7d, heatmap, theta_target, total_power_normalized)
    """
    def __init__(self, params_raw, temps_raw, t_amb=T_AMB, board_mm=BOARD_MM,
                 sigma_mm=SIGMA_MM, max_power_ref=None,
                 p_total_ref=None, theta_mean=None, theta_std=None):
        """
        params_raw   : (N, max_comp, 3) [x_mm, y_mm, power_W] 0=缺失
        temps_raw    : (N, 100, 100)   温度场 (°C)
        p_total_ref  : 训练集实际最大总功率；None→从本数据估算（仅 train 时）
        theta_mean/std: train dataset 的 scaler；None→从本数据计算（仅 train 时）
        """
        N, M, _ = params_raw.shape
        self.t_amb = t_amb

        # 先将 NaN 替换为 0（缺失组件功率视为0）
        params_raw = np.nan_to_num(params_raw, nan=0.0)

        # 总功率 (N,)
        total_power = params_raw[:, :, 2].sum(axis=1)  # (N,)
        total_power = np.maximum(total_power, 0.1)
        self.total_power = total_power.astype(np.float32)

        # 热阻场 θ = (T - T_amb) / P_total
        theta = (temps_raw - t_amb) / total_power[:, None, None]  # (N, 100, 100)

        # 参考最大功率（用于归一化 pairwise 特征）
        valid_powers = params_raw[:, :, 2][params_raw[:, :, 2] > 0]
        self.max_power_ref = float(valid_powers.max()) if len(valid_powers) else 1.0
        if max_power_ref is not None:
            self.max_power_ref = max_power_ref

        # 坐标归一化 (mm -> 0~1)，功率归一化
        params_norm = params_raw.copy()
        params_norm[:, :, 0] /= board_mm
        params_norm[:, :, 1] /= board_mm
        params_norm[:, :, 2] /= self.max_power_ref
        params_norm = np.nan_to_num(params_norm, nan=0.0)

        # p_total_ref：训练集实际最大总功率（从外部传入保证 val/test 一致）
        if p_total_ref is not None:
            self.p_total_ref = float(p_total_ref)
        else:
            # 仅 train dataset 时从数据估算
            self.p_total_ref = float(
                np.nan_to_num(params_raw, nan=0.0)[:, :, 2].sum(axis=1).max())
        self.p_total_ref = max(self.p_total_ref, 1e-6)

        # Pairwise features（用实际训练集最大总功率归一化）
        pw_feat = make_pairwise_features(
            np.nan_to_num(params_raw, nan=0.0),
            p_total_ref=self.p_total_ref,
            board_mm=board_mm)  # (N, M, 4)

        # 拼接 7d 输入
        params_7d = np.concatenate([params_norm, pw_feat], axis=-1)  # (N, M, 7)

        # Heat Source Map（2通道：形状 + 功率量级，用同一个 p_total_ref）
        hmaps = make_heatmap(
            np.nan_to_num(params_raw, nan=0.0),
            p_total_ref=self.p_total_ref,
            grid=GRID, board_mm=board_mm, sigma_mm=sigma_mm)  # (N, 2, H, W)

        # θ 的 StandardScaler（val/test 必须用 train 的 scaler）
        theta_flat = theta.reshape(N, -1)
        if theta_mean is not None and theta_std is not None:
            self.theta_mean = float(theta_mean)
            self.theta_std  = float(theta_std)
        else:
            self.theta_mean = float(theta_flat.mean())
            self.theta_std  = float(theta_flat.std()) + 1e-8
        theta_scaled = (theta - self.theta_mean) / self.theta_std

        self.params_7d     = torch.tensor(params_7d, dtype=torch.float32)
        self.hmaps         = torch.tensor(hmaps, dtype=torch.float32)
        self.theta_scaled  = torch.tensor(theta_scaled[:, None, :, :], dtype=torch.float32)
        self.theta_raw     = torch.tensor(theta[:, None, :, :], dtype=torch.float32)
        self.temps_raw     = torch.tensor(temps_raw[:, None, :, :], dtype=torch.float32)

    def __len__(self):
        return self.params_7d.size(0)

    def __getitem__(self, idx):
        return (self.params_7d[idx], self.hmaps[idx],
                self.theta_scaled[idx], self.total_power[idx],
                self.temps_raw[idx])


# ──────────────────────────────────────────────────────────────────────────────
# 辅助：反归一化预测
# ──────────────────────────────────────────────────────────────────────────────

def denorm_and_restore_T(theta_scaled_pred, total_power_w,
                          theta_mean, theta_std, t_amb=T_AMB):
    """
    theta_scaled_pred : (B, 1, H, W) 网络输出
    total_power_w     : (B,) 原始总功率 (W)
    返回 T_pred (B, H, W) 单位 °C
    """
    theta = theta_scaled_pred.squeeze(1) * theta_std + theta_mean  # (B, H, W)
    T = theta * total_power_w[:, None, None] + t_amb
    return T


# ──────────────────────────────────────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────────────────────────────────────

def power_scale_augment(params_raw, temps_raw, n_copies, scale_min, scale_max,
                        t_amb=T_AMB, seed=42, include_original=True):
    """
    功率缩放数据增强（物理线性性）：
      new_T = α*(T - T_amb) + T_amb
      new_params[:,:,2] *= α
      θ = (T-T_amb)/P 不变 → 目标不变，输入功率量级多样化

    n_copies  : 每条原始样本增强几份（不含原始）
    scale_min : 随机缩放系数下界（如 1.2）
    scale_max : 随机缩放系数上界（如 2.5）
    """
    if n_copies <= 0 and not include_original:
        raise ValueError('power_scale_augment: n_copies must be > 0 when include_original=False')

    rng = np.random.default_rng(seed)
    aug_p_list = [params_raw] if include_original else []
    aug_t_list = [temps_raw] if include_original else []
    for _ in range(n_copies):
        alpha = rng.uniform(scale_min, scale_max, size=(len(params_raw),)).astype(np.float32)
        new_p = params_raw.copy()
        new_p[:, :, 2] *= alpha[:, None]           # (N, M) scale power
        new_t = alpha[:, None, None] * (temps_raw - t_amb) + t_amb  # (N, H, W)
        aug_p_list.append(new_p)
        aug_t_list.append(new_t)
    return np.concatenate(aug_p_list, axis=0), np.concatenate(aug_t_list, axis=0)


def build_model(args, device):
    return SetFNOv3(
        d_model=args.d_model, num_heads=args.num_heads, n_sab=args.n_sab,
        fno_ch=args.fno_ch, fno_modes=args.fno_modes, n_fno=args.n_fno,
        dropout=args.dropout, grid=GRID,
        use_corrector=not args.no_corrector
    ).to(device)


def count_parameters(model, trainable_only=False):
    params = model.parameters() if not trainable_only else [p for p in model.parameters() if p.requires_grad]
    return sum(p.numel() for p in params)


def freeze_backbone_for_phase2(model):
    for p in model.parameters():
        p.requires_grad = False

    # Phase 2: 只微调最后几层（AdaIN + 输出头）
    for module in [model.adain_layers, model.out_head]:
        for p in module.parameters():
            p.requires_grad = True


def save_norm_info(norm_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'norm_info.json'), 'w') as f:
        json.dump(norm_info, f, indent=2)


def run_training_phase(model, dl_tr, dl_val, ds_tr, device, norm_info, out_dir,
                       epochs, lr, weight_decay,
                       lambda_bc, lambda_pde,
                       early_stopping, patience, min_delta,
                       log_every, phase_name, ckpt_args):
    os.makedirs(out_dir, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n=== {phase_name} ===")
    print(f"Trainable parameters: {n_trainable:,}")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    patience_cnt  = 0
    history = {'train': [], 'val': [], 'lr': []}
    best_path = os.path.join(out_dir, 'setfno_v3_best.pth')
    final_path = os.path.join(out_dir, 'setfno_v3_final.pth')

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for params_7d, hmaps, theta_sc, total_p, temps_r in dl_tr:
            params_7d = params_7d.to(device)
            hmaps     = hmaps.to(device)
            theta_sc  = theta_sc.to(device)
            total_p   = total_p.to(device)

            optimizer.zero_grad()
            theta_pred_sc = model(params_7d, hmaps)

            loss_data = F.mse_loss(theta_pred_sc, theta_sc)
            theta_pred_raw = (theta_pred_sc * ds_tr.theta_std + ds_tr.theta_mean)
            L_bc, L_pde = physics_loss(
                theta_pred_raw, hmaps,
                lambda_bc=lambda_bc,
                lambda_pde=lambda_pde)

            loss = loss_data + L_bc + L_pde
            if not torch.isfinite(loss):
                print(f"  [warn] NaN/Inf loss skipped (data={loss_data.item():.4f} bc={L_bc.item():.4f} pde={L_pde.item():.4f})")
                optimizer.zero_grad()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(params_7d)

        tr_loss /= len(ds_tr)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for params_7d, hmaps, theta_sc, _, _ in dl_val:
                params_7d = params_7d.to(device)
                hmaps     = hmaps.to(device)
                theta_sc  = theta_sc.to(device)
                pred = model(params_7d, hmaps)
                val_loss += F.mse_loss(pred, theta_sc).item() * len(params_7d)
        val_loss /= len(dl_val.dataset)

        scheduler.step()
        history['train'].append(tr_loss)
        history['val'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])

        if epoch % log_every == 0:
            print(f"[{phase_name} {epoch:4d}/{epochs}] train={tr_loss:.5f}  val={val_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                'model': model.state_dict(),
                'norm_info': norm_info,
                'args': ckpt_args,
                'phase': phase_name,
            }, best_path)
        else:
            patience_cnt += 1
            if early_stopping and patience_cnt >= patience:
                print(f"{phase_name} early stopping at epoch {epoch} (patience={patience})")
                break

    torch.save({
        'model': model.state_dict(),
        'norm_info': norm_info,
        'args': ckpt_args,
        'phase': phase_name,
    }, final_path)

    _plot_loss(history, os.path.join(out_dir, 'loss_curves.png'))
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_path': best_path,
        'final_path': final_path,
        'epochs_ran': len(history['train']),
        'trainable_params': n_trainable,
    }


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载数据
    params_raw = np.load(args.params)   # (N, 5, 3) or (N, 15) → reshape
    temps_raw  = np.load(args.temps)    # (N, 10000) or (N, 100, 100)

    if params_raw.ndim == 2:
        N = params_raw.shape[0]
        params_raw = params_raw.reshape(N, -1, 3)
    if temps_raw.ndim == 2:
        N = temps_raw.shape[0]
        gs = int(round(np.sqrt(temps_raw.shape[1])))
        temps_raw = temps_raw.reshape(N, gs, gs)

    print(f"Loaded: params={params_raw.shape}, temps={temps_raw.shape}")
    print(f"Temps range: {temps_raw.min():.2f} - {temps_raw.max():.2f} °C")

    # 按组件数分层分割
    comp_counts = (params_raw[:, :, 2] > 0).sum(axis=1)
    idx_all = np.arange(len(params_raw))
    idx_tr, idx_te = train_test_split(idx_all, test_size=args.test_ratio,
                                       random_state=42, stratify=comp_counts)
    tr_counts = comp_counts[idx_tr]
    idx_tr, idx_val = train_test_split(idx_tr, test_size=args.val_ratio,
                                        random_state=42, stratify=tr_counts)

    print(f"Train: {len(idx_tr)}, Val: {len(idx_val)}, Test: {len(idx_te)}")

    raw_tr_p, raw_tr_t = params_raw[idx_tr], temps_raw[idx_tr]
    raw_p_total_ref = float(
        np.nan_to_num(raw_tr_p, nan=0.0)[:, :, 2].sum(axis=1).max())
    print(f"raw train max total power: {raw_p_total_ref:.2f} W")

    if args.two_phase:
        phase1_dir = os.path.join(args.out_dir, 'phase1')
        phase2_dir = os.path.join(args.out_dir, 'phase2')

        ds_tr_phase1 = ThermalDatasetV3(raw_tr_p, raw_tr_t,
                                        p_total_ref=raw_p_total_ref)
        ds_val = ThermalDatasetV3(params_raw[idx_val], temps_raw[idx_val],
                                  max_power_ref=ds_tr_phase1.max_power_ref,
                                  p_total_ref=ds_tr_phase1.p_total_ref,
                                  theta_mean=ds_tr_phase1.theta_mean,
                                  theta_std=ds_tr_phase1.theta_std)
        ds_te  = ThermalDatasetV3(params_raw[idx_te], temps_raw[idx_te],
                                  max_power_ref=ds_tr_phase1.max_power_ref,
                                  p_total_ref=ds_tr_phase1.p_total_ref,
                                  theta_mean=ds_tr_phase1.theta_mean,
                                  theta_std=ds_tr_phase1.theta_std)

        norm_info = {
            'max_power_ref': ds_tr_phase1.max_power_ref,
            'p_total_ref': ds_tr_phase1.p_total_ref,
            'theta_mean': ds_tr_phase1.theta_mean,
            'theta_std': ds_tr_phase1.theta_std,
            't_amb': T_AMB,
            'board_mm': BOARD_MM,
        }
        save_norm_info(norm_info, args.out_dir)
        save_norm_info(norm_info, phase1_dir)
        save_norm_info(norm_info, phase2_dir)

        dl_tr_phase1 = DataLoader(ds_tr_phase1, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

        model = build_model(args, device)
        total_params = count_parameters(model)
        print(f"Model parameters: {total_params:,}")

        phase1_result = run_training_phase(
            model, dl_tr_phase1, dl_val, ds_tr_phase1, device, norm_info, phase1_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_bc=args.lambda_bc,
            lambda_pde=args.lambda_pde,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            log_every=args.log_every,
            phase_name='phase1',
            ckpt_args=vars(args),
        )

        ckpt_phase1 = torch.load(phase1_result['best_path'], map_location=device)
        model.load_state_dict(ckpt_phase1['model'])
        freeze_backbone_for_phase2(model)

        ft_p, ft_t = power_scale_augment(
            raw_tr_p, raw_tr_t,
            n_copies=args.phase2_power_aug_copies,
            scale_min=args.phase2_power_aug_min,
            scale_max=args.phase2_power_aug_max,
            include_original=False,
        )
        print(f"Phase2 high-power finetune set: {len(ft_p)} samples "
              f"(copies={args.phase2_power_aug_copies}, scale [{args.phase2_power_aug_min:.1f},{args.phase2_power_aug_max:.1f}])")

        ds_tr_phase2 = ThermalDatasetV3(
            ft_p, ft_t,
            max_power_ref=ds_tr_phase1.max_power_ref,
            p_total_ref=ds_tr_phase1.p_total_ref,
            theta_mean=ds_tr_phase1.theta_mean,
            theta_std=ds_tr_phase1.theta_std)
        dl_tr_phase2 = DataLoader(ds_tr_phase2, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

        phase2_result = run_training_phase(
            model, dl_tr_phase2, dl_val, ds_tr_phase2, device, norm_info, phase2_dir,
            epochs=args.phase2_epochs,
            lr=args.phase2_lr,
            weight_decay=args.weight_decay,
            lambda_bc=args.lambda_bc,
            lambda_pde=args.lambda_pde,
            early_stopping=args.early_stopping,
            patience=args.phase2_patience,
            min_delta=args.min_delta,
            log_every=args.log_every,
            phase_name='phase2',
            ckpt_args=vars(args),
        )

        best_ckpt = torch.load(phase2_result['best_path'], map_location=device)
        model.load_state_dict(best_ckpt['model'])

        shutil.copy2(phase2_result['best_path'], os.path.join(args.out_dir, 'setfno_v3_best.pth'))
        shutil.copy2(phase2_result['final_path'], os.path.join(args.out_dir, 'setfno_v3_final.pth'))
        shutil.copy2(os.path.join(phase2_dir, 'loss_curves.png'), os.path.join(args.out_dir, 'loss_curves.png'))

        print("\n=== Test Set Evaluation ===")
        _evaluate(model, dl_te, ds_te, device, norm_info,
                  os.path.join(args.out_dir, 'test_results'))

        cfg = vars(args).copy()
        cfg['training_mode'] = 'two_phase'
        cfg['model_params'] = total_params
        cfg['phase1_best_val_loss'] = phase1_result['best_val_loss']
        cfg['phase1_epochs_ran'] = phase1_result['epochs_ran']
        cfg['phase2_best_val_loss'] = phase2_result['best_val_loss']
        cfg['phase2_epochs_ran'] = phase2_result['epochs_ran']
        cfg['phase2_trainable_params'] = phase2_result['trainable_params']
        cfg['best_val_loss'] = phase2_result['best_val_loss']
        with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        print(f"\n✓ 两阶段训练完成！结果保存到: {args.out_dir}")
        return

    tr_p, tr_t = raw_tr_p, raw_tr_t
    if getattr(args, 'power_aug_copies', 0) > 0:
        tr_p, tr_t = power_scale_augment(
            tr_p, tr_t,
            n_copies=args.power_aug_copies,
            scale_min=args.power_aug_min,
            scale_max=args.power_aug_max,
        )
        print(f"Power-scale augmented: {len(raw_tr_p)} -> {len(tr_p)} samples "
              f"(x{args.power_aug_copies+1}, scale [{args.power_aug_min:.1f},{args.power_aug_max:.1f}])")

    p_total_ref = float(
        np.nan_to_num(tr_p, nan=0.0)[:, :, 2].sum(axis=1).max())
    print(f"p_total_ref (train max total power): {p_total_ref:.2f} W")

    ds_tr  = ThermalDatasetV3(tr_p, tr_t,
                              p_total_ref=p_total_ref)
    ds_val = ThermalDatasetV3(params_raw[idx_val], temps_raw[idx_val],
                              max_power_ref=ds_tr.max_power_ref,
                              p_total_ref=ds_tr.p_total_ref,
                              theta_mean=ds_tr.theta_mean,
                              theta_std=ds_tr.theta_std)
    ds_te  = ThermalDatasetV3(params_raw[idx_te], temps_raw[idx_te],
                              max_power_ref=ds_tr.max_power_ref,
                              p_total_ref=ds_tr.p_total_ref,
                              theta_mean=ds_tr.theta_mean,
                              theta_std=ds_tr.theta_std)

    norm_info = {
        'max_power_ref': ds_tr.max_power_ref,
        'p_total_ref': ds_tr.p_total_ref,
        'theta_mean': ds_tr.theta_mean,
        'theta_std': ds_tr.theta_std,
        't_amb': T_AMB,
        'board_mm': BOARD_MM,
    }
    save_norm_info(norm_info, args.out_dir)

    dl_tr  = DataLoader(ds_tr,  batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te  = DataLoader(ds_te,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args, device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    phase_result = run_training_phase(
        model, dl_tr, dl_val, ds_tr, device, norm_info, args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_bc=args.lambda_bc,
        lambda_pde=args.lambda_pde,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        log_every=args.log_every,
        phase_name='train',
        ckpt_args=vars(args),
    )

    best_ckpt = torch.load(phase_result['best_path'], map_location=device)
    model.load_state_dict(best_ckpt['model'])

    print("\n=== Test Set Evaluation ===")
    _evaluate(model, dl_te, ds_te, device, norm_info,
              os.path.join(args.out_dir, 'test_results'))

    cfg = vars(args).copy()
    cfg['model_params'] = n_params
    cfg['best_val_loss'] = phase_result['best_val_loss']
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✓ 训练完成！结果保存到: {args.out_dir}")


def _evaluate(model, dataloader, dataset, device, norm_info, save_prefix):
    """评估：计算 R² 并保存可视化"""
    os.makedirs(save_prefix, exist_ok=True)
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for params_7d, hmaps, _, total_p, temps_r in dataloader:
            params_7d = params_7d.to(device)
            hmaps     = hmaps.to(device)
            total_p_dev = torch.tensor(total_p.numpy()).to(device)

            theta_pred = model(params_7d, hmaps)
            T_pred = denorm_and_restore_T(
                theta_pred, total_p_dev,
                norm_info['theta_mean'], norm_info['theta_std'])
            T_true = temps_r.squeeze(1)  # (B, H, W)

            all_true.append(T_true.cpu().numpy().reshape(len(T_true), -1))
            all_pred.append(T_pred.cpu().numpy().reshape(len(T_pred), -1))

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    r2_total = r2_score(all_true.ravel(), all_pred.ravel())
    r2_per   = [r2_score(all_true[i], all_pred[i]) for i in range(len(all_true))]
    print(f"  R² (all pixels): {r2_total:.4f}")
    print(f"  R² (per sample mean): {np.mean(r2_per):.4f}  "
          f"std={np.std(r2_per):.4f}  min={np.min(r2_per):.4f}")

    # 保存几张对比图
    n_vis = min(6, len(all_true))
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 3*n_vis))
    for i in range(n_vis):
        T_t = all_true[i].reshape(GRID, GRID)
        T_p = all_pred[i].reshape(GRID, GRID)
        vmin, vmax = T_t.min(), T_t.max()
        axes[i, 0].imshow(T_t.T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('True T')
        axes[i, 1].imshow(T_p.T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Pred T  R²={r2_per[i]:.3f}')
        err = T_p - T_t
        im = axes[i, 2].imshow(err.T, cmap='coolwarm', origin='lower')
        axes[i, 2].set_title(f'Error (MAE={np.abs(err).mean():.2f}°C)')
        plt.colorbar(im, ax=axes[i, 2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, 'comparison.png'), dpi=150)
    plt.close()

    results = {
        'r2_all_pixels': float(r2_total),
        'r2_per_sample_mean': float(np.mean(r2_per)),
        'r2_per_sample_std': float(np.std(r2_per)),
        'r2_per_sample_min': float(np.min(r2_per)),
    }
    with open(os.path.join(save_prefix, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def _plot_loss(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history['train'], label='train')
    ax.semilogy(history['val'],   label='val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 泛化测试（--test-only 模式）
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# DeepOHeat-v1 风格 SOR Residual Refinement
# ──────────────────────────────────────────────────────────────────────────────

def sor_refine_theta(theta_np, hmap_shape_np, total_power_w,
                     max_iters=500, omega=1.5, tol=1e-4,
                     k_fr4=K_FR4, h_conv=H_CONV, dx_m=DX_M,
                     source_rel_threshold=0.1):
    """
    DeepOHeat-v1 hybrid: 用 NN 粗解初始化，再跑少量 SOR 修正。

    原理：
      1. 计算每个 interior 非热源点的 PDE residual: r[i,j] = Δθ[i,j]
         （非热源稳态区 Laplacian 应 = 0；residual 越大表示 NN 越不满足方程）
      2. 只对 |r[i,j]| > tol 的点做 SOR Gauss-Seidel 更新
      3. 每次迭代后重置 Robin BC: θ_edge = c_adj * θ_adj
      4. 若本轮最大更新量 < tol 则提前停止

    热源掩码用相对阈值（source_rel_threshold * max），与 physics_loss 一致：
      1元件时 Gaussian 峰值高，阈值成比例高；9元件分散后峰值低，阈值同步降低。
      始终覆盖各热源，不受元件数/功率分布影响。

    theta_np             : (H, W) numpy float32, 真实 θ 空间粗解（已反标准化）
    hmap_shape_np        : (H, W) numpy float32, 热源形状图（ch0, ~[0,1]）
    total_power_w        : float, 该样本总功率 (W)（保留接口，当前未使用）
    source_rel_threshold : 相对阈值，热源区 = hmap > source_rel_threshold * hmap.max()
    返回                 : (H, W) 修正后的热阻场 float32
    """
    T = theta_np.copy().astype(np.float64)
    H, W = T.shape

    # 相对阈值：与 physics_loss 中的处理保持一致
    hmap_max = float(hmap_shape_np.max())
    abs_threshold = source_rel_threshold * hmap_max if hmap_max > 1e-6 else 1e-6
    is_source = hmap_shape_np > abs_threshold  # (H, W) bool

    c = K_FR4 / (K_FR4 + H_CONV * dx_m)  # Robin BC 系数（与训练一致）

    for it in range(max_iters):
        max_update = 0.0
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if is_source[i, j]:
                    continue  # 热源区不修正，保留 NN 预测

                # 计算 PDE residual: r = Δθ = Σneighbors - 4*θ[i,j]
                neighbor_sum = T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1]
                residual = neighbor_sum - 4.0 * T[i, j]

                # 只对 residual 超过阈值的点做 SOR 更新
                if abs(residual) <= tol:
                    continue

                T_new = 0.25 * neighbor_sum
                update = omega * (T_new - T[i, j])
                T[i, j] += update
                max_update = max(max_update, abs(update))

        # 重置 Robin BC: θ_edge = c_adj * θ_adj
        T[0,  :] = c * T[1,  :]
        T[-1, :] = c * T[-2, :]
        T[:,  0] = c * T[:,  1]
        T[:, -1] = c * T[:, -2]

        # 若本轮最大更新量已低于 tol，提前收敛
        if max_update < tol:
            break

    return T.astype(np.float32)


def test_only(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.model_path, map_location=device)
    norm_info = ckpt['norm_info']

    model = SetFNOv3(
        d_model=args.d_model, num_heads=args.num_heads, n_sab=args.n_sab,
        fno_ch=args.fno_ch, fno_modes=args.fno_modes, n_fno=args.n_fno,
        dropout=0.0, grid=GRID, use_corrector=not args.no_corrector
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    params_raw = np.load(args.params)
    temps_raw  = np.load(args.temps)
    if params_raw.ndim == 2:
        params_raw = params_raw.reshape(len(params_raw), -1, 3)
    if temps_raw.ndim == 2:
        gs = int(round(np.sqrt(temps_raw.shape[1])))
        temps_raw = temps_raw.reshape(len(temps_raw), gs, gs)

    ds = ThermalDatasetV3(params_raw, temps_raw,
                           max_power_ref=norm_info['max_power_ref'],
                           p_total_ref=norm_info['p_total_ref'],
                           theta_mean=norm_info['theta_mean'],
                           theta_std=norm_info['theta_std'])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Testing {len(ds)} samples from: {args.params}")

    if getattr(args, 'sor_refine', False):
        print(f"SOR Refinement ON (max_iters={args.sor_iters})")
        theta_std = norm_info['theta_std']
        theta_mean = norm_info['theta_mean']
        # 手动推理 + SOR 修正
        all_true, all_pred = [], []
        with torch.no_grad():
            for params_7d, hmaps, _, total_p, temps_r in dl:
                params_7d_d = params_7d.to(device)
                hmaps_d     = hmaps.to(device)
                theta_pred_sc = model(params_7d_d, hmaps_d)  # (B,1,H,W)，scaled

                # 反标准化到真实 θ 空间
                theta_pred_raw = theta_pred_sc * theta_std + theta_mean  # (B,1,H,W)

                for b in range(len(params_7d)):
                    theta_np  = theta_pred_raw[b, 0].cpu().numpy()  # 真实 θ
                    hmap_sh   = hmaps[b, 0].numpy()                  # ch0 形状
                    tp        = float(total_p[b])
                    theta_ref = sor_refine_theta(
                        theta_np, hmap_sh, tp,
                        max_iters=args.sor_iters)
                    # 恢复 T = θ * P_total + T_amb
                    T_pred = theta_ref * tp + norm_info['t_amb']
                    T_true = temps_r[b, 0].numpy()
                    all_true.append(T_true.ravel())
                    all_pred.append(T_pred.ravel())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        r2 = r2_score(all_true.ravel(), all_pred.ravel())
        r2_per = [r2_score(all_true[i], all_pred[i]) for i in range(len(all_true))]
        print(f"  [SOR Refined] R²={r2:.4f}  per-sample mean={np.mean(r2_per):.4f}")
    else:
        results = _evaluate(model, dl, ds, device, norm_info, args.out_dir)
        print("Results:", results)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='SetFNO V3 Training')
    p.add_argument('--params',      default='training_data/params_count_sweep.npy')
    p.add_argument('--temps',       default='training_data/temps_count_sweep.npy')
    p.add_argument('--out-dir',     default='my_scripts/results_v3')
    p.add_argument('--model-path',  default='my_scripts/results_v3/setfno_v3_best.pth')
    p.add_argument('--test-only',   action='store_true')
    # 架构
    p.add_argument('--d-model',     type=int,   default=256)
    p.add_argument('--num-heads',   type=int,   default=8)
    p.add_argument('--n-sab',       type=int,   default=4)
    p.add_argument('--fno-ch',      type=int,   default=64)
    p.add_argument('--fno-modes',   type=int,   default=24)
    p.add_argument('--n-fno',       type=int,   default=6)
    p.add_argument('--dropout',     type=float, default=0.0)
    p.add_argument('--no-corrector',action='store_true')
    # 训练
    p.add_argument('--epochs',      type=int,   default=2000)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=5e-5)
    p.add_argument('--weight-decay',type=float, default=1e-5)
    p.add_argument('--test-ratio',  type=float, default=0.1)
    p.add_argument('--val-ratio',   type=float, default=0.1)
    p.add_argument('--early-stopping', action='store_true')
    p.add_argument('--patience',    type=int,   default=200)
    p.add_argument('--min-delta',   type=float, default=0.0)
    p.add_argument('--log-every',   type=int,   default=50)
    # 物理 loss 系数
    p.add_argument('--lambda-bc',     type=float, default=0.001)
    p.add_argument('--lambda-pde',    type=float, default=0.001)
    p.add_argument('--sor-refine',    action='store_true',
                   help='推理时对 PDE residual 大的区域做 SOR 修正（DeepOHeat-v1）')
    p.add_argument('--sor-iters',     type=int,   default=50,
                   help='SOR 修正最大迭代次数')
    # 功率缩放增强
    p.add_argument('--power-aug-copies', type=int,   default=0,
                   help='每条训练样本增强几份（0=关闭），如 3 则训练集扩大4倍')
    p.add_argument('--power-aug-min',    type=float, default=1.2,
                   help='功率缩放系数下界')
    p.add_argument('--power-aug-max',    type=float, default=2.5,
                   help='功率缩放系数上界')
    # 两阶段训练
    p.add_argument('--two-phase',         action='store_true',
                   help='两阶段训练：Phase1 原始数据；Phase2 冻结主干后高功率微调')
    p.add_argument('--phase2-epochs',     type=int,   default=400,
                   help='Phase2 微调轮次')
    p.add_argument('--phase2-lr',         type=float, default=1e-5,
                   help='Phase2 微调学习率')
    p.add_argument('--phase2-patience',   type=int,   default=80,
                   help='Phase2 早停 patience')
    p.add_argument('--phase2-power-aug-copies', type=int,   default=3,
                   help='Phase2 高倍增强份数（不含原始样本）')
    p.add_argument('--phase2-power-aug-min',    type=float, default=2.0,
                   help='Phase2 高倍增强缩放下界')
    p.add_argument('--phase2-power-aug-max',    type=float, default=2.5,
                   help='Phase2 高倍增强缩放上界')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.test_only:
        test_only(args)
    else:
        train(args)
