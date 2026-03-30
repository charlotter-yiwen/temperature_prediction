"""
test_generalization.py
=====================
对 6 组件数据进行泛化测试。
直接 import set_fno_thermal.py 模型，保证和训练时完全一致。
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import r2_score

# 路径
TP_DIR   = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction"
MODEL_DIR = os.path.join(TP_DIR, "hp_search", "small_lr_small_batch", "run_01")
TRAIN_DIR = os.path.join(TP_DIR, "training_data")

sys.path.insert(0, os.path.join(TP_DIR, "models"))
from set_fno_thermal import SetFNOModel

MODEL_PATH  = os.path.join(MODEL_DIR, "set_fno_model.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "run_config_latest.json")

GRID_SIZE  = 100
D_PER_COMP = 3


def prepare_gen_test_data(gen_params_raw, d_per_comp, norm_info, grid_size):
    """预处理泛化测试数据（与 set_fno_thermal.py 的逻辑一致）。"""
    n, max_comp, _ = gen_params_raw.shape
    params_3d = gen_params_raw.copy()

    valid_mask = ~np.isnan(params_3d[:, :, 0])
    comp_counts = valid_mask.sum(axis=1)

    total_power = np.nansum(params_3d[:, :, 2], axis=1)
    total_power = np.maximum(total_power, 0.1)

    params_3d = np.nan_to_num(params_3d, nan=0.0)
    params_3d[:, :, 0] /= norm_info["board_size"]
    params_3d[:, :, 1] /= norm_info["board_size"]
    if norm_info["max_power"] > 0:
        params_3d[:, :, 2] /= norm_info["max_power"]

    temps_2d = gen_params_raw  # temps 已经在外面 reshape 好了
    return params_3d.astype(np.float32), temps_2d, comp_counts, total_power


def main():
    # ── 1. 加载 checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args  = ckpt["args"]
    print(f"Model: n_components={args['n_components']}, "
          f"physics_norm={args['physics_norm']}, "
          f"d_model={args['d_model']}, fno_ch={args['fno_ch']}")

    scaler_y_mean  = ckpt["scaler_y_mean"]   # (10000,)
    scaler_y_scale = ckpt["scaler_y_scale"]

    # ── 2. 加载 gen_test 数据 ─────────────────────────────────────────────────
    gen_params = np.load(os.path.join(TRAIN_DIR, "gen_test_params.npy")).astype(np.float32)
    gen_temps  = np.load(os.path.join(TRAIN_DIR, "gen_test_temps.npy")).astype(np.float32)
    print(f"\nGen_test: params {gen_params.shape}, temps {gen_temps.shape}")

    n_gen = gen_params.shape[0]
    valid_mask = ~np.isnan(gen_params[:, :, 0])
    comp_counts = valid_mask.sum(axis=1)
    print(f"Component counts: {dict(zip(*np.unique(comp_counts, return_counts=True)))}")

    # ── 3. 归一化参数 ─────────────────────────────────────────────────────────
    # max_power 必须是训练时的值（2.0），不是 gen_test 里的
    valid_powers = gen_params[:, :, 2][~np.isnan(gen_params[:, :, 2])]
    max_power_gen = float(valid_powers.max()) if valid_powers.size else 1.0

    norm_info = {
        "board_size": 100.0,
        "max_power": args["n_components"] * 2.0,   # 训练时的 max_power = 5*2 = 10W? 不对
        "physics_norm": args["physics_norm"],
        "T_ambient": args["t_ambient"],
    }
    # 从 training_data params_count_sweep 重新计算正确的 max_power
    params_tr = np.load(os.path.join(TRAIN_DIR, "params_count_sweep.npy")).astype(np.float32)
    params_tr_3d = params_tr.reshape(params_tr.shape[0], args["n_components"], D_PER_COMP)
    max_power_tr = float(params_tr_3d[:, :, 2][~np.isnan(params_tr_3d[:, :, 2])].max())
    norm_info["max_power"] = max_power_tr
    print(f"max_power (from training): {max_power_tr}")

    # ── 4. 加载模型 ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SetFNOModel(
        d_in=D_PER_COMP,
        d_model=args["d_model"],
        num_heads=args["num_heads"],
        n_sab=args["n_sab"],
        fno_ch=args["fno_ch"],
        fno_modes=args["fno_modes"],
        n_fno=args["n_fno"],
        dropout=args["dropout"],
        out_size=GRID_SIZE,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded OK")

    # ── 5. 预处理 gen_test ───────────────────────────────────────────────────
    # gen_temps shape: (N, 10000) → (N, 100, 100)
    gen_temps_2d = gen_temps.reshape(n_gen, GRID_SIZE, GRID_SIZE)

    params_norm, _, comp_counts, total_power = prepare_gen_test_data(
        gen_params, D_PER_COMP, norm_info, GRID_SIZE)

    print(f"\nNormalized params[0] first 3 comps:")
    print(params_norm[0, :3])
    print(f"Total power[0]: {total_power[0]:.1f} W")

    # ── 6. 推理 ──────────────────────────────────────────────────────────────
    print(f"\nPredicting {n_gen} samples...")
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, n_gen, 8):
            batch = torch.from_numpy(params_norm[i:i+8]).to(device)
            pred  = model(batch).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    print(f"Model raw output range: [{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
    print(f"Model raw output mean: {preds_scaled.mean():.4f}")

    # ── 7. 逆归一化 ─────────────────────────────────────────────────────────
    physics_norm = args["physics_norm"]
    T_amb = args["t_ambient"]

    # 逆 StandardScaler: T_raw = T_scaled * scale + mean
    # preds_scaled: (N, 100, 100), scaler: (10000,)
    preds_flat = preds_scaled.reshape(n_gen, -1)                    # (N, 10000)
    preds_inv_flat = preds_flat * scaler_y_scale + scaler_y_mean    # (N, 10000)
    preds_inv = preds_inv_flat.reshape(n_gen, GRID_SIZE, GRID_SIZE) # (N, 100, 100)
    print(f"After scaler inverse range: [{preds_inv.min():.4f}, {preds_inv.max():.4f}]")

    # 逆 physics norm: T_final = T_raw * total_power + T_ambient
    if physics_norm:
        preds_final = preds_inv * total_power[:, None, None] + T_amb
    else:
        preds_final = preds_inv
    print(f"After physics inverse range: [{preds_final.min():.4f}, {preds_final.max():.4f}]")

    # ── 8. R² ───────────────────────────────────────────────────────────────
    r2_vals, r2_avg = [], []
    for i in range(n_gen):
        p = preds_final[i].ravel()
        t = gen_temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2_vals = np.array(r2_vals)
    finite = np.isfinite(r2_vals)
    r2_avg = float(np.mean(r2_vals[finite])) if finite.any() else float("nan")

    print(f"\n{'='*60}")
    print(f"  Generalization Results (6 components)")
    print(f"{'='*60}")
    print(f"  Overall R²: {r2_avg:.4f}")
    for i, r2 in enumerate(r2_vals):
        print(f"    Sample {i+1:2d}: R²={r2:8.4f}  "
              f"pred_range=[{preds_final[i].min():.1f},{preds_final[i].max():.1f}]  "
              f"true_range=[{gen_temps_2d[i].min():.1f},{gen_temps_2d[i].max():.1f}]")

    print(f"\n{'='*60}")
    if r2_avg > 0.9:   result = "EXCELLENT"
    elif r2_avg > 0.8: result = "GOOD"
    elif r2_avg > 0.5: result = "MODERATE"
    elif r2_avg > 0.0: result = "POOR"
    else:               result = "FAILED"
    print(f"  Result: {result} (R²={r2_avg:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
