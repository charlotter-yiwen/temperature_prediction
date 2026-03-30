"""
predict_gen_test.py
用 hp_search 最佳模型预测新产生的 10 个随机 6 组件样本。
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import r2_score

TP_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(TP_DIR, "hp_search", "slow_learning_1000ep", "run_01")
MODEL_PATH  = os.path.join(MODEL_DIR, "set_fno_model.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "run_config_latest.json")
GEN_DIR  = os.path.join(TP_DIR, "data", "generation_dataset")

sys.path.insert(0, os.path.join(TP_DIR, "models"))
from set_fno_thermal import SetFNOModel

GRID_SIZE  = 100
D_PER_COMP = 3


# 训练功率分布：U1~U5 + U6(2W)
POWERS = [2.5, 2.2, 3.0, 2.8, 3.2, 2.0]


def parse_gen_filename(filename):
    """从文件名解析 6 组件位置."""
    name = filename.replace(".json", "")
    parts = name.split("_")
    nums = [p for p in parts if p.lstrip("-").isdigit()]
    positions = []
    for i in range(0, len(nums) - 1, 2):
        x = int(nums[i])
        y = int(nums[i + 1])
        positions.append([x, y, POWERS[i // 2]])
    return len(positions), positions


def load_gen_json(json_path):
    """加载 JSON 并返回 (params_3d, temps_2d)."""
    with open(json_path, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    temps = np.array([d["temperature"] for d in temp_data], dtype=np.float32)
    fname = os.path.basename(json_path)
    n_comp, positions = parse_gen_filename(fname)
    max_comp = 6
    params = np.full((max_comp, 3), np.nan, dtype=np.float32)
    for j, pos in enumerate(positions):
        params[j] = pos
    return params, temps


def main():
    # ── 1. 加载 checkpoint ─────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args  = ckpt["args"]
    print(f"Model: n_components={args['n_components']}, physics_norm={args['physics_norm']}, "
          f"d_model={args['d_model']}, fno_ch={args['fno_ch']}")

    scaler_y_mean  = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]

    # ── 2. 加载 gen_test 数据 ─────────────────────────────────────
    json_files = sorted([f for f in os.listdir(GEN_DIR) if f.endswith(".json")])
    print(f"\nFound {len(json_files)} JSON files")

    gen_params_list = []
    gen_temps_list = []
    for fname in json_files:
        params, temps = load_gen_json(os.path.join(GEN_DIR, fname))
        gen_params_list.append(params)
        gen_temps_list.append(temps)

    gen_params = np.stack(gen_params_list, axis=0)   # (N, 6, 3)
    gen_temps  = np.stack(gen_temps_list, axis=0)    # (N, 10000)
    n_gen = gen_params.shape[0]
    print(f"Gen_test: params {gen_params.shape}, temps {gen_temps.shape}")

    # ── 3. 归一化参数 ─────────────────────────────────────────────
    params_tr = np.load(os.path.join(TP_DIR, "training_data", "params_count_sweep.npy")).astype(np.float32)
    params_tr_3d = params_tr.reshape(params_tr.shape[0], args["n_components"], D_PER_COMP)
    max_power_tr = float(params_tr_3d[:, :, 2][~np.isnan(params_tr_3d[:, :, 2])].max())

    norm_info = {
        "board_size": 100.0,
        "max_power": max_power_tr,
        "physics_norm": args["physics_norm"],
        "T_ambient": args["t_ambient"],
    }
    print(f"max_power (from training): {max_power_tr}")

    # ── 4. 预处理数据 ─────────────────────────────────────────────
    valid_mask = ~np.isnan(gen_params[:, :, 0])
    comp_counts = valid_mask.sum(axis=1)
    total_power = np.nansum(gen_params[:, :, 2], axis=1)
    total_power = np.maximum(total_power, 0.1)

    params_3d = gen_params.copy()
    params_3d = np.nan_to_num(params_3d, nan=0.0)
    params_3d[:, :, 0] /= norm_info["board_size"]
    params_3d[:, :, 1] /= norm_info["board_size"]
    if norm_info["max_power"] > 0:
        params_3d[:, :, 2] /= norm_info["max_power"]

    gen_temps_2d = gen_temps.reshape(n_gen, GRID_SIZE, GRID_SIZE)

    print(f"\nNormalized params[0] first 3 comps:")
    print(params_3d[0, :3])
    print(f"Total power[0]: {total_power[0]:.1f} W")

    # ── 5. 加载模型 ───────────────────────────────────────────────
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

    # ── 6. 推理 ───────────────────────────────────────────────────
    print(f"\nPredicting {n_gen} samples...")
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, n_gen, 8):
            batch = torch.from_numpy(params_3d[i:i+8]).to(device)
            pred  = model(batch).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    print(f"Model raw output range: [{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
    print(f"Model raw output mean: {preds_scaled.mean():.4f}")

    # ── 7. 逆归一化 ───────────────────────────────────────────────
    physics_norm = args["physics_norm"]
    T_amb = args["t_ambient"]

    preds_flat = preds_scaled.reshape(n_gen, -1)
    preds_inv_flat = preds_flat * scaler_y_scale + scaler_y_mean
    preds_inv = preds_inv_flat.reshape(n_gen, GRID_SIZE, GRID_SIZE)
    print(f"After scaler inverse range: [{preds_inv.min():.4f}, {preds_inv.max():.4f}]")

    if physics_norm:
        preds_final = preds_inv * total_power[:, None, None] + T_amb
    else:
        preds_final = preds_inv
    print(f"After physics inverse range: [{preds_final.min():.4f}, {preds_final.max():.4f}]")

    # ── 8. R² ─────────────────────────────────────────────────────
    r2_vals = []
    for i in range(n_gen):
        p = preds_final[i].ravel()
        t = gen_temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2_vals = np.array(r2_vals)
    r2_avg = float(np.mean(r2_vals[np.isfinite(r2_vals)]))

    print(f"\n{'='*60}")
    print(f"  Generalization Results (6 components, random positions)")
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
