"""
predict_plan_a_physics_gen.py
=============================
测试 Plan A + Physics 模型在 6/7/8/9 组件上的泛化性能。
"""

import os, sys, json, argparse, numpy as np, torch

TP_DIR = r"c:/Users/jkong/Documents/power brain_new/yiwen version/temperature_prediction"
sys.path.insert(0, TP_DIR)

MODEL_DIR = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics")
GEN_DIR   = os.path.join(TP_DIR, "data", "generation_dataset")
MODEL_PATH = os.path.join(MODEL_DIR, "plan_a_physics_phase2.pth")


def load_json_and_params(json_path, n_comp, powers):
    with open(json_path, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    temps = np.array([d["temperature"] for d in temp_data], dtype=np.float32)
    fname = os.path.basename(json_path)
    name  = fname.replace(".json", "")
    parts = name.split("_")
    nums  = [p for p in parts if p.lstrip("-").isdigit()]
    positions = []
    for i in range(0, len(nums) - 1, 2):
        x = int(nums[i])
        y = int(nums[i + 1])
        positions.append([x, y, powers[i // 2]])
    params = np.full((n_comp, 3), np.nan, dtype=np.float32)
    for j, pos in enumerate(positions):
        params[j] = pos
    return params, temps


def run_test(model, device, scaler_y_mean, scaler_y_scale,
             norm_info, model_args, gen_dir, n_comp, powers, label):
    json_files = sorted([f for f in os.listdir(gen_dir)
                         if f.startswith(f"count{n_comp}") and f.endswith(".json")])
    print(f"\n{label}: found {len(json_files)} files")

    params_list, temps_list = [], []
    for fname in json_files:
        p, t = load_json_and_params(os.path.join(gen_dir, fname), n_comp, powers)
        params_list.append(p)
        temps_list.append(t)

    params = np.stack(params_list, axis=0)
    temps  = np.stack(temps_list, axis=0)
    n      = params.shape[0]
    temps_2d = temps.reshape(n, 100, 100)

    # 归一化参数
    params_3d = params.copy()
    params_3d = np.nan_to_num(params_3d, nan=0.0)
    params_3d[:, :, 0] /= norm_info["board_size"]
    params_3d[:, :, 1] /= norm_info["board_size"]
    max_power = norm_info["max_power"]
    params_3d[:, :, 2] /= max_power

    total_p = np.nansum(params[:, :, 2], axis=1)
    total_p = np.maximum(total_p, 0.1)

    # 预测
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, n, 8):
            xb = torch.from_numpy(params_3d[i:i+8]).float().to(device)
            pred = model(xb).squeeze(1).cpu().numpy()  # (B, 100, 100) normalized
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)  # (B, 100, 100)

    # 反归一化: 使用 sklearn StandardScaler inverse_transform
    # preds_scaled: (B, 100, 100) normalized via scaler
    # scaler fitted on flattened (N, 10000) data per-pixel
    B = preds_scaled.shape[0]
    preds_flat = preds_scaled.reshape(B, -1)  # (B, 10000)
    preds_inv_flat = preds_flat * scaler_y_scale.reshape(1, -1) + scaler_y_mean.reshape(1, -1)
    preds_inv = preds_inv_flat.reshape(B, 100, 100)  # back to (B, 100, 100)
    # physics norm: multiply by total power and add ambient
    preds_inv = preds_inv * total_p[:, None, None] + norm_info["T_ambient"]

    # R²
    r2_vals = []
    for i in range(n):
        p = preds_inv[i].ravel()
        t = temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(np.corrcoef(t[mask], p[mask])[0, 1] ** 2 if mask.sum() > 1 else np.nan)
    r2_vals = np.array(r2_vals)
    r2_avg  = float(np.nanmean(r2_vals))

    print(f"  Overall R2: {r2_avg:.4f}")
    for i, r2 in enumerate(r2_vals):
        print(f"    Sample {i+1:2d}: R2={r2:8.4f}  "
              f"pred=[{preds_inv[i].min():.1f},{preds_inv[i].max():.1f}]  "
              f"true=[{temps_2d[i].min():.1f},{temps_2d[i].max():.1f}]")
    return r2_avg


def main():
    print(f"Loading model from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    print(f"Model args: d_model={args['d_model']}, physics_norm={args['physics_norm']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用原始 Plan A 架构
    from models.set_fno_thermal import SetFNOModel
    base_model = SetFNOModel(
        d_in=3,
        d_model=args["d_model"],
        num_heads=args["num_heads"],
        n_sab=args["n_sab"],
        fno_ch=args["fno_ch"],
        fno_modes=args["fno_modes"],
        n_fno=args["n_fno"],
        dropout=args["dropout"],
        out_size=100,
    ).to(device)

    # 检查 state_dict 的 key
    sd_keys = list(ckpt["state_dict"].keys())
    model_keys = list(base_model.state_dict().keys())
    print(f"Checkpoint keys (first 5): {sd_keys[:5]}")
    print(f"Model keys (first 5): {model_keys[:5]}")

    # 如果 checkpoint 包含 wrapper 的键，需要重新映射
    has_base_prefix = any(k.startswith("base.") for k in ckpt["state_dict"].keys())
    if has_base_prefix:
        print("Detected PlanAPlusPhysics wrapper checkpoint - extracting base weights")
        new_sd = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("base."):
                new_sd[k[5:]] = v  # remove "base." from base model keys
            # Skip wrapper's own buffers (grid_x, grid_y) - not in SetFNOModel
            elif k not in ("grid_x", "grid_y"):
                new_sd[k] = v
        ckpt["state_dict"] = new_sd

    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    norm_info = ckpt.get("norm_info", {
        "board_size": 100.0,
        "max_power": 13.7,
        "T_ambient": 25.0,
        "physics_norm": True,
    })

    scaler_y_mean  = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]

    print(f"\n{'='*60}")
    print(f"  Plan A + Physics Generalization Results")
    print(f"{'='*60}")
    print(f"Norm info: board_size={norm_info['board_size']}, max_power={norm_info['max_power']}, "
          f"T_ambient={norm_info['T_ambient']}")

    r2_6 = run_test(base_model, device, scaler_y_mean, scaler_y_scale,
                    norm_info, args, GEN_DIR,
                    6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], "6-Component (15.7W)")

    r2_7 = run_test(base_model, device, scaler_y_mean, scaler_y_scale,
                    norm_info, args, GEN_DIR,
                    7, [2.5]*7, "7-Component (17.5W)")

    r2_8 = run_test(base_model, device, scaler_y_mean, scaler_y_scale,
                    norm_info, args, GEN_DIR,
                    8, [2.5]*8, "8-Component (20.0W)")

    r2_9 = run_test(base_model, device, scaler_y_mean, scaler_y_scale,
                    norm_info, args, GEN_DIR,
                    9, [2.5]*8 + [10.0], "9-Component (30W)")

    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"    6-Component: R2 = {r2_6:.4f}")
    print(f"    7-Component: R2 = {r2_7:.4f}")
    print(f"    8-Component: R2 = {r2_8:.4f}")
    print(f"    9-Component: R2 = {r2_9:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
