"""
bc_sweep_test.py
Compare generalization R² for 3 bc_sweep models + original Plan B on 6/7/8/9 components.
"""
import os, json, numpy as np, torch, sys

TP_DIR = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction"
MODEL_DIR = os.path.join(TP_DIR, "model_v3")
GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")
sys.path.insert(0, TP_DIR)
from models.set_fno_thermal import SetFNOModel

def load_base_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    base_model = SetFNOModel(
        d_in=3, d_model=args["d_model"], num_heads=args["num_heads"],
        n_sab=args["n_sab"], fno_ch=args["fno_ch"], fno_modes=args["fno_modes"],
        n_fno=args["n_fno"], dropout=args["dropout"], out_size=100,
    ).to("cpu")
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
    base_model.load_state_dict(sd, strict=False)
    base_model.eval()
    scaler_y_mean = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]
    norm_info = ckpt.get("norm_info", {"board_size": 100.0, "max_power": 13.7, "T_ambient": 25.0, "physics_norm": True})
    return base_model, scaler_y_mean, scaler_y_scale, norm_info

def load_json_and_params(json_path, n_comp, powers):
    with open(json_path, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    temps = np.array([d["temperature"] for d in temp_data], dtype=np.float32)
    fname = os.path.basename(json_path)
    name = fname.replace(".json", "")
    parts = name.split("_")
    nums = [p for p in parts if p.lstrip("-").isdigit()]
    positions = []
    for i in range(0, len(nums) - 1, 2):
        x = int(nums[i])
        y = int(nums[i + 1])
        positions.append([x, y, powers[i // 2]])
    params = np.full((n_comp, 3), np.nan, dtype=np.float32)
    for j, pos in enumerate(positions):
        params[j] = pos
    return params, temps

def predict(model, params_3d, scaler_y_mean, scaler_y_scale, total_p, norm_info):
    model.eval()
    preds_scaled = []
    n = params_3d.shape[0]
    with torch.no_grad():
        for i in range(0, n, 8):
            xb = torch.from_numpy(params_3d[i:i+8]).float()
            pred = model(xb).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    B = preds_scaled.shape[0]
    preds_flat = preds_scaled.reshape(B, -1)
    preds_inv_flat = preds_flat * scaler_y_scale.reshape(1, -1) + scaler_y_mean.reshape(1, -1)
    preds_inv = preds_inv_flat.reshape(B, 100, 100)
    preds_inv = preds_inv * total_p[:, None, None] + norm_info["T_ambient"]
    return preds_inv

def test_model(model, scaler_y_mean, scaler_y_scale, norm_info, n_comp, powers, gen_dir):
    json_files = sorted([f for f in os.listdir(gen_dir) if f.startswith(f"count{n_comp}") and f.endswith(".json")])
    params_list, temps_list = [], []
    for fname in json_files:
        p, t = load_json_and_params(os.path.join(gen_dir, fname), n_comp, powers)
        params_list.append(p)
        temps_list.append(t)
    params = np.stack(params_list, axis=0)
    temps = np.stack(temps_list, axis=0)
    n = params.shape[0]
    temps_2d = temps.reshape(n, 100, 100)
    params_3d = params.copy()
    params_3d = np.nan_to_num(params_3d, nan=0.0)
    params_3d[:, :, 0] /= norm_info["board_size"]
    params_3d[:, :, 1] /= norm_info["board_size"]
    params_3d[:, :, 2] /= norm_info.get("max_power", 13.7)
    total_p = np.nansum(params[:, :, 2], axis=1)
    total_p = np.maximum(total_p, 0.1)
    preds_inv = predict(model, params_3d, scaler_y_mean, scaler_y_scale, total_p, norm_info)
    r2_vals = []
    for i in range(n):
        p, t = preds_inv[i].ravel(), temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(np.corrcoef(t[mask], p[mask])[0, 1]**2 if mask.sum() > 1 else np.nan)
    return np.array(r2_vals)

configs = [
    ("Original (bc=0.01)", "results_plan_a_physics", "plan_a_physics_phase2.pth"),
    ("Exp2 (bc=0.001)", "results_bc_0_001", "plan_a_physics_phase2.pth"),
    ("10k (bc=0.001)", "results_bc_0_001_10k", "plan_a_physics_phase2.pth"),
    ("10k (bc=0.0005)", "results_bc_0_0005_10k", "plan_a_physics_phase2.pth"),
    ("100k (bc=0.0001)", "results_bc_0_0001_100k", "plan_a_physics_phase2.pth"),
    # Freq Branch 需要单独测试 (不同架构: decoder.out_conv=128ch)
]

test_configs = [
    (6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], "6-Component (15.7W)"),
    (7, [2.5]*7, "7-Component (17.5W)"),
    (8, [2.5]*8, "8-Component (20.0W)"),
    (9, [3.3]*9, "9-Component (30W)"),
]

print(f"{'Model':<22} {'6C-R2':>8} {'7C-R2':>8} {'8C-R2':>8} {'9C-R2':>8}")
print("-" * 58)

for model_name, model_subdir, model_file in configs:
    model_path = os.path.join(MODEL_DIR, model_subdir, model_file)
    if not os.path.exists(model_path):
        print(f"{model_name:<22} {'N/A':>8}")
        continue
    model, scaler_y_mean, scaler_y_scale, norm_info = load_base_model(model_path)
    r2_results = []
    for n_comp, powers, label in test_configs:
        r2 = test_model(model, scaler_y_mean, scaler_y_scale, norm_info, n_comp, powers, GEN_DIR)
        r2_results.append(np.nanmean(r2))
    print(f"{model_name:<22} {r2_results[0]:>8.4f} {r2_results[1]:>8.4f} {r2_results[2]:>8.4f} {r2_results[3]:>8.4f}")