"""
compare_gen.py - 统一测试 3 种方法的 6 组件泛化性能
"""
import os, sys, json, numpy as np, torch
from sklearn.metrics import r2_score

TP_DIR = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction"
POWERS = [2.5, 2.2, 3.0, 2.8, 3.2, 2.0]

def parse_gen_filename(filename):
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

def test_model(model_dir, model_name):
    MODEL_PATH = os.path.join(model_dir, "set_fno_model.pth")
    sys.path.insert(0, os.path.join(TP_DIR, "models"))
    from set_fno_thermal import SetFNOModel

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    scaler_y_mean = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]

    GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")
    json_files = sorted([f for f in os.listdir(GEN_DIR) if f.endswith(".json")])

    gen_params_list, gen_temps_list = [], []
    for fname in json_files:
        params, temps = load_gen_json(os.path.join(GEN_DIR, fname))
        gen_params_list.append(params)
        gen_temps_list.append(temps)
    gen_params = np.stack(gen_params_list, axis=0)
    gen_temps = np.stack(gen_temps_list, axis=0)
    n_gen = gen_params.shape[0]

    # 归一化
    params_tr = np.load(os.path.join(TP_DIR, "training_data", "params_count_sweep.npy")).astype(np.float32)
    params_tr_3d = params_tr.reshape(params_tr.shape[0], 5, 3)
    max_power_tr = float(params_tr_3d[:, :, 2][~np.isnan(params_tr_3d[:, :, 2])].max())

    norm_info = {"board_size": 100.0, "max_power": max_power_tr, "physics_norm": args["physics_norm"], "T_ambient": args["t_ambient"]}

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

    gen_temps_2d = gen_temps.reshape(n_gen, 100, 100)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetFNOModel(d_in=3, d_model=args["d_model"], num_heads=args["num_heads"],
                        n_sab=args["n_sab"], fno_ch=args["fno_ch"], fno_modes=args["fno_modes"],
                        n_fno=args["n_fno"], dropout=args["dropout"], out_size=100).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 推理
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, n_gen, 8):
            batch = torch.from_numpy(params_3d[i:i+8]).to(device)
            pred = model(batch).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    # 逆归一化
    preds_flat = preds_scaled.reshape(n_gen, -1)
    preds_inv_flat = preds_flat * scaler_y_scale + scaler_y_mean
    preds_inv = preds_inv_flat.reshape(n_gen, 100, 100)

    physics_norm = args["physics_norm"]
    T_amb = args["t_ambient"]
    if physics_norm:
        preds_final = preds_inv * total_power[:, None, None] + T_amb
    else:
        preds_final = preds_inv

    # R²
    r2_vals = []
    for i in range(n_gen):
        p = preds_final[i].ravel()
        t = gen_temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2_vals = np.array(r2_vals)
    r2_avg = float(np.mean(r2_vals[np.isfinite(r2_vals)]))

    return r2_avg, r2_vals


METHODS = [
    ("Method1 bs8+physics",    os.path.join(TP_DIR, "hp_search", "method1_bs8_physics", "run_01")),
    ("Method2 bs4+physics",    os.path.join(TP_DIR, "hp_search", "method2_bs4_physics", "run_01")),
    ("Method3 bs8 no physics", os.path.join(TP_DIR, "hp_search", "method3_bs8_no_physics", "run_01")),
    ("Method bs32+physics",    os.path.join(TP_DIR, "hp_search", "method_bs32_physics", "run_01")),
]

print("=" * 70)
print("6 Component Generalization Results (Total Power = 15.7W)")
print("=" * 70)
results = {}
for name, model_dir in METHODS:
    r2_avg, r2_vals = test_model(model_dir, name)
    results[name] = r2_avg
    print(f"\n{name}: R² = {r2_avg:.4f}")
    for i, r2 in enumerate(r2_vals):
        print(f"  Sample {i+1:2d}: R²={r2:8.4f}")

print(f"\n{'='*70}")
best = max(results, key=results.get)
print(f"Best: {best} with R² = {results[best]:.4f}")
print(f"{'='*70}")
