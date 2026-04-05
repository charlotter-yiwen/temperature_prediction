"""
predict_plan_a_gen.py
用方案A大模型预测 6/7/8 组件泛化
"""
import os, sys, json, numpy as np, torch
from sklearn.metrics import r2_score

TP_DIR = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction"
MODEL_DIR = os.path.join(TP_DIR, "hp_search", "plan_a_balanced")
MODEL_PATH = os.path.join(TP_DIR, "set_fno_model.pth")
GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")

sys.path.insert(0, os.path.join(TP_DIR, "models"))
from set_fno_thermal import SetFNOModel


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


def test_gen(model_path, model_dir, label):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    scaler_y_mean = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]

    # 归一化
    params_tr = np.load(os.path.join(TP_DIR, "training_data", "params_count_sweep.npy")).astype(np.float32)
    params_tr_3d = params_tr.reshape(params_tr.shape[0], 5, 3)
    max_power_tr = float(params_tr_3d[:, :, 2][~np.isnan(params_tr_3d[:, :, 2])].max())
    norm_info = {"board_size": 100.0, "max_power": max_power_tr, "physics_norm": args["physics_norm"], "T_ambient": args["t_ambient"]}

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetFNOModel(d_in=3, d_model=args["d_model"], num_heads=args["num_heads"],
                        n_sab=args["n_sab"], fno_ch=args["fno_ch"], fno_modes=args["fno_modes"],
                        n_fno=args["n_fno"], dropout=args["dropout"], out_size=100).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return device, model, args, norm_info, scaler_y_mean, scaler_y_scale


def run_test(n_comp, powers, total_power, label):
    json_files = sorted([f for f in os.listdir(GEN_DIR) if f.startswith(f"count{n_comp}") and f.endswith(".json")])
    print(f"\n{label}: found {len(json_files)} files")

    params_list, temps_list = [], []
    for fname in json_files:
        p, t = load_json_and_params(os.path.join(GEN_DIR, fname), n_comp, powers)
        params_list.append(p)
        temps_list.append(t)

    params = np.stack(params_list, axis=0)
    temps = np.stack(temps_list, axis=0)
    n = params.shape[0]
    temps_2d = temps.reshape(n, 100, 100)

    total_p = np.nansum(params[:, :, 2], axis=1)
    total_p = np.maximum(total_p, 0.1)

    params_3d = np.nan_to_num(params.copy(), nan=0.0)
    params_3d[:, :, 0] /= 100.0
    params_3d[:, :, 1] /= 100.0
    params_3d[:, :, 2] /= max_power_tr

    preds_scaled = []
    with torch.no_grad():
        for i in range(0, n, 8):
            batch = torch.from_numpy(params_3d[i:i+8]).to(device)
            pred = model(batch).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    preds_flat = preds_scaled.reshape(n, -1)
    preds_inv_flat = preds_flat * scaler_y_scale + scaler_y_mean
    preds_inv = preds_inv_flat.reshape(n, 100, 100)

    if args["physics_norm"]:
        preds_final = preds_inv * total_p[:, None, None] + args["t_ambient"]
    else:
        preds_final = preds_inv

    r2_vals = []
    for i in range(n):
        p = preds_final[i].ravel()
        t = temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2_vals = np.array(r2_vals)
    r2_avg = float(np.mean(r2_vals[np.isfinite(r2_vals)]))

    print(f"  Overall R²: {r2_avg:.4f}")
    for i, r2 in enumerate(r2_vals):
        print(f"    Sample {i+1:2d}: R²={r2:8.4f}  pred=[{preds_final[i].min():.1f},{preds_final[i].max():.1f}]  true=[{temps_2d[i].min():.1f},{temps_2d[i].max():.1f}]")
    return r2_avg


# 加载模型
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
args = ckpt["args"]
print(f"Plan A Model: d_model={args['d_model']}, n_sab={args['n_sab']}, fno_ch={args['fno_ch']}")
scaler_y_mean = ckpt["scaler_y_mean"]
scaler_y_scale = ckpt["scaler_y_scale"]

params_tr = np.load(os.path.join(TP_DIR, "training_data", "params_count_sweep.npy")).astype(np.float32)
params_tr_3d = params_tr.reshape(params_tr.shape[0], 5, 3)
max_power_tr = float(params_tr_3d[:, :, 2][~np.isnan(params_tr_3d[:, :, 2])].max())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SetFNOModel(d_in=3, d_model=args["d_model"], num_heads=args["num_heads"],
                    n_sab=args["n_sab"], fno_ch=args["fno_ch"], fno_modes=args["fno_modes"],
                    n_fno=args["n_fno"], dropout=args["dropout"], out_size=100).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

print(f"\n{'='*60}")
print(f"  Plan A Generalization Results")
print(f"  Model: {args['d_model']}d_model, {args['n_sab']}SAB, fno_ch={args['fno_ch']}")
print(f"{'='*60}")

# 6组件: [2.5, 2.2, 3.0, 2.8, 3.2, 2.0] = 15.7W
r2_6 = run_test(6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], 15.7, "6-Component (15.7W)")

# 7组件: all 2.5W = 17.5W
r2_7 = run_test(7, [2.5]*7, 17.5, "7-Component (17.5W)")

# 8组件: all 2.5W = 20W
r2_8 = run_test(8, [2.5]*8, 20.0, "8-Component (20.0W)")

print(f"\n{'='*60}")
print(f"  Summary:")
print(f"    6-Component: R² = {r2_6:.4f}")
print(f"    7-Component: R² = {r2_7:.4f}")
print(f"    8-Component: R² = {r2_8:.4f}")
print(f"{'='*60}")