"""
predict_pinn_gen_physics_fix.py
===============================
测试修正物理方程 PINN 在 6/7/8/9 组件上的泛化性能。
"""

import os, sys, json, argparse, numpy as np, torch
from sklearn.metrics import r2_score

TP_DIR = r"c:/Users/jkong/Documents/power brain_new/yiwen version/temperature_prediction"
sys.path.insert(0, os.path.join(TP_DIR, "model_v3", "models"))
from pinn_v3_physics_fix import ThermalPINNPhysicsFix


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


def run_test(model, device, model_args, gen_dir, n_comp, powers, label):
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

    params_3d = params.copy()
    params_3d[:, :, 0] /= 100.0
    params_3d[:, :, 1] /= 100.0

    total_p = np.nansum(params_3d[:, :, 2], axis=1)
    total_p = np.maximum(total_p, 0.1)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n, 8):
            xb = torch.from_numpy(params_3d[i:i+8]).float().to(device)
            pb = torch.from_numpy(total_p[i:i+8]).float().to(device)
            pred = model.predict_grid(xb, pb).squeeze(1).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    r2_vals = []
    for i in range(n):
        p = preds[i].ravel()
        t = temps_2d[i].ravel()
        mask = np.isfinite(p) & np.isfinite(t)
        r2_vals.append(r2_score(t[mask], p[mask]) if mask.any() else np.nan)
    r2_vals = np.array(r2_vals)
    r2_avg  = float(np.mean(r2_vals[np.isfinite(r2_vals)]))

    print(f"  Overall R2: {r2_avg:.4f}")
    for i, r2 in enumerate(r2_vals):
        print(f"    Sample {i+1:2d}: R2={r2:8.4f}  "
              f"pred=[{preds[i].min():.1f},{preds[i].max():.1f}]  "
              f"true=[{temps_2d[i].min():.1f},{temps_2d[i].max():.1f}]")
    return r2_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gen-dir",    required=True)
    args_cli = parser.parse_args()

    ckpt = torch.load(args_cli.model_path, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]

    print(f"PINN Physics-Fix Model: d_hidden={model_args['d_hidden']}, n_layers={model_args['n_layers']}, "
          f"n_freqs={model_args['n_freqs']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalPINNPhysicsFix(
        d_hidden=model_args["d_hidden"],
        n_layers=model_args["n_layers"],
        n_freqs=model_args["n_freqs"],
        n_sources=model_args["max_components"],
        dropout=model_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"  PINN Physics-Fix Generalization Results")
    print(f"{'='*60}")

    r2_6 = run_test(model, device, model_args, args_cli.gen_dir,
                    6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], "6-Component (15.7W)")

    r2_7 = run_test(model, device, model_args, args_cli.gen_dir,
                    7, [2.5]*7, "7-Component (17.5W)")

    r2_8 = run_test(model, device, model_args, args_cli.gen_dir,
                    8, [2.5]*8, "8-Component (20.0W)")

    r2_9 = run_test(model, device, model_args, args_cli.gen_dir,
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
