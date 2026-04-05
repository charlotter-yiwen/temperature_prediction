"""
plot_gen_comparison.py
======================
Generate GT vs Prediction comparison heatmaps for 6/7/8 component generalization data.
Uses the trained Plan A + Physics model (plan_a_physics_phase2.pth).
"""

import os, json, numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import Normalize

TP_DIR = r"c:/Users/jkong/Documents/power brain_new/yiwen version/temperature_prediction"
MODEL_DIR = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics")
GEN_DIR   = os.path.join(TP_DIR, "data", "generation_dataset")
OUT_DIR   = os.path.join(MODEL_DIR, "gen_comparison")
MODEL_PATH = os.path.join(MODEL_DIR, "plan_a_physics_phase2.pth")

os.makedirs(OUT_DIR, exist_ok=True)


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


def load_model():
    import sys
    sys.path.insert(0, TP_DIR)
    from models.set_fno_thermal import SetFNOModel

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args = ckpt["args"]

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
    ).to("cpu")

    # Extract base weights from wrapper checkpoint
    has_base_prefix = any(k.startswith("base.") for k in ckpt["state_dict"].keys())
    if has_base_prefix:
        new_sd = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("base."):
                new_sd[k[5:]] = v
            elif k not in ("grid_x", "grid_y"):
                new_sd[k] = v
        ckpt["state_dict"] = new_sd

    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    scaler_y_mean  = ckpt["scaler_y_mean"]
    scaler_y_scale = ckpt["scaler_y_scale"]
    norm_info = ckpt.get("norm_info", {
        "board_size": 100.0,
        "max_power": 13.7,
        "T_ambient": 25.0,
        "physics_norm": True,
    })

    return base_model, scaler_y_mean, scaler_y_scale, norm_info


def predict(model, params_3d, scaler_y_mean, scaler_y_scale, total_p, norm_info, device="cpu"):
    model.eval()
    preds_scaled = []
    n = params_3d.shape[0]
    with torch.no_grad():
        for i in range(0, n, 8):
            xb = torch.from_numpy(params_3d[i:i+8]).float().to(device)
            pred = model(xb).squeeze(1).cpu().numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    # Inverse transform
    B = preds_scaled.shape[0]
    preds_flat = preds_scaled.reshape(B, -1)
    preds_inv_flat = preds_flat * scaler_y_scale.reshape(1, -1) + scaler_y_mean.reshape(1, -1)
    preds_inv = preds_inv_flat.reshape(B, 100, 100)
    preds_inv = preds_inv * total_p[:, None, None] + norm_info["T_ambient"]
    return preds_inv


def plot_comparison(pred, true, title, save_path):
    """Plot GT | Pred | Error side by side."""
    err = np.abs(pred - true)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    # Ground Truth
    im0 = axes[0].imshow(true, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth\n[{true.min():.1f}, {true.max():.1f}]°C', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Prediction
    im1 = axes[1].imshow(pred, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Prediction\n[{pred.min():.1f}, {pred.max():.1f}]°C', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error
    im2 = axes[2].imshow(err, cmap='Reds', origin='lower')
    axes[2].set_title(f'Absolute Error\nMax={err.max():.1f}°C', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    print("Loading model...")
    model, scaler_y_mean, scaler_y_scale, norm_info = load_model()

    configs = [
        (6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], "6-Component (15.7W)"),
        (7, [2.5]*7,                         "7-Component (17.5W)"),
        (8, [2.5]*8,                         "8-Component (20.0W)"),
    ]

    for n_comp, powers, label in configs:
        json_files = sorted([f for f in os.listdir(GEN_DIR)
                            if f.startswith(f"count{n_comp}") and f.endswith(".json")])

        params_list, temps_list = [], []
        for fname in json_files:
            p, t = load_json_and_params(os.path.join(GEN_DIR, fname), n_comp, powers)
            params_list.append(p)
            temps_list.append(t)

        params = np.stack(params_list, axis=0)
        temps  = np.stack(temps_list, axis=0)
        n = params.shape[0]
        temps_2d = temps.reshape(n, 100, 100)

        # Normalize params
        params_3d = params.copy()
        params_3d = np.nan_to_num(params_3d, nan=0.0)
        params_3d[:, :, 0] /= norm_info["board_size"]
        params_3d[:, :, 1] /= norm_info["board_size"]
        max_power = norm_info.get("max_power", 13.7)
        params_3d[:, :, 2] /= max_power

        total_p = np.nansum(params[:, :, 2], axis=1)
        total_p = np.maximum(total_p, 0.1)

        print(f"\n{label}: {n} samples, predicting...")
        preds_inv = predict(model, params_3d, scaler_y_mean, scaler_y_scale, total_p, norm_info)

        # Compute R²
        r2_vals = []
        for i in range(n):
            p, t = preds_inv[i].ravel(), temps_2d[i].ravel()
            mask = np.isfinite(p) & np.isfinite(t)
            r2_vals.append(np.corrcoef(t[mask], p[mask])[0, 1]**2 if mask.sum()>1 else np.nan)
        r2_vals = np.array(r2_vals)

        # Plot each sample
        comp_dir = os.path.join(OUT_DIR, label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"))
        os.makedirs(comp_dir, exist_ok=True)

        for i in range(n):
            fname = json_files[i]
            r2 = r2_vals[i]
            title = f"{label} Sample {i+1:02d}  R2={r2:.4f}"
            save_path = os.path.join(comp_dir, f"sample_{i+1:02d}.png")
            plot_comparison(preds_inv[i], temps_2d[i], title, save_path)

        # Summary
        print(f"  {label} Overall R2: {np.nanmean(r2_vals):.4f}")

    print(f"\nAll comparisons saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
