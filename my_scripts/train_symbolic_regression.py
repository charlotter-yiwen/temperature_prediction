"""
train_symbolic_regression.py
============================

Physics-inspired symbolic regression baseline for PCB temperature fields.

The fitted formula is linear in symbolic basis functions:

    T(x, y) = T_amb + b0 + sum_j c_j * phi_j(x, y, components)

Each phi_j is an interpretable expression built from source power, source
distance, boundary distance, and component count. This keeps the model easy to
inspect while using the existing SOR-generated dataset as supervision.

Example:
    python my_scripts/train_symbolic_regression.py \
        --params training_data/params_count_sweep.npy \
        --temps training_data/temps_count_sweep.npy \
        --samples-per-case 800 \
        --out-dir my_scripts/results_symbolic_regression

Quick smoke test:
    python my_scripts/train_symbolic_regression.py --samples-per-case 40 \
        --max-train-cases 60 --max-gen-cases 2 --out-dir my_scripts/results_symbolic_quick
"""

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


GRID = 100
BOARD_MM = 100.0
T_AMB = 25.0

TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")

TEST_CONFIGS = [
    (6, [2.5, 2.2, 3.0, 2.8, 3.2, 2.0], "6-Component (15.7W)"),
    (7, [2.5] * 7, "7-Component (17.5W)"),
    (8, [2.5] * 8, "8-Component (20.0W)"),
    (9, [2.5] * 8 + [10.0], "9-Component (30.0W)"),
    (10, [2.5] * 9 + [10.0], "10-Component (32.5W)"),
]


def make_grid(grid=GRID):
    lin = np.linspace(0.5 / grid, 1.0 - 0.5 / grid, grid, dtype=np.float32)
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    x = gx.reshape(-1)
    y = gy.reshape(-1)
    edge = np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]).astype(np.float32)
    return x, y, edge


GRID_X, GRID_Y, GRID_EDGE = make_grid()


FEATURE_NAMES = [
    "1",
    "P_total",
    "n_active",
    "edge",
    "edge^2",
    "P_total*n_active",
    "P_total*edge",
    "P_total*edge^2",
    "sum_i P_i/(r_i+eps)",
    "sum_i P_i/(r_i^2+eps^2)",
    "sum_i P_i*log(r_i+eps)",
    "sum_i P_i*r_i",
    "sum_i P_i*exp(-r_i^2/(2*0.03^2))",
    "sum_i P_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i P_i*exp(-r_i^2/(2*0.10^2))",
    "edge*sum_i P_i/(r_i+eps)",
    "edge*sum_i P_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i P_i*s_i*exp(-r_i^2/(2*0.10^2))",
    "n_active*sum_i P_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i w_i/(r_i+eps)",
    "sum_i w_i/(r_i^2+eps^2)",
    "sum_i w_i*log(r_i+eps)",
    "sum_i w_i*r_i",
    "sum_i w_i*exp(-r_i^2/(2*0.03^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.10^2))",
    "edge*sum_i w_i/(r_i+eps)",
    "edge*sum_i w_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i w_i*s_i*exp(-r_i^2/(2*0.10^2))",
    "n_active*sum_i w_i*exp(-r_i^2/(2*0.06^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.015^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.020^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.040^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.080^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.140^2))",
    "sum_i w_i*exp(-r_i^2/(2*0.200^2))",
    "edge*sum_i w_i*exp(-r_i^2/(2*0.040^2))",
    "edge*sum_i w_i*exp(-r_i^2/(2*0.080^2))",
    "edge*sum_i w_i*exp(-r_i^2/(2*0.140^2))",
    "edge*sum_i w_i*exp(-r_i^2/(2*0.200^2))",
]

FEATURE_SETS = {
    "full": list(range(len(FEATURE_NAMES))),
    # Conservative formula for component-count extrapolation. It avoids direct
    # n_active and P_total terms when fitting theta=(T-Tamb)/P_total.
    "physics": [0, 3, 4, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] + list(range(30, len(FEATURE_NAMES))),
}


@dataclass
class SymbolicModel:
    scaler: StandardScaler
    regressor: object
    feature_names: list
    feature_indices: list
    eps_norm: float
    target: str

    def predict_target(self, feature_matrix):
        selected = feature_matrix[:, self.feature_indices]
        return self.regressor.predict(self.scaler.transform(selected))


def load_npy_dataset(params_path, temps_path):
    params = np.load(params_path)
    temps = np.load(temps_path)
    if params.ndim == 2:
        params = params.reshape(params.shape[0], -1, 3)
    if temps.ndim == 2:
        grid = int(round(np.sqrt(temps.shape[1])))
        temps = temps.reshape(temps.shape[0], grid, grid)
    params = np.nan_to_num(params.astype(np.float32), nan=0.0)
    temps = temps.astype(np.float32)
    return params, temps


def component_counts(params):
    return (params[:, :, 2] > 0).sum(axis=1)


def build_features_for_case(params_case, pixel_indices=None, eps_norm=0.01):
    if pixel_indices is None:
        x = GRID_X
        y = GRID_Y
        edge = GRID_EDGE
    else:
        x = GRID_X[pixel_indices]
        y = GRID_Y[pixel_indices]
        edge = GRID_EDGE[pixel_indices]

    n_points = len(x)
    active = params_case[:, 2] > 0
    comps = params_case[active]
    n_active = float(len(comps))
    p_total = float(comps[:, 2].sum()) if len(comps) else 0.0

    features = np.zeros((n_points, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = 1.0
    features[:, 1] = p_total
    features[:, 2] = n_active
    features[:, 3] = edge
    features[:, 4] = edge * edge
    features[:, 5] = p_total * n_active
    features[:, 6] = p_total * edge
    features[:, 7] = p_total * edge * edge

    if len(comps) == 0:
        return features

    sum_inv_r = np.zeros(n_points, dtype=np.float32)
    sum_inv_r2 = np.zeros(n_points, dtype=np.float32)
    sum_log_r = np.zeros(n_points, dtype=np.float32)
    sum_r = np.zeros(n_points, dtype=np.float32)
    sum_g03 = np.zeros(n_points, dtype=np.float32)
    sum_g06 = np.zeros(n_points, dtype=np.float32)
    sum_g10 = np.zeros(n_points, dtype=np.float32)
    sum_source_edge_g10 = np.zeros(n_points, dtype=np.float32)
    extra_sigmas = [0.015, 0.020, 0.040, 0.080, 0.140, 0.200]
    extra_gaussians = [np.zeros(n_points, dtype=np.float32) for _ in extra_sigmas]

    eps2 = eps_norm * eps_norm
    for x_mm, y_mm, power_w in comps:
        cx = float(x_mm) / BOARD_MM
        cy = float(y_mm) / BOARD_MM
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy + eps2
        r = np.sqrt(r2)
        source_edge = min(cx, cy, 1.0 - cx, 1.0 - cy)

        g03 = np.exp(-r2 / (2.0 * 0.03 * 0.03))
        g06 = np.exp(-r2 / (2.0 * 0.06 * 0.06))
        g10 = np.exp(-r2 / (2.0 * 0.10 * 0.10))
        extra_vals = [np.exp(-r2 / (2.0 * sigma * sigma)) for sigma in extra_sigmas]

        sum_inv_r += power_w / r
        sum_inv_r2 += power_w / r2
        sum_log_r += power_w * np.log(r)
        sum_r += power_w * r
        sum_g03 += power_w * g03
        sum_g06 += power_w * g06
        sum_g10 += power_w * g10
        sum_source_edge_g10 += power_w * source_edge * g10
        for arr, val in zip(extra_gaussians, extra_vals):
            arr += power_w * val

    features[:, 8] = sum_inv_r
    features[:, 9] = sum_inv_r2
    features[:, 10] = sum_log_r
    features[:, 11] = sum_r
    features[:, 12] = sum_g03
    features[:, 13] = sum_g06
    features[:, 14] = sum_g10
    features[:, 15] = edge * sum_inv_r
    features[:, 16] = edge * sum_g06
    features[:, 17] = sum_source_edge_g10
    features[:, 18] = n_active * sum_g06

    inv_total = 1.0 / max(p_total, 1e-6)
    features[:, 19] = sum_inv_r * inv_total
    features[:, 20] = sum_inv_r2 * inv_total
    features[:, 21] = sum_log_r * inv_total
    features[:, 22] = sum_r * inv_total
    features[:, 23] = sum_g03 * inv_total
    features[:, 24] = sum_g06 * inv_total
    features[:, 25] = sum_g10 * inv_total
    features[:, 26] = edge * sum_inv_r * inv_total
    features[:, 27] = edge * sum_g06 * inv_total
    features[:, 28] = sum_source_edge_g10 * inv_total
    features[:, 29] = n_active * sum_g06 * inv_total
    for offset, arr in enumerate(extra_gaussians):
        features[:, 30 + offset] = arr * inv_total
    for offset, arr in enumerate(extra_gaussians[2:]):
        features[:, 36 + offset] = edge * arr * inv_total
    return features


def sample_regression_rows(params, temps, case_indices, samples_per_case, rng, eps_norm, target):
    rows = []
    targets = []
    n_pixels = GRID * GRID
    replace = samples_per_case > n_pixels
    for case_idx in case_indices:
        pixel_idx = rng.choice(n_pixels, size=samples_per_case, replace=replace)
        rows.append(build_features_for_case(params[case_idx], pixel_idx, eps_norm=eps_norm))
        delta_t = temps[case_idx].reshape(-1)[pixel_idx] - T_AMB
        if target == "theta":
            total_power = max(float(params[case_idx, :, 2][params[case_idx, :, 2] > 0].sum()), 1e-6)
            y = delta_t / total_power
        elif target == "delta_t":
            y = delta_t
        else:
            raise ValueError(f"Unknown target: {target}")
        targets.append(y.astype(np.float32))
    return np.concatenate(rows, axis=0), np.concatenate(targets, axis=0)


def fit_symbolic_model(x_train, y_train, regressor_name, alpha, target, feature_set):
    feature_indices = FEATURE_SETS[feature_set]
    x_train = x_train[:, feature_indices]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    if regressor_name == "ridgecv":
        reg = RidgeCV(alphas=np.logspace(-6, 5, 16))
    elif regressor_name == "ridge":
        reg = Ridge(alpha=alpha)
    elif regressor_name == "lasso":
        reg = Lasso(alpha=alpha, max_iter=20000)
    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")
    reg.fit(x_scaled, y_train)
    return SymbolicModel(
        scaler=scaler,
        regressor=reg,
        feature_names=[FEATURE_NAMES[i] for i in feature_indices],
        feature_indices=feature_indices,
        eps_norm=0.01,
        target=target,
    )


def original_space_coefficients(model):
    coef_scaled = np.asarray(model.regressor.coef_, dtype=np.float64)
    scale = np.asarray(model.scaler.scale_, dtype=np.float64)
    mean = np.asarray(model.scaler.mean_, dtype=np.float64)
    coef = coef_scaled / scale
    intercept = float(model.regressor.intercept_ - np.sum(coef_scaled * mean / scale))
    return intercept, coef


def predict_case(model, params_case):
    features = build_features_for_case(params_case, eps_norm=model.eps_norm)
    pred_target = model.predict_target(features)
    if model.target == "theta":
        total_power = max(float(params_case[:, 2][params_case[:, 2] > 0].sum()), 1e-6)
        delta_t = pred_target * total_power
    else:
        delta_t = pred_target
    return (delta_t.reshape(GRID, GRID) + T_AMB).astype(np.float32)


def evaluate_cases(model, params, temps, indices, label, out_dir=None, max_plots=4):
    all_true = []
    all_pred = []
    r2_per = []

    plot_items = []
    for n_seen, case_idx in enumerate(indices):
        pred = predict_case(model, params[case_idx])
        true = temps[case_idx]
        all_true.append(true.reshape(-1))
        all_pred.append(pred.reshape(-1))
        r2_val = r2_score(true.reshape(-1), pred.reshape(-1))
        r2_per.append(float(r2_val))
        if len(plot_items) < max_plots:
            plot_items.append((true, pred, r2_val))

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    results = {
        "label": label,
        "n_cases": int(len(indices)),
        "r2_all_pixels": float(r2_score(all_true.ravel(), all_pred.ravel())),
        "r2_per_sample_mean": float(np.mean(r2_per)),
        "r2_per_sample_std": float(np.std(r2_per)),
        "r2_per_sample_min": float(np.min(r2_per)),
        "r2_per_sample_max": float(np.max(r2_per)),
    }
    print(
        f"{label}: R2_all={results['r2_all_pixels']:.4f}, "
        f"R2_mean={results['r2_per_sample_mean']:.4f}, "
        f"min={results['r2_per_sample_min']:.4f}"
    )

    if out_dir and plot_items:
        save_comparison_plot(plot_items, os.path.join(out_dir, f"{safe_name(label)}_comparison.png"))
    return results


def save_comparison_plot(items, save_path):
    fig, axes = plt.subplots(len(items), 3, figsize=(12, 3 * len(items)))
    if len(items) == 1:
        axes = axes[None, :]
    for row, (true, pred, r2_val) in enumerate(items):
        vmin = float(true.min())
        vmax = float(true.max())
        axes[row, 0].imshow(true.T, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title("True T")
        axes[row, 1].imshow(pred.T, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Pred T R2={r2_val:.3f}")
        err = pred - true
        im = axes[row, 2].imshow(err.T, cmap="coolwarm", origin="lower")
        axes[row, 2].set_title(f"Error MAE={np.abs(err).mean():.2f} C")
        plt.colorbar(im, ax=axes[row, 2])
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_gen_samples(n_comp, powers, gen_dir, max_cases=None):
    files = sorted(
        f for f in os.listdir(gen_dir)
        if f.startswith(f"count{n_comp}") and f.endswith(".json")
    )
    if max_cases is not None:
        files = files[:max_cases]
    if not files:
        return None, None

    params_list = []
    temps_list = []
    for filename in files:
        path = os.path.join(gen_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        temps = np.array([item["temperature"] for item in data], dtype=np.float32).reshape(GRID, GRID)

        stem = filename.replace(".json", "")
        tail = re.sub(r"^count\d+_idx\d+_", "", stem)
        nums = [int(v) for v in re.findall(r"-?\d+", tail)]
        if len(nums) < 2 * n_comp:
            raise ValueError(f"Could not parse {2*n_comp} coordinates from {filename}")

        pad_size = max(n_comp, 10)
        params = np.full((pad_size, 3), np.nan, dtype=np.float32)
        for comp_idx in range(n_comp):
            x_mm = nums[2 * comp_idx]
            y_mm = nums[2 * comp_idx + 1]
            params[comp_idx] = [x_mm, y_mm, powers[comp_idx]]

        params_list.append(params)
        temps_list.append(temps)

    return np.stack(params_list, axis=0), np.stack(temps_list, axis=0)


def evaluate_generalization(model, out_dir, gen_dir, max_gen_cases=None):
    results = {}
    for n_comp, powers, label in TEST_CONFIGS:
        params, temps = load_gen_samples(n_comp, powers, gen_dir, max_cases=max_gen_cases)
        if params is None:
            print(f"{label}: no data found")
            continue
        indices = np.arange(len(params))
        results[label] = evaluate_cases(
            model, params, temps, indices, label=f"gen_{label}", out_dir=out_dir, max_plots=2
        )
    return results


def save_formula(model, out_dir):
    intercept, coef = original_space_coefficients(model)
    terms = []
    for name, value in zip(model.feature_names, coef):
        terms.append({"name": name, "coefficient": float(value)})

    payload = {
        "target": model.target,
        "formula": "delta_T = intercept + sum_j coefficient_j * feature_j",
        "intercept": intercept,
        "terms": terms,
        "regressor": type(model.regressor).__name__,
        "alpha": float(getattr(model.regressor, "alpha_", getattr(model.regressor, "alpha", 0.0))),
        "eps_norm": model.eps_norm,
        "feature_notes": {
            "r_i": "normalized distance from query pixel (x,y) to component i",
            "edge": "normalized distance from query pixel to closest board edge",
            "s_i": "normalized distance from component i to closest board edge",
        },
    }
    with open(os.path.join(out_dir, "formula.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(os.path.join(out_dir, "symbolic_model.pkl"), "wb") as handle:
        pickle.dump(model, handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-inspired symbolic regression for temperature fields")
    parser.add_argument("--params", default="training_data/params_count_sweep.npy")
    parser.add_argument("--temps", default="training_data/temps_count_sweep.npy")
    parser.add_argument("--out-dir", default="my_scripts/results_symbolic_regression")
    parser.add_argument("--samples-per-case", type=int, default=800)
    parser.add_argument("--max-train-cases", type=int, default=0, help="0 means use all train cases")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regressor", choices=["ridgecv", "ridge", "lasso"], default="ridgecv")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--target", choices=["theta", "delta_t"], default="theta",
                        help="theta fits (T-Tamb)/P_total; delta_t fits T-Tamb directly")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="physics",
                        help="physics is conservative for count extrapolation; full uses all basis terms")
    parser.add_argument("--eps-mm", type=float, default=1.0)
    parser.add_argument("--skip-gen", action="store_true", help="skip generation_dataset extrapolation tests")
    parser.add_argument("--gen-dir", default=GEN_DIR)
    parser.add_argument("--max-gen-cases", type=int, default=0, help="0 means use all generated cases")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    eps_norm = args.eps_mm / BOARD_MM

    params, temps = load_npy_dataset(args.params, args.temps)
    counts = component_counts(params)
    all_indices = np.arange(len(params))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=args.test_ratio, random_state=args.seed, stratify=counts
    )
    train_counts = counts[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=args.val_ratio, random_state=args.seed, stratify=train_counts
    )
    if args.max_train_cases and args.max_train_cases < len(train_idx):
        train_idx = rng.choice(train_idx, size=args.max_train_cases, replace=False)

    print(f"Loaded params={params.shape}, temps={temps.shape}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Sampling {args.samples_per_case} pixels per training case")

    x_train, y_train = sample_regression_rows(
        params, temps, train_idx, args.samples_per_case, rng, eps_norm=eps_norm, target=args.target
    )
    print(f"Regression matrix: X={x_train.shape}, y={y_train.shape}")

    model = fit_symbolic_model(x_train, y_train, args.regressor, args.alpha, args.target, args.feature_set)
    model.eps_norm = eps_norm
    save_formula(model, args.out_dir)

    split_results = {
        "val": evaluate_cases(model, params, temps, val_idx, "val", out_dir=args.out_dir),
        "test": evaluate_cases(model, params, temps, test_idx, "test", out_dir=args.out_dir),
    }
    with open(os.path.join(args.out_dir, "split_results.json"), "w", encoding="utf-8") as handle:
        json.dump(split_results, handle, indent=2)

    gen_results = {}
    if not args.skip_gen:
        max_gen_cases = args.max_gen_cases if args.max_gen_cases > 0 else None
        gen_results = evaluate_generalization(model, args.out_dir, args.gen_dir, max_gen_cases=max_gen_cases)
        with open(os.path.join(args.out_dir, "generalization_results.json"), "w", encoding="utf-8") as handle:
            json.dump(gen_results, handle, indent=2)

    config = vars(args).copy()
    config["feature_names"] = model.feature_names
    config["feature_indices"] = model.feature_indices
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Done. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()