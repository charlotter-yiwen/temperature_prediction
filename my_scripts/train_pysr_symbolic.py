"""
train_pysr_symbolic.py
======================

Automatic symbolic expression discovery for PCB temperature fields using PySR.

This script is intentionally separate from train_symbolic_regression.py:

- train_symbolic_regression.py:
    predefined symbolic basis + Ridge/Lasso regression
- train_pysr_symbolic.py:
    automatic symbolic expression discovery over physics aggregate variables

The target is theta by default:

    theta(x, y) = (T(x, y) - T_amb) / P_total

The final temperature is restored as:

    T(x, y) = T_amb + P_total * theta(x, y)

PySR requires the Python package `pysr` and a working Julia installation.
Use --check-only to verify whether the current environment is ready.
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import train_symbolic_regression as base


PYSR_FEATURES = [
    ("p_total", 1),
    ("n_active", 2),
    ("edge", 3),
    ("edge2", 4),
    ("inv_r", 19),
    ("inv_r2", 20),
    ("log_r", 21),
    ("r_bar", 22),
    ("g015", 30),
    ("g020", 31),
    ("g030", 23),
    ("g040", 32),
    ("g060", 24),
    ("g080", 33),
    ("g100", 25),
    ("g140", 34),
    ("g200", 35),
    ("edge_inv_r", 26),
    ("edge_g040", 36),
    ("edge_g080", 37),
    ("edge_g140", 38),
    ("edge_g200", 39),
    ("p_max", 40),
    ("p_max_ratio", 41),
    ("p_std", 42),
    ("d_min", 43),
    ("d_mean", 44),
    ("pairwise_inv", 45),
    ("pairwise_inv2", 46),
    ("src_edge_min", 47),
    ("src_edge_mean", 48),
    ("src_edge_wmean", 49),
    ("src_edge_wmean2", 50),
    ("edge_pmax_ratio", 51),
    ("edge_d_min", 52),
    ("edge_pairwise", 53),
]

EXTENDED_FEATURE_NAMES = [
    "P_max",
    "P_max/P_total",
    "power_std",
    "d_min_pairwise",
    "d_mean_pairwise",
    "sum_ij w_i*w_j/(d_ij+eps)",
    "sum_ij w_i*w_j/(d_ij^2+eps^2)",
    "source_edge_min",
    "source_edge_mean",
    "sum_i w_i*source_edge_i",
    "sum_i w_i*source_edge_i^2",
    "edge*P_max/P_total",
    "edge*d_min_pairwise",
    "edge*pairwise_interaction",
]

GREEN_FEATURES = [
    "r",
    "r2",
    "inv_r",
    "inv_r2",
    "log_r",
    "pixel_edge",
    "pixel_edge2",
    "source_edge",
    "source_edge2",
    "min_pixel_source_edge",
    "pixel_edge_source_edge",
    "r_pixel_edge",
    "r_source_edge",
]

TMAX_FEATURES = [
    "n_active",
    "P_total",
    "P_max",
    "P_max_ratio",
    "power_mean",
    "power_std",
    "d_min_pairwise",
    "d_mean_pairwise",
    "pairwise_inv",
    "pairwise_inv2",
    "source_edge_min",
    "source_edge_mean",
    "source_edge_wmean",
    "source_edge_wmean2",
    "P_total_d_min_pairwise",
    "P_total_pairwise_inv",
]


@dataclass
class PySRTemperatureModel:
    model: object
    variable_names: list
    feature_indices: list
    eps_norm: float
    target: str
    field_mode: str = "aggregate"

    def predict_target(self, feature_matrix):
        selected = feature_matrix[:, self.feature_indices]
        return np.asarray(self.model.predict(selected), dtype=np.float32).reshape(-1)


@dataclass
class PySRTmaxModel:
    model: object
    variable_names: list
    eps_norm: float

    def predict(self, case_features):
        return np.asarray(self.model.predict(case_features), dtype=np.float32).reshape(-1)


def import_pysr():
    try:
        from pysr import PySRRegressor
    except Exception as exc:
        raise RuntimeError(
            "PySR is not available in this Python environment. Install PySR and Julia first. "
            "Typical steps: pip install pysr, install Julia, then run python -c \"import pysr\". "
            f"Original import error: {exc}"
        ) from exc
    return PySRRegressor


def parse_operator_list(text):
    if text is None:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def select_features(feature_matrix, feature_spec):
    indices = [idx for _, idx in feature_spec]
    names = [name for name, _ in feature_spec]
    return feature_matrix[:, indices], names, indices


def component_stats(params_case, eps_norm=0.01):
    active = params_case[:, 2] > 0
    comps = params_case[active]
    if len(comps) == 0:
        return {
            "n_active": 0.0,
            "p_total": 0.0,
            "p_max": 0.0,
            "p_max_ratio": 0.0,
            "p_mean": 0.0,
            "p_std": 0.0,
            "d_min": 1.0,
            "d_mean": 1.0,
            "pairwise_inv": 0.0,
            "pairwise_inv2": 0.0,
            "source_edge_min": 0.0,
            "source_edge_mean": 0.0,
            "source_edge_wmean": 0.0,
            "source_edge_wmean2": 0.0,
        }

    powers = comps[:, 2].astype(np.float32)
    p_total = max(float(powers.sum()), 1e-6)
    weights = powers / p_total
    xs = comps[:, 0].astype(np.float32) / base.BOARD_MM
    ys = comps[:, 1].astype(np.float32) / base.BOARD_MM
    source_edges = np.minimum.reduce([xs, ys, 1.0 - xs, 1.0 - ys]).astype(np.float32)

    distances = []
    pairwise_inv = 0.0
    pairwise_inv2 = 0.0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = float(dx * dx + dy * dy + eps_norm * eps_norm)
            d = float(np.sqrt(d2))
            distances.append(d)
            wij = float(weights[i] * weights[j])
            pairwise_inv += wij / d
            pairwise_inv2 += wij / d2

    if distances:
        d_min = float(np.min(distances))
        d_mean = float(np.mean(distances))
    else:
        d_min = 1.0
        d_mean = 1.0

    return {
        "n_active": float(len(comps)),
        "p_total": p_total,
        "p_max": float(powers.max()),
        "p_max_ratio": float(powers.max() / p_total),
        "p_mean": float(powers.mean()),
        "p_std": float(powers.std()),
        "d_min": d_min,
        "d_mean": d_mean,
        "pairwise_inv": float(pairwise_inv),
        "pairwise_inv2": float(pairwise_inv2),
        "source_edge_min": float(source_edges.min()),
        "source_edge_mean": float(source_edges.mean()),
        "source_edge_wmean": float(np.sum(weights * source_edges)),
        "source_edge_wmean2": float(np.sum(weights * source_edges * source_edges)),
    }


def build_extended_features_for_case(params_case, pixel_indices=None, eps_norm=0.01):
    base_features = base.build_features_for_case(params_case, pixel_indices=pixel_indices, eps_norm=eps_norm)
    edge = base_features[:, 3]
    stats = component_stats(params_case, eps_norm=eps_norm)
    n_points = base_features.shape[0]
    extra = np.zeros((n_points, len(EXTENDED_FEATURE_NAMES)), dtype=np.float32)
    extra[:, 0] = stats["p_max"]
    extra[:, 1] = stats["p_max_ratio"]
    extra[:, 2] = stats["p_std"]
    extra[:, 3] = stats["d_min"]
    extra[:, 4] = stats["d_mean"]
    extra[:, 5] = stats["pairwise_inv"]
    extra[:, 6] = stats["pairwise_inv2"]
    extra[:, 7] = stats["source_edge_min"]
    extra[:, 8] = stats["source_edge_mean"]
    extra[:, 9] = stats["source_edge_wmean"]
    extra[:, 10] = stats["source_edge_wmean2"]
    extra[:, 11] = edge * stats["p_max_ratio"]
    extra[:, 12] = edge * stats["d_min"]
    extra[:, 13] = edge * stats["pairwise_inv"]
    return np.concatenate([base_features, extra], axis=1)


def sample_rows(params, temps, case_indices, samples_per_case, rng, eps_norm, target, feature_spec):
    rows = []
    targets = []
    n_pixels = base.GRID * base.GRID
    replace = samples_per_case > n_pixels
    for case_idx in case_indices:
        pixel_idx = rng.choice(n_pixels, size=samples_per_case, replace=replace)
        rows.append(build_extended_features_for_case(params[case_idx], pixel_indices=pixel_idx, eps_norm=eps_norm))
        delta_t = temps[case_idx].reshape(-1)[pixel_idx] - base.T_AMB
        if target == "theta":
            y = delta_t / total_power(params[case_idx])
        elif target == "delta_t":
            y = delta_t
        else:
            raise ValueError(f"Unknown target: {target}")
        targets.append(y.astype(np.float32))
    x_all = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)
    x, names, indices = select_features(x_all, feature_spec)
    return x.astype(np.float32), y.astype(np.float32), names, indices


def build_green_features_for_component(x_mm, y_mm, pixel_indices=None, eps_norm=0.01):
    if pixel_indices is None:
        x = base.GRID_X
        y = base.GRID_Y
        pixel_edge = base.GRID_EDGE
    else:
        x = base.GRID_X[pixel_indices]
        y = base.GRID_Y[pixel_indices]
        pixel_edge = base.GRID_EDGE[pixel_indices]

    cx = float(x_mm) / base.BOARD_MM
    cy = float(y_mm) / base.BOARD_MM
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy + eps_norm * eps_norm
    r = np.sqrt(r2)
    source_edge = min(cx, cy, 1.0 - cx, 1.0 - cy)

    features = np.zeros((len(x), len(GREEN_FEATURES)), dtype=np.float32)
    features[:, 0] = r
    features[:, 1] = r2
    features[:, 2] = 1.0 / r
    features[:, 3] = 1.0 / r2
    features[:, 4] = np.log(r)
    features[:, 5] = pixel_edge
    features[:, 6] = pixel_edge * pixel_edge
    features[:, 7] = source_edge
    features[:, 8] = source_edge * source_edge
    features[:, 9] = np.minimum(pixel_edge, source_edge)
    features[:, 10] = pixel_edge * source_edge
    features[:, 11] = r * pixel_edge
    features[:, 12] = r * source_edge
    return features


def sample_green_rows(params, temps, case_indices, samples_per_case, rng, eps_norm):
    rows = []
    targets = []
    n_pixels = base.GRID * base.GRID
    replace = samples_per_case > n_pixels
    for case_idx in case_indices:
        active = params[case_idx, :, 2] > 0
        if int(active.sum()) != 1:
            continue
        comp = params[case_idx, active][0]
        pixel_idx = rng.choice(n_pixels, size=samples_per_case, replace=replace)
        rows.append(build_green_features_for_component(comp[0], comp[1], pixel_idx, eps_norm=eps_norm))
        delta_t = temps[case_idx].reshape(-1)[pixel_idx] - base.T_AMB
        targets.append((delta_t / max(float(comp[2]), 1e-6)).astype(np.float32))
    if not rows:
        raise ValueError("Green mode needs at least one single-component training case")
    return np.concatenate(rows, axis=0), np.concatenate(targets, axis=0)


def build_tmax_features_for_case(params_case, eps_norm=0.01):
    stats = component_stats(params_case, eps_norm=eps_norm)
    values = [
        stats["n_active"],
        stats["p_total"],
        stats["p_max"],
        stats["p_max_ratio"],
        stats["p_mean"],
        stats["p_std"],
        stats["d_min"],
        stats["d_mean"],
        stats["pairwise_inv"],
        stats["pairwise_inv2"],
        stats["source_edge_min"],
        stats["source_edge_mean"],
        stats["source_edge_wmean"],
        stats["source_edge_wmean2"],
        stats["p_total"] * stats["d_min"],
        stats["p_total"] * stats["pairwise_inv"],
    ]
    return np.asarray(values, dtype=np.float32)


def build_tmax_matrix(params, temps, case_indices):
    x = np.stack([build_tmax_features_for_case(params[idx]) for idx in case_indices], axis=0)
    y = np.asarray([temps[idx].max() for idx in case_indices], dtype=np.float32)
    return x, y


def build_regressor(args, variable_names):
    PySRRegressor = import_pysr()
    binary_ops = parse_operator_list(args.binary_ops)
    unary_ops = parse_operator_list(args.unary_ops)
    kwargs = {
        "niterations": args.niterations,
        "binary_operators": binary_ops,
        "unary_operators": unary_ops,
        "model_selection": args.model_selection,
        "maxsize": args.maxsize,
        "populations": args.populations,
        "random_state": args.seed,
        "verbosity": args.verbosity,
    }
    try:
        return PySRRegressor(**kwargs)
    except TypeError:
        # Older PySR versions do not support every keyword above.
        kwargs.pop("random_state", None)
        return PySRRegressor(**kwargs)


def total_power(params_case):
    active = params_case[:, 2] > 0
    return max(float(params_case[active, 2].sum()), 1e-6)


def predict_case(model_bundle, params_case):
    if model_bundle.field_mode == "green":
        delta_t = np.zeros(base.GRID * base.GRID, dtype=np.float32)
        active = params_case[:, 2] > 0
        for x_mm, y_mm, power_w in params_case[active]:
            green_features = build_green_features_for_component(
                x_mm, y_mm, pixel_indices=None, eps_norm=model_bundle.eps_norm
            )
            delta_t += float(power_w) * np.asarray(model_bundle.model.predict(green_features), dtype=np.float32).reshape(-1)
    else:
        features = build_extended_features_for_case(params_case, eps_norm=model_bundle.eps_norm)
        pred_target = model_bundle.predict_target(features)
        if model_bundle.target == "theta":
            delta_t = pred_target * total_power(params_case)
        else:
            delta_t = pred_target
    return (delta_t.reshape(base.GRID, base.GRID) + base.T_AMB).astype(np.float32)


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
        axes[row, 1].set_title(f"PySR Pred R2={r2_val:.3f}")
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


def evaluate_cases(model_bundle, params, temps, indices, label, out_dir=None, max_plots=4):
    all_true = []
    all_pred = []
    r2_per = []
    plot_items = []

    for case_idx in indices:
        pred = predict_case(model_bundle, params[case_idx])
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
        save_comparison_plot(plot_items, os.path.join(out_dir, f"{base.safe_name(label)}_comparison.png"))
    return results


def evaluate_generalization(model_bundle, out_dir, gen_dir, max_gen_cases=None):
    results = {}
    for n_comp, powers, label in base.TEST_CONFIGS:
        params, temps = base.load_gen_samples(n_comp, powers, gen_dir, max_cases=max_gen_cases)
        if params is None:
            print(f"{label}: no data found")
            continue
        results[label] = evaluate_cases(
            model_bundle,
            params,
            temps,
            np.arange(len(params)),
            label=f"pysr_gen_{label}",
            out_dir=out_dir,
            max_plots=2,
        )
    return results


def evaluate_tmax_cases(model_bundle, params, temps, indices, label):
    x = np.stack([build_tmax_features_for_case(params[idx], eps_norm=model_bundle.eps_norm) for idx in indices], axis=0)
    true = np.asarray([temps[idx].max() for idx in indices], dtype=np.float32)
    pred = model_bundle.predict(x)
    results = {
        "label": label,
        "n_cases": int(len(indices)),
        "r2": float(r2_score(true, pred)),
        "mae": float(np.mean(np.abs(pred - true))),
        "true_min": float(true.min()),
        "true_max": float(true.max()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
    }
    print(f"{label}: R2={results['r2']:.4f}, MAE={results['mae']:.3f} C")
    return results


def evaluate_tmax_generalization(model_bundle, gen_dir, max_gen_cases=None):
    results = {}
    for n_comp, powers, label in base.TEST_CONFIGS:
        params, temps = base.load_gen_samples(n_comp, powers, gen_dir, max_cases=max_gen_cases)
        if params is None:
            print(f"{label}: no data found")
            continue
        results[label] = evaluate_tmax_cases(
            model_bundle, params, temps, np.arange(len(params)), f"tmax_gen_{label}"
        )
    return results


def save_equations(model, out_dir):
    equations = getattr(model, "equations_", None)
    if equations is None:
        return None
    csv_path = os.path.join(out_dir, "pysr_equations.csv")
    json_path = os.path.join(out_dir, "pysr_equations.json")
    try:
        equations.to_csv(csv_path, index=False)
        safe_records = []
        for record in equations.to_dict(orient="records"):
            safe_records.append({key: str(value) for key, value in record.items()})
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(safe_records, handle, indent=2)
    except Exception as exc:
        with open(os.path.join(out_dir, "pysr_equations_error.txt"), "w", encoding="utf-8") as handle:
            handle.write(str(exc))
    return csv_path


def read_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_comparison_summary(args, out_dir, split_results, gen_results):
    previous_pysr_split = read_json_if_exists(os.path.join(args.previous_pysr_dir, "split_results.json"))
    previous_pysr_gen = read_json_if_exists(os.path.join(args.previous_pysr_dir, "generalization_results.json"))
    baseline_split = read_json_if_exists(os.path.join(args.baseline_dir, "split_results.json"))
    baseline_gen = read_json_if_exists(os.path.join(args.baseline_dir, "generalization_results.json"))
    v3_distill = read_json_if_exists(os.path.join(args.v3_distill_dir, "test_results", "results.json"))

    summary = {
        "improved_pysr_symbolic": {
            "task": args.task,
            "split": split_results,
            "generalization": gen_results,
        },
        "previous_pysr_symbolic": {
            "split": previous_pysr_split,
            "generalization": previous_pysr_gen,
            "source_dir": args.previous_pysr_dir,
        },
        "predefined_symbolic_baseline": {
            "split": baseline_split,
            "generalization": baseline_gen,
            "source_dir": args.baseline_dir,
        },
        "v3_distill": {
            "test_results": v3_distill,
            "source_dir": args.v3_distill_dir,
        },
    }
    with open(os.path.join(out_dir, "comparison_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="PySR automatic symbolic regression for temperature fields")
    parser.add_argument("--params", default="training_data/params_count_sweep.npy")
    parser.add_argument("--temps", default="training_data/temps_count_sweep.npy")
    parser.add_argument("--out-dir", default="my_scripts/results_pysr_symbolic")
    parser.add_argument("--samples-per-case", type=int, default=200)
    parser.add_argument("--max-train-cases", type=int, default=120, help="0 means use all train cases")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["theta", "delta_t"], default="theta")
    parser.add_argument("--task", choices=["aggregate", "green", "tmax"], default="aggregate",
                        help="aggregate predicts full-field theta from aggregate features; green learns G(r,boundary); tmax predicts max temperature only")
    parser.add_argument("--eps-mm", type=float, default=1.0)
    parser.add_argument("--niterations", type=int, default=150)
    parser.add_argument("--populations", type=int, default=20)
    parser.add_argument("--maxsize", type=int, default=28)
    parser.add_argument("--model-selection", choices=["accuracy", "best", "score"], default="best")
    parser.add_argument("--binary-ops", default="+,-,*,/", help="comma-separated PySR binary operators")
    parser.add_argument("--unary-ops", default="", help="comma-separated PySR unary operators; default keeps only +,-,*,/")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--gen-dir", default=base.GEN_DIR)
    parser.add_argument("--max-gen-cases", type=int, default=0, help="0 means use all generated cases")
    parser.add_argument("--previous-pysr-dir", default="my_scripts/results_pysr_symbolic",
                        help="previous/current PySR result dir for comparison_summary.json")
    parser.add_argument("--baseline-dir", default="my_scripts/results_symbolic_regression")
    parser.add_argument("--v3-distill-dir", default="my_scripts/results_v3_distill")
    parser.add_argument("--check-only", action="store_true", help="only verify PySR import availability")
    parser.add_argument("--verbosity", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.check_only:
        import_pysr()
        print("PySR import check passed.")
        return

    rng = np.random.default_rng(args.seed)
    eps_norm = args.eps_mm / base.BOARD_MM

    params, temps = base.load_npy_dataset(args.params, args.temps)
    counts = base.component_counts(params)
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
    print(f"Task: {args.task}")
    print(f"Binary operators: {args.binary_ops}")
    print(f"Unary operators: {args.unary_ops or '(none)'}")

    if args.task == "aggregate":
        print(f"Sampling {args.samples_per_case} pixels per training case for aggregate theta PySR")
        x_train, y_train, variable_names, feature_indices = sample_rows(
            params=params,
            temps=temps,
            case_indices=train_idx,
            samples_per_case=args.samples_per_case,
            rng=rng,
            eps_norm=eps_norm,
            target=args.target,
            feature_spec=PYSR_FEATURES,
        )
        print(f"PySR matrix: X={x_train.shape}, y={y_train.shape}")
        print(f"Variables: {', '.join(variable_names)}")
        regressor = build_regressor(args, variable_names)
        try:
            regressor.fit(x_train, y_train, variable_names=variable_names)
        except TypeError:
            regressor.fit(x_train, y_train)
        save_equations(regressor, args.out_dir)
        model_bundle = PySRTemperatureModel(
            model=regressor,
            variable_names=variable_names,
            feature_indices=feature_indices,
            eps_norm=eps_norm,
            target=args.target,
            field_mode="aggregate",
        )
        model_filename = "pysr_model.pkl"
        split_results = {
            "val": evaluate_cases(model_bundle, params, temps, val_idx, "pysr_val", out_dir=args.out_dir),
            "test": evaluate_cases(model_bundle, params, temps, test_idx, "pysr_test", out_dir=args.out_dir),
        }
        gen_results = {}
        if not args.skip_gen:
            max_gen_cases = args.max_gen_cases if args.max_gen_cases > 0 else None
            gen_results = evaluate_generalization(model_bundle, args.out_dir, args.gen_dir, max_gen_cases=max_gen_cases)

    elif args.task == "green":
        print(f"Sampling {args.samples_per_case} pixels per single-component case for Green-function PySR")
        x_train, y_train = sample_green_rows(
            params=params,
            temps=temps,
            case_indices=train_idx,
            samples_per_case=args.samples_per_case,
            rng=rng,
            eps_norm=eps_norm,
        )
        variable_names = GREEN_FEATURES
        feature_indices = []
        print(f"Green PySR matrix: X={x_train.shape}, y={y_train.shape}")
        print(f"Variables: {', '.join(variable_names)}")
        regressor = build_regressor(args, variable_names)
        try:
            regressor.fit(x_train, y_train, variable_names=variable_names)
        except TypeError:
            regressor.fit(x_train, y_train)
        save_equations(regressor, args.out_dir)
        model_bundle = PySRTemperatureModel(
            model=regressor,
            variable_names=variable_names,
            feature_indices=feature_indices,
            eps_norm=eps_norm,
            target="green",
            field_mode="green",
        )
        model_filename = "pysr_green_model.pkl"
        split_results = {
            "val": evaluate_cases(model_bundle, params, temps, val_idx, "pysr_green_val", out_dir=args.out_dir),
            "test": evaluate_cases(model_bundle, params, temps, test_idx, "pysr_green_test", out_dir=args.out_dir),
        }
        gen_results = {}
        if not args.skip_gen:
            max_gen_cases = args.max_gen_cases if args.max_gen_cases > 0 else None
            gen_results = evaluate_generalization(model_bundle, args.out_dir, args.gen_dir, max_gen_cases=max_gen_cases)

    elif args.task == "tmax":
        print("Training PySR for scalar T_max prediction")
        x_train, y_train = build_tmax_matrix(params, temps, train_idx)
        variable_names = TMAX_FEATURES
        feature_indices = []
        print(f"Tmax PySR matrix: X={x_train.shape}, y={y_train.shape}")
        print(f"Variables: {', '.join(variable_names)}")
        regressor = build_regressor(args, variable_names)
        try:
            regressor.fit(x_train, y_train, variable_names=variable_names)
        except TypeError:
            regressor.fit(x_train, y_train)
        save_equations(regressor, args.out_dir)
        model_bundle = PySRTmaxModel(
            model=regressor,
            variable_names=variable_names,
            eps_norm=eps_norm,
        )
        model_filename = "pysr_tmax_model.pkl"
        split_results = {
            "val": evaluate_tmax_cases(model_bundle, params, temps, val_idx, "pysr_tmax_val"),
            "test": evaluate_tmax_cases(model_bundle, params, temps, test_idx, "pysr_tmax_test"),
        }
        gen_results = {}
        if not args.skip_gen:
            max_gen_cases = args.max_gen_cases if args.max_gen_cases > 0 else None
            gen_results = evaluate_tmax_generalization(model_bundle, args.gen_dir, max_gen_cases=max_gen_cases)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    with open(os.path.join(args.out_dir, model_filename), "wb") as handle:
        pickle.dump(model_bundle, handle)

    with open(os.path.join(args.out_dir, "split_results.json"), "w", encoding="utf-8") as handle:
        json.dump(split_results, handle, indent=2)
    with open(os.path.join(args.out_dir, "generalization_results.json"), "w", encoding="utf-8") as handle:
        json.dump(gen_results, handle, indent=2)

    config = vars(args).copy()
    config["variable_names"] = variable_names
    config["feature_indices"] = feature_indices
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    write_comparison_summary(args, args.out_dir, split_results, gen_results)
    print(f"Done. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()