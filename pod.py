import argparse
import datetime
import os
import numpy as np
from sklearn.metrics import r2_score
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM

#python pod.py --pred-out preds.npy
def load_arrays(train_params_path, train_temps_path, test_params_path, test_temps_path):
    params_train = np.load(train_params_path)
    temps_train = np.load(train_temps_path)
    params_test = np.load(test_params_path)
    temps_test = np.load(test_temps_path)
    return params_train, temps_train, params_test, temps_test


def build_and_fit_rom(params_train, temps_train, svd_method='randomized_svd', rank=None, rbf_kernel='auto'):
    pod = POD(svd_method) if rank is None else POD(svd_method, rank=rank)
    rbf = RBF() if rbf_kernel == 'auto' else RBF(rbf_kernel)
    db = Database(params_train, temps_train)
    rom = ROM(db, pod, rbf)
    rom.fit()
    return rom


def predict_rom(rom, params_test):
    preds = []
    for param in params_test:
        res = rom.predict(param)
        # ezyrb ROM.predict may return a structure with snapshots_matrix or a raw ndarray
        snap = res.snapshots_matrix if hasattr(res, 'snapshots_matrix') else res
        preds.append(np.array(snap).reshape(-1))
    return np.vstack(preds)


def compute_r2(preds, temps_test):
    r2_values = []
    for i in range(preds.shape[0]):
        true_flat = temps_test[i].flatten()
        pred_flat = preds[i].flatten()
        mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
        if not np.any(mask):
            r2_values.append(np.nan)
            continue
        r2 = r2_score(true_flat[mask], pred_flat[mask])
        r2_values.append(r2)
    return np.array(r2_values)


def main():
    parser = argparse.ArgumentParser(description="POD-RBF ROM inference and R2 evaluation")
    parser.add_argument('--train-params', default='thermal_analysis_output/training data/params_training.npy', help='Path to training parameters npy')
    parser.add_argument('--train-temps', default='thermal_analysis_output/training data/temps_training.npy', help='Path to training temperatures npy')
    parser.add_argument('--test-params', default='thermal_analysis_output/test data/params_testing.npy', help='Path to testing parameters npy')
    parser.add_argument('--test-temps', default='thermal_analysis_output/test data/temps_testing.npy', help='Path to testing temperatures npy')
    parser.add_argument('--pred-out', default=None, help='Optional path to save predicted snapshots npy')
    parser.add_argument('--svd-method', default='randomized_svd', choices=['svd', 'randomized_svd'], help='POD SVD method')
    parser.add_argument('--rank', type=int, default=None, help='Number of POD modes to retain (None = auto)')
    parser.add_argument('--rbf-kernel', default='auto', choices=['auto', 'multiquadric', 'gaussian', 'linear', 'thin_plate_spline', 'cubic'], help='RBF kernel function (auto = ezyrb default)')
    args = parser.parse_args()

    params_train, temps_train, params_test, temps_test = load_arrays(
        args.train_params, args.train_temps, args.test_params, args.test_temps)

    if params_train.shape[0] != temps_train.shape[0]:
        raise ValueError('Training params and temps must have the same number of samples')
    if params_test.shape[0] != temps_test.shape[0]:
        raise ValueError('Testing params and temps must have the same number of samples')

    rom = build_and_fit_rom(params_train, temps_train,
                            svd_method=args.svd_method,
                            rank=args.rank,
                            rbf_kernel=args.rbf_kernel)
    preds = predict_rom(rom, params_test)

    if preds.shape != temps_test.shape:
        print(f"Warning: predicted shape {preds.shape} != test shape {temps_test.shape}; proceeding with flatten comparison")

    r2_values = compute_r2(preds, temps_test)
    print(f"Per-sample R2: {r2_values}")
    finite_mask = np.isfinite(r2_values)
    if np.any(finite_mask):
        avg_r2 = np.mean(r2_values[finite_mask])
        print(f"Average R2 (finite): {avg_r2}")
    else:
        avg_r2 = float('nan')
        print("Average R2: all NaN (no finite samples)")

    if args.pred_out:
        np.save(args.pred_out, preds)
        print(f"Saved predictions to {args.pred_out}")

    # ---------- Write training log ----------
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_log.md')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_train = params_train.shape[0]
    n_test  = params_test.shape[0]
    n_nan   = int(np.sum(~finite_mask))
    n_valid = int(np.sum(finite_mask))
    min_r2  = float(np.nanmin(r2_values)) if n_valid > 0 else float('nan')
    max_r2  = float(np.nanmax(r2_values)) if n_valid > 0 else float('nan')

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## POD-RBF Run — {timestamp}\n")
    lines.append(f"\n### Hyperparameters\n")
    lines.append(f"| Param | Value |\n|---|---|\n")
    lines.append(f"| SVD method  | {args.svd_method} |\n")
    lines.append(f"| POD rank    | {args.rank if args.rank is not None else 'auto'} |\n")
    lines.append(f"| RBF kernel  | {args.rbf_kernel} |\n")
    lines.append(f"\n### Dataset\n")
    lines.append(f"| Item | Value |\n|---|---|\n")
    lines.append(f"| Train samples | {n_train} |\n")
    lines.append(f"| Test samples  | {n_test}  |\n")
    lines.append(f"| Param dim     | {params_train.shape[1]} |\n")
    lines.append(f"| Temp field size | {temps_train.shape[1:]} |\n")
    lines.append(f"\n### Results\n")
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| Average R² (finite) | {avg_r2:.6f} |\n")
    lines.append(f"| Max R²              | {max_r2:.6f} |\n")
    lines.append(f"| Min R²              | {min_r2:.6f} |\n")
    lines.append(f"| Valid samples       | {n_valid}/{n_test} |\n")
    lines.append(f"| NaN samples         | {n_nan} |\n")
    lines.append(f"\n### Per-sample R²\n\n")
    lines.append(f"| Sample | R² |\n|---|---|\n")
    for i, r2 in enumerate(r2_values):
        val = f"{r2:.6f}" if np.isfinite(r2) else "NaN"
        lines.append(f"| {i+1:02d} | {val} |\n")
    lines.append(f"\n### File Paths\n")
    lines.append(f"- Train params: `{args.train_params}`\n")
    lines.append(f"- Train temps:  `{args.train_temps}`\n")
    lines.append(f"- Test params:  `{args.test_params}`\n")
    lines.append(f"- Test temps:   `{args.test_temps}`\n")
    if args.pred_out:
        lines.append(f"- Predictions saved: `{args.pred_out}`\n")

    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as lf:
        if write_header:
            lf.write('# Training Log\n')
        lf.writelines(lines)
    print(f"Log written to {log_path}")


if __name__ == '__main__':
    main()
