"""
hyperparam_search.py
====================
超参数搜索 - 寻找最佳模型配置
"""

import subprocess
import json
import os
import numpy as np

# 参数搜索空间
PARAM_GRID = [
    # 基准配置
    {
        "name": "baseline",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 12, "n_fno": 4,
        "lr": 5e-4, "batch_size": 8, "dropout": 0.0,
    },
    # 大学习率 + 大batch
    {
        "name": "large_lr_big_batch",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 12, "n_fno": 4,
        "lr": 1e-3, "batch_size": 16, "dropout": 0.0,
    },
    # 小学习率 + 小batch
    {
        "name": "small_lr_small_batch",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 12, "n_fno": 4,
        "lr": 1e-4, "batch_size": 4, "dropout": 0.0,
    },
    # 大模型
    {
        "name": "big_model",
        "d_model": 256, "num_heads": 8, "n_sab": 3,
        "fno_ch": 64, "fno_modes": 16, "n_fno": 6,
        "lr": 5e-4, "batch_size": 8, "dropout": 0.0,
    },
    # 小模型 (防止过拟合)
    {
        "name": "small_model",
        "d_model": 64, "num_heads": 4, "n_sab": 1,
        "fno_ch": 16, "fno_modes": 8, "n_fno": 2,
        "lr": 5e-4, "batch_size": 8, "dropout": 0.1,
    },
    # 更多FNO modes
    {
        "name": "more_fno_modes",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 20, "n_fno": 4,
        "lr": 5e-4, "batch_size": 8, "dropout": 0.0,
    },
    # dropout正则化
    {
        "name": "dropout_reg",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 12, "n_fno": 4,
        "lr": 5e-4, "batch_size": 8, "dropout": 0.2,
    },
    # 小学习率长训练
    {
        "name": "slow_learning",
        "d_model": 128, "num_heads": 4, "n_sab": 2,
        "fno_ch": 32, "fno_modes": 12, "n_fno": 4,
        "lr": 1e-4, "batch_size": 8, "dropout": 0.0,
    },
]

RESULTS_FILE = "hyperparam_results.json"


def run_experiment(params, base_dir, epochs=500, n_runs=2):
    """运行一次实验"""
    out_dir = f"hp_search/{params['name']}"
    python_exe = r"C:\anaconda3\envs\magnet2\python.exe"
    script_path = os.path.join(base_dir, "models", "set_fno_thermal.py")
    params_path = os.path.join(base_dir, "training_data", "params_count_sweep.npy")
    temps_path = os.path.join(base_dir, "training_data", "temps_count_sweep.npy")

    cmd = [
        python_exe,
        script_path,
        "--count-sweep-params", params_path,
        "--count-sweep-temps", temps_path,
        "--n-components", "5",
        "--d-per-comp", "3",
        "--epochs", str(epochs),
        "--batch-size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--d-model", str(params["d_model"]),
        "--num-heads", str(params["num_heads"]),
        "--n-sab", str(params["n_sab"]),
        "--fno-ch", str(params["fno_ch"]),
        "--fno-modes", str(params["fno_modes"]),
        "--n-fno", str(params["n_fno"]),
        "--dropout", str(params["dropout"]),
        "--val-ratio", "0.1",
        "--physics-norm",
        "--t-ambient", "25.0",
        "--out-dir", out_dir,
        "--log-every", "100",
        "--n-runs", str(n_runs),
        "--seed-base", "42",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {params['name']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None

    # 读取结果
    summary_path = os.path.join(base_dir, out_dir, "multi_run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return {
            "name": params["name"],
            "r2_mean": summary["r2_mean"],
            "r2_std": summary["r2_std"],
            "params": {k: v for k, v in params.items() if k != "name"},
        }
    return None


def main():
    base_dir = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction"
    os.makedirs(os.path.join(base_dir, "hp_search"), exist_ok=True)

    all_results = []

    for params in PARAM_GRID:
        result = run_experiment(params, base_dir=base_dir, epochs=500, n_runs=2)
        if result:
            all_results.append(result)
            print(f"\n>>> {params['name']}: R² = {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # 按R²排序
    all_results.sort(key=lambda x: x["r2_mean"], reverse=True)

    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH RESULTS (sorted by R²)")
    print("="*70)
    for i, r in enumerate(all_results):
        print(f"{i+1}. {r['name']}: R² = {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")

    # 保存结果
    results_path = os.path.join(base_dir, "hp_search", RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"\nBest configuration: {all_results[0]['name']}")
    print(f"Best R²: {all_results[0]['r2_mean']:.4f}")


if __name__ == "__main__":
    main()