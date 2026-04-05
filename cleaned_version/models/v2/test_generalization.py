"""
test_generalization.py
==============================
测试模型在 6、7、8 组件数据上的泛化能力。

用法:
  python test_generalization.py --model-path ../results_setfno_phase2/setfno_30w_phase2.pth \
      --data-dir ../../training_data_30W_generalization \
      --component-count 6
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 添加父目录到路径
TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TP_DIR)

from models.set_fno_thermal import SetFNOModel
from model_30w.train_setfno_30w import load_30w_dataset, inverse_transform_temps


def plot_r2_bar(r2_scores, output_dir, component_count):
    """绘制 R² 分数柱状图"""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(r2_scores)), r2_scores, color='steelblue')
    plt.axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f'Mean R²={np.mean(r2_scores):.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('R² Score')
    plt.title(f'R² Scores for {component_count} Components')
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'r2_bar_{component_count}comp.png'), dpi=150)
    plt.close()


def plot_scatter(preds, targets, output_dir, component_count):
    """绘制预测vs真实散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(targets.flatten(), preds.flatten(), alpha=0.3, s=1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title(f'Prediction vs Truth ({component_count} Components)')
    r2 = r2_score(targets.flatten(), preds.flatten())
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_{component_count}comp.png'), dpi=150)
    plt.close()


def plot_thermal_comparisons(preds, targets, output_dir, component_count, n_samples=6):
    """绘制温度图对比"""
    n_samples = min(n_samples, len(preds))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

    for i in range(n_samples):
        # 真实温度图
        im1 = axes[i, 0].imshow(targets[i], cmap='hot')
        axes[i, 0].set_title(f'Sample {i+1} - Ground Truth')
        plt.colorbar(im1, ax=axes[i, 0])

        # 预测温度图
        im2 = axes[i, 1].imshow(preds[i], cmap='hot')
        axes[i, 1].set_title(f'Sample {i+1} - Prediction')
        plt.colorbar(im2, ax=axes[i, 1])

        # 误差图
        error = preds[i] - targets[i]
        im3 = axes[i, 2].imshow(error, cmap='RdBu', vmin=-10, vmax=10)
        axes[i, 2].set_title(f'Sample {i+1} - Error')
        plt.colorbar(im3, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'thermal_comparison_{component_count}comp.png'), dpi=150)
    plt.close()


def evaluate_generalization(model, params, powers, temps, scaler, device, T_ambient, component_count, output_dir):
    """
    评估模型在泛化数据上的表现。

    参数说明:
        params: (N, max_components, 3) - 已经归一化的参数 [x_norm, y_norm, power_norm]
        powers: (N,) - 每个样本的总功率 (原始值，W)
        temps: (N, 100, 100) - 原始温度场
    """
    model.eval()
    all_preds = []
    all_targets = []

    n_samples = len(params)

    with torch.no_grad():
        for i in range(n_samples):
            # 获取单个样本
            param = params[i]  # (max_components, 3) - already normalized
            temp = temps[i]    # (100, 100)
            total_power = powers[i]  # scalar

            # 找到有效组件 (power > 0)
            valid_mask = param[:, 2] > 1e-6
            n_valid = np.sum(valid_mask)

            if n_valid == 0:
                continue

            # 提取有效参数 (已经归一化)
            valid_params = param[valid_mask]  # (n_valid, 3)

            # 构建输入张量 (n_valid, 3)
            # params already has [x_norm, y_norm, power_norm]
            node_features = valid_params.copy()

            # 转换为张量
            node_tensor = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_valid, 3)

            # 预测
            pred = model(node_tensor)  # (1, 1, 100, 100)

            # 反归一化预测
            # Step 1: Inverse StandardScaler
            pred_np = pred.cpu().numpy()[0, 0]  # (100, 100) - squeeze batch and channel dims
            pred_flat = pred_np.flatten()
            pred_phys = pred_flat * scaler.scale_ + scaler.mean_
            pred_phys = pred_phys.reshape(100, 100)

            # Step 2: Inverse physics normalization
            # temps_phys = (temps - T_ambient) / power
            # => temps = temps_phys * power + T_ambient
            pred_denorm = pred_phys * total_power + T_ambient

            all_preds.append(pred_denorm)
            all_targets.append(temp)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # 计算每个样本的 R²
    r2_scores = []
    for i in range(len(preds)):
        r2 = r2_score(targets[i].flatten(), preds[i].flatten())
        r2_scores.append(r2)

    r2_scores = np.array(r2_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"\n=== Generalization Results ({component_count} components) ===")
    print(f"Number of samples: {len(r2_scores)}")
    print(f"Per-sample R2: {np.round(r2_scores, 4)}")
    print(f"Mean R2: {mean_r2:.4f}")
    print(f"Std R2: {std_r2:.4f}")
    print(f"Min R2: {r2_scores.min():.4f}")
    print(f"Max R2: {r2_scores.max():.4f}")

    # 保存结果
    results = {
        'component_count': component_count,
        'n_samples': len(r2_scores),
        'per_sample_r2': r2_scores.tolist(),
        'mean_r2': float(mean_r2),
        'std_r2': float(std_r2),
        'min_r2': float(r2_scores.min()),
        'max_r2': float(r2_scores.max()),
    }

    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    plot_r2_bar(r2_scores, output_dir, component_count)
    plot_scatter(preds, targets, output_dir, component_count)
    plot_thermal_comparisons(preds, targets, output_dir, component_count, n_samples=min(6, len(preds)))

    return results


def main():
    parser = argparse.ArgumentParser(description='Test model generalization on 6, 7, 8 components')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--component-count', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default='./generalization_results')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--physics-norm', action='store_true')
    parser.add_argument('--t-ambient', type=float, default=25.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"\nLoading model from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)

    model = SetFNOModel(
        d_in=3,
        d_model=256,
        num_heads=8,
        n_sab=4,
        fno_ch=64,
        fno_modes=24,
        n_fno=6,
        dropout=0.0,
        out_size=100,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])

    # 加载 scaler
    scaler_mean = np.array(ckpt['scaler_mean'])
    scaler_scale = np.array(ckpt['scaler_scale'])

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    print(f"Loaded model and scaler")

    # 加载数据
    print(f"\nLoading data from: {args.data_dir}")
    params, temps, powers, max_power = load_30w_dataset(args.data_dir, max_components=9)

    # 筛选指定组件数的数据 (通过params[:,:,2]功率>0来判断有效组件)
    # 注意：params[:,:,2] 在 load_30w_dataset 中已经被归一化到 0-1 范围
    comp_counts = np.sum(params[:, :, 2] > 1e-6, axis=1)
    mask = comp_counts == args.component_count
    params_subset = params[mask]
    temps_subset = temps[mask]
    powers_subset = powers[mask]

    print(f"Filtered to {len(params_subset)} samples with {args.component_count} components")

    if len(params_subset) == 0:
        print(f"No samples found with {args.component_count} components!")
        return

    # 评估
    results = evaluate_generalization(
        model, params_subset, powers_subset, temps_subset,
        scaler, device, args.t_ambient, args.component_count, args.output_dir
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"generalization_{args.component_count}comp_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
