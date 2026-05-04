"""
gen_test_v3.py
==============
对 SetFNO V3 模型做 6/7/8/9 组件泛化测试。
从 data/generation_dataset/ 读取 JSON 文件，按组件数分组，
使用训练 checkpoint 中的 norm_info 做一致的归一化，输出各组 R²。
"""
import os, sys, json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

TP_DIR   = os.path.dirname(os.path.dirname(__file__))
GEN_DIR  = os.path.join(TP_DIR, "data", "generation_dataset")
sys.path.insert(0, TP_DIR)

# ── 导入 V3 模型和帮助函数（2ch 和 3ch 均支持，自动检测）─────────────────────
from my_scripts.train_setfno_v3 import (
    SetFNOv3 as SetFNOv3_2ch, ThermalDatasetV3, denorm_and_restore_T,
    GRID, BOARD_MM, T_AMB, MAX_COMP_REF
)
from my_scripts.train_setfno_v3_3ch import (
    SetFNOv3 as SetFNOv3_3ch, ThermalDatasetV3 as ThermalDatasetV3_3ch,
    make_heatmap as make_heatmap_3ch
)

# ── 功率配置（与生成数据时一致）──────────────────────────────────────────────
TEST_CONFIGS = [
    (6,  [2.5, 2.2, 3.0, 2.8, 3.2, 2.0],  "6-Component (15.7W)"),
    (7,  [2.5]*7,                           "7-Component (17.5W)"),
    (8,  [2.5]*8,                           "8-Component (20.0W)"),
    (9,  [2.5]*8 + [10.0],                  "9-Component (30.0W)"),
]

MODEL_PATH = os.path.join(TP_DIR, "my_scripts", "results_v3_poweraug_fixed", "setfno_v3_best.pth")


def load_gen_samples(n_comp, powers, gen_dir):
    """加载某组件数的所有 JSON 文件，返回 (params_raw, temps_raw)。"""
    files = sorted([f for f in os.listdir(gen_dir)
                    if f.startswith(f"count{n_comp}") and f.endswith(".json")])
    if not files:
        return None, None

    params_list, temps_list = [], []
    for fname in files:
        path = os.path.join(gen_dir, fname)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        temps = np.array([d["temperature"] for d in data], dtype=np.float32)
        temps_2d = temps.reshape(100, 100)

        # 从文件名解析坐标
        name  = fname.replace(".json", "")
        nums  = [v for v in name.split("_") if v.lstrip("-").isdigit()]
        # 坐标对数 = n_comp
        positions = []
        for i in range(0, 2 * n_comp, 2):
            x, y = int(nums[i]), int(nums[i + 1])
            positions.append([x, y, powers[len(positions)]])

        # 填入 MAX_COMP_REF 大小的 params 矩阵（其余 NaN）
        p = np.full((MAX_COMP_REF, 3), np.nan, dtype=np.float32)
        for j, pos in enumerate(positions):
            p[j] = pos

        params_list.append(p)
        temps_list.append(temps_2d)

    params_raw = np.stack(params_list, axis=0)   # (N, MAX_COMP_REF, 3)
    temps_raw  = np.stack(temps_list,  axis=0)   # (N, 100, 100)
    return params_raw, temps_raw


def evaluate_group(model, device, norm_info, params_raw, temps_raw, label,
                   dataset_cls=None):
    # 用检查点的 norm_info 构建 dataset（保证归一化一致）
    if dataset_cls is None:
        dataset_cls = ThermalDatasetV3
    ds = dataset_cls(
        params_raw, temps_raw,
        max_power_ref=norm_info["max_power_ref"],
        p_total_ref=norm_info["p_total_ref"],
        theta_mean=norm_info["theta_mean"],
        theta_std=norm_info["theta_std"],
    )
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for params_7d, hmaps, _, total_p, temps_r in dl:
            params_7d = params_7d.to(device)
            hmaps     = hmaps.to(device)
            theta_sc  = model(params_7d, hmaps)          # (B,1,H,W)
            T_pred    = denorm_and_restore_T(
                theta_sc, total_p.to(device),
                norm_info["theta_mean"], norm_info["theta_std"],
                t_amb=norm_info.get("t_amb", T_AMB)
            ).cpu().numpy()                               # (B,H,W)
            T_true    = temps_r.squeeze(1).numpy()        # (B,H,W)
            all_true.append(T_true.reshape(T_true.shape[0], -1))
            all_pred.append(T_pred.reshape(T_pred.shape[0], -1))

    all_true = np.concatenate(all_true, axis=0)   # (N, 10000)
    all_pred = np.concatenate(all_pred, axis=0)

    r2_all = r2_score(all_true.ravel(), all_pred.ravel())
    r2_per = [r2_score(all_true[i], all_pred[i]) for i in range(len(all_true))]
    r2_mean = float(np.mean(r2_per))
    r2_std  = float(np.std(r2_per))
    r2_min  = float(np.min(r2_per))

    print(f"\n{label}")
    print(f"  R²(all pixels) : {r2_all:.4f}")
    print(f"  R²(per-sample) : mean={r2_mean:.4f}  std={r2_std:.4f}  min={r2_min:.4f}")
    for i, r2 in enumerate(r2_per):
        print(f"    sample {i+1:2d}: R²={r2:.4f}  "
              f"pred=[{all_pred[i].min():.1f},{all_pred[i].max():.1f}]  "
              f"true=[{all_true[i].min():.1f},{all_true[i].max():.1f}]")
    return r2_mean


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    cli_args = parser.parse_args()
    model_path = cli_args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    norm_info = ckpt["norm_info"]
    args_ckpt = ckpt.get("args", {})

    print(f"norm_info: {norm_info}")

    # 自动检测是 2ch 还是 3ch 模型
    spatial_in_ch = ckpt["model"]["spatial_embed.0.weight"].shape[1]
    is_3ch = (spatial_in_ch == 5)
    print(f"Model type: {'3-channel' if is_3ch else '2-channel'} (spatial_embed input={spatial_in_ch}ch)")
    ModelCls   = SetFNOv3_3ch   if is_3ch else SetFNOv3_2ch
    DatasetCls = ThermalDatasetV3_3ch if is_3ch else ThermalDatasetV3

    model = ModelCls(
        d_model   = args_ckpt.get("d_model",    256),
        num_heads = args_ckpt.get("num_heads",   8),
        n_sab     = args_ckpt.get("n_sab",       4),
        fno_ch    = args_ckpt.get("fno_ch",      64),
        fno_modes = args_ckpt.get("fno_modes",   24),
        n_fno     = args_ckpt.get("n_fno",       6),
        dropout   = 0.0,
        grid      = GRID,
        use_corrector = not args_ckpt.get("no_corrector", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"  SetFNO V3 — Generalization Test")
    print(f"  lambda_bc={args_ckpt.get('lambda_bc','?')}, lambda_pde={args_ckpt.get('lambda_pde','?')}")
    print(f"{'='*60}")

    summary = {}
    for n_comp, powers, label in TEST_CONFIGS:
        params_raw, temps_raw = load_gen_samples(n_comp, powers, GEN_DIR)
        if params_raw is None:
            print(f"\n{label}: no data found in {GEN_DIR}")
            continue
        r2 = evaluate_group(model, device, norm_info, params_raw, temps_raw, label,
                           dataset_cls=DatasetCls)
        summary[label] = r2

    print(f"\n{'='*60}")
    print("Summary:")
    for label, r2 in summary.items():
        print(f"  {label}: R²={r2:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
