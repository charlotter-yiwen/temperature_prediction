"""
gen_test_plan_a.py
==================
对 PlanA+Physics 模型做 6/7/8/9/10 组件泛化测试。
从 data/generation_dataset/ 读取 JSON 文件, 按组件数分组, 输出各组 R²。

用法:
  python my_scripts/gen_test_plan_a.py
  python my_scripts/gen_test_plan_a.py --model-path model_v3/results_plan_a_physics/plan_a_physics_model.pth
"""
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.metrics import r2_score

TP_DIR  = os.path.dirname(os.path.dirname(__file__))
GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")
sys.path.insert(0, TP_DIR)

from models.set_fno_thermal import SetFNOModel

GRID     = 100
BOARD_MM = 100.0
T_AMB    = 25.0

# (n_comp, powers_list, label)
TEST_CONFIGS = [
    (6,  [2.5, 2.2, 3.0, 2.8, 3.2, 2.0],  "6-Component  (15.7W)"),
    (7,  [2.5]*7,                           "7-Component  (17.5W)"),
    (8,  [2.5]*8,                           "8-Component  (20.0W)"),
    (9,  [2.5]*8 + [10.0],                  "9-Component  (30.0W)"),
    (10, [2.5]*9 + [10.0],                  "10-Component (32.5W)"),
]


def load_gen_samples(n_comp, powers, gen_dir):
    files = sorted([f for f in os.listdir(gen_dir)
                    if f.startswith(f"count{n_comp}") and f.endswith(".json")])
    if not files:
        return None, None

    params_list, temps_list = [], []
    for fname in files:
        path = os.path.join(gen_dir, fname)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        temps   = np.array([d["temperature"] for d in data], dtype=np.float32)
        temps2d = temps.reshape(GRID, GRID)

        name = fname.replace(".json", "")
        nums = [v for v in name.split("_") if v.lstrip("-").isdigit()]
        positions = []
        for i in range(0, 2 * n_comp, 2):
            x, y = int(nums[i]), int(nums[i + 1])
            positions.append([float(x), float(y), powers[len(positions)]])

        # PlanA 接受 (n_comp, 3) 可变长度输入
        p = np.array(positions, dtype=np.float32)   # (n_comp, 3)
        params_list.append(p)
        temps_list.append(temps2d)

    return params_list, np.stack(temps_list, axis=0)  # list of (n_comp,3), (N,H,W)


def planA_predict(model, params_list, temps_raw, norm_info,
                  scaler_y_mean, scaler_y_scale, device, physics_norm):
    """
    params_list : list of np.array (n_comp, 3)  — 每样本可变长度
    temps_raw   : (N, H, W)
    返回 all_pred (N,H,W), all_true (N,H,W)
    """
    max_power   = norm_info["max_power"]
    board_size  = norm_info["board_size"]

    all_pred, all_true = [], []
    model.eval()
    with torch.no_grad():
        for i, (p_raw, T_true) in enumerate(zip(params_list, temps_raw)):
            n = p_raw.shape[0]
            # 归一化坐标和功率
            p_norm = p_raw.copy()
            p_norm[:, 0] /= board_size          # x → [0,1]
            p_norm[:, 1] /= board_size          # y → [0,1]
            p_norm[:, 2] /= max_power           # power → [0,1]

            if physics_norm:
                total_p = float(p_raw[:, 2].sum())
                # 输入归一化为 θ=(T-T_amb)/P_total 对应的空间
                # 模型期望 (1, n_comp, 3), 输出 (1,1,H,W) in scaled θ-space
                pass  # 归一化仍为 (x/board, y/board, p/max_p)

            inp = torch.tensor(p_norm, dtype=torch.float32,
                               device=device).unsqueeze(0)  # (1, n, 3)
            out_sc = model(inp)                              # (1, 1, H, W)

            # 反归一化: scaled_theta → T
            sc_mean  = torch.tensor(np.array(scaler_y_mean).ravel(),  dtype=torch.float32, device=device).view(GRID, GRID)
            sc_scale = torch.tensor(np.array(scaler_y_scale).ravel(), dtype=torch.float32, device=device).view(GRID, GRID)
            theta = out_sc.squeeze() * sc_scale + sc_mean    # (H, W)

            if physics_norm:
                total_p = float(p_raw[:, 2].sum())
                T_pred = theta.cpu().numpy() * total_p + T_AMB
            else:
                T_pred = theta.cpu().numpy()

            all_pred.append(T_pred)
            all_true.append(T_true)

    return np.stack(all_pred, 0), np.stack(all_true, 0)


def evaluate_group(model, params_list, temps_raw, norm_info,
                   scaler_y_mean, scaler_y_scale, device, physics_norm, label):
    preds, trues = planA_predict(model, params_list, temps_raw,
                                  norm_info, scaler_y_mean, scaler_y_scale,
                                  device, physics_norm)

    r2_all = r2_score(trues.ravel(), preds.ravel())
    r2_per = [r2_score(trues[i].ravel(), preds[i].ravel()) for i in range(len(trues))]
    r2_mean = float(np.mean(r2_per))
    r2_std  = float(np.std(r2_per))
    r2_min  = float(np.min(r2_per))

    print(f"\n{label}")
    print(f"  R²(all pixels) : {r2_all:.4f}")
    print(f"  R²(per-sample) : mean={r2_mean:.4f}  std={r2_std:.4f}  min={r2_min:.4f}")
    for i, r2 in enumerate(r2_per):
        print(f"    sample {i+1:2d}: R²={r2:.4f}  "
              f"pred=[{preds[i].min():.1f},{preds[i].max():.1f}]  "
              f"true=[{trues[i].min():.1f},{trues[i].max():.1f}]")
    return r2_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",
        default=os.path.join(TP_DIR, "model_v3", "results_plan_a_physics", "plan_a_physics_model.pth"))
    parser.add_argument("--params",
        default=os.path.join(TP_DIR, "training_data", "params_count_sweep.npy"))
    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading PlanA model: {cli.model_path}")

    ckpt = torch.load(cli.model_path, map_location=device, weights_only=False)
    args_ckpt    = ckpt.get("args", {})
    physics_norm = bool(args_ckpt.get("physics_norm", True))
    grid_size    = int(ckpt.get("grid_size", GRID))

    # 构建 norm_info（max_power 来自训练集）
    params_tr = np.load(cli.params).astype(np.float32)
    # params_tr shape: (N, MAX_COMP*3) 或 (N, MAX_COMP, 3)
    if params_tr.ndim == 2:
        params_tr = params_tr.reshape(params_tr.shape[0], -1, 3)
    valid_powers = params_tr[:, :, 2][~np.isnan(params_tr[:, :, 2])]
    max_power    = float(valid_powers.max())
    norm_info = {
        "board_size": BOARD_MM,
        "max_power":  max_power,
        "physics_norm": physics_norm,
    }
    print(f"norm_info: max_power={max_power:.4f}, physics_norm={physics_norm}")

    scaler_y_mean  = ckpt["scaler_y_mean"]   # (H, W) or scalar numpy
    scaler_y_scale = ckpt["scaler_y_scale"]

    # 处理 base. 前缀 和 physics-wrapper 专用 buffers
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("base.") for k in sd.keys()):
        sd = {k[5:]: v for k, v in sd.items() if k.startswith("base.")}
    extra = {"lap_kernel", "interior_mask"}
    sd = {k: v for k, v in sd.items() if k not in extra}

    model = SetFNOModel(
        d_in      = args_ckpt.get("d_per_comp", 3),
        d_model   = args_ckpt.get("d_model",    256),
        num_heads = args_ckpt.get("num_heads",   8),
        n_sab     = args_ckpt.get("n_sab",       4),
        fno_ch    = args_ckpt.get("fno_ch",      64),
        fno_modes = args_ckpt.get("fno_modes",   24),
        n_fno     = args_ckpt.get("n_fno",       6),
        dropout   = 0.0,
        out_size  = grid_size,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n{'='*60}")
    print(f"  PlanA+Physics — Generalization Test (6~10 comp)")
    print(f"{'='*60}")

    summary = {}
    for n_comp, powers, label in TEST_CONFIGS:
        params_list, temps_raw = load_gen_samples(n_comp, powers, GEN_DIR)
        if params_list is None:
            print(f"\n{label}: 无测试数据，跳过")
            continue
        r2 = evaluate_group(model, params_list, temps_raw, norm_info,
                            scaler_y_mean, scaler_y_scale, device, physics_norm, label)
        summary[label] = r2

    print(f"\n{'='*60}")
    print("Summary:")
    for label, r2 in summary.items():
        print(f"  {label}: R²={r2:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
