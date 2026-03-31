# Plan A + Physics Loss 技术报告
**日期**: 2026-03-30
**模型**: PlanAPlusPhysics (Plan A + 物理约束辅助Loss)
**状态**: ✅ 完成

---

## 1. 研究目标

验证物理方程作为辅助loss是否能够提升Plan A大模型在功率外推任务上的泛化性能。

**核心问题**: 训练数据为1-5组件（功率2.5-13.7W），测试为6-9组件（功率15.7-30W），纯粹的data-driven模型在边界处性能急剧下降。

---

## 2. 物理方程推导

### 2.1 热传导PDE（稳态）
```
∂²T/∂x² + ∂²T/∂y² = 0
```
离散为5点Laplacian：
```
Laplacian = T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4·T[i,j] = 0
```

### 2.2 边界条件（Robin BC）
```
-k · ∂T/∂n = h · (T - T_amb)
```

**离散边界方程（边缘节点 i=0）**:
```
(k/dx² + h/dx) · T_edge - k/dx² · T_adj - h/dx · T_amb = 0
```

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| k_fr4 | 0.35 W/(m·K) | FR4基板热导率 |
| h_conv | 30 W/(m²·K) | 对流换热系数 |
| dx_norm | 1/99 ≈ 0.0101 | 归一化网格间距 |
| k_dx2 | 3430.35 | k/dx² |
| h_dx | 2970.0 | h/dx |
| a_edge | 6400.35 | k_dx2 + h_dx |
| **c_adj** | **0.536** | 归一化BC系数 = k_dx2/a_edge |
| c_amb | 0.464 | = h_dx/a_edge |

### 2.3 物理归一化
```
T_norm = (T - T_amb) / P_total
```
在physics_norm下，BC方程简化为（无ambient项）:
```
T_norm_edge - c_adj · T_norm_adj = 0
```

### 2.4 热源掩码
```
is_source[i,j] = 1 if min_dist(component_center, grid[i,j]) < 0.06 else 0
```
非热源的内网点才计算PDE loss。

---

## 3. 模型架构

### 3.1 PlanAPlusPhysics 包装器

```python
class PlanAPlusPhysics(nn.Module):
    def __init__(self, base_model, k_fr4=0.35, h_conv=30.0, dx_norm=1.0/99.0):
        self.base = base_model  # SetFNOModel
        self.k_fr4 = k_fr4
        self.h_conv = h_conv
        # 计算归一化BC系数
        k_dx2 = k_fr4 / (dx_norm ** 2)
        h_dx  = h_conv / dx_norm
        self.c_adj = k_dx2 / (k_dx2 + h_dx)  # = 0.536
        # 注册网格buffer用于热源掩码计算
        ...

    def compute_physics_loss(self, xb):
        # BC Loss: T_edge - c_adj*T_adj = 0
        # PDE Loss: Laplacian on interior non-source = 0
```

### 3.2 基础模型 SetFNOModel

| 组件 | 参数 |
|------|------|
| d_model | 256 |
| num_heads | 8 |
| n_sab (SetAttention块) | 4 |
| fno_ch | 64 |
| fno_modes | 24 |
| n_fno | 6 |
| **总参数量** | **42,830,209 (~42.8M)** |

### 3.3 训练配置

**Phase 1** (数据only，无物理loss):
```bash
python train_plan_a_physics_v2.py \
    --count-sweep-params ../training_data/params_count_sweep.npy \
    --count-sweep-temps ../training_data/temps_count_sweep.npy \
    --physics-norm --t-ambient 25.0 \
    --d-model 256 --num-heads 8 --n-sab 4 --fno-ch 64 --fno-modes 24 --n-fno 6 \
    --lambda-pde 0.0 --lambda-bc 0.0 \
    --epochs 500 --batch-size 32 --lr 1e-4 \
    --early-stopping --patience 200 \
    --out-dir ./results_plan_a_physics --model-out plan_a_physics_phase1.pth
```

**Phase 2** (+ 物理约束):
```bash
python train_plan_a_physics_v2.py \
    --count-sweep-params ../training_data/params_count_sweep.npy \
    --count-sweep-temps ../training_data/temps_count_sweep.npy \
    --physics-norm --t-ambient 25.0 \
    --d-model 256 --num-heads 8 --n-sab 4 --fno-ch 64 --fno-modes 24 --n-fno 6 \
    --lambda-pde 0.001 --lambda-bc 0.01 \
    --epochs 2000 --batch-size 32 --lr 5e-5 \
    --early-stopping --patience 200 \
    --out-dir ./results_plan_a_physics --model-out plan_a_physics_phase2.pth
```

---

## 4. 训练结果

### 4.1 Phase 1 (数据only)
- Epochs: 500
- 早停: 否（仅数据训练）
- 最终Train Loss: ~0.04

### 4.2 Phase 2 (+物理Loss)
- 早停Epoch: **1510** (patience=200触发)
- Best Val Loss: **0.0155**
- Loss分解:
  - Data Loss: ~0.002
  - PDE Loss: ~0.03
  - BC Loss: ~0.14

### 4.3 ⚠️ 训练脚本评估Bug说明
`train_plan_a_physics_v2.py` 中的评估存在bug：计算测试集R²时未做反归一化，直接拿归一化预测值与真实温度比较，导致 `summary.json` 中 `r2_mean = -168.57`，**这不是真实性能**。

**真实泛化结果**来自修复后的测试脚本 `predict_plan_a_physics_gen.py` 和 `plot_gen_comparison.py`（已正确反归一化）。

训练时生成的可视化文件（`sample_01~06.png`, `r2_scores.png`, `scatter_pred_vs_true.png`）均受此bug影响，不代表真实性能。

---

## 5. 泛化测试结果（✅正确，已修复反归一化）

测试数据集: `temperature_prediction/data/generation_dataset/`

| 组件数 | 功率 | 样本数 | R² (方案B) | R² (方案A无物理) | 提升 |
|--------|------|--------|-----------|----------------|------|
| 6组件 | 15.7W | 10 | **0.9228** | 0.9068 | +0.016 |
| 7组件 | 17.5W | 10 | **0.9101** | 0.8797 | +0.030 |
| 8组件 | 20.0W | 10 | **0.8981** | 0.8620 | +0.036 |
| 9组件 | 30W | 10 | **0.4449** | 0.0760 | **+0.369** |

### 5.1 各样本详细结果

**6组件 (15.7W)**:
| 样本 | R² | Pred范围 | True范围 |
|------|-----|---------|---------|
| 1 | 0.9867 | [68.9, 82.6] | [69.5, 81.0] |
| 2 | 0.9366 | [69.9, 79.0] | [71.0, 79.0] |
| 3 | 0.8290 | [72.7, 77.4] | [73.0, 77.9] |
| 4 | 0.9622 | [68.9, 82.9] | [70.0, 83.4] |
| 5 | 0.9052 | [70.3, 81.8] | [72.0, 81.7] |
| 6 | 0.9786 | [68.7, 80.0] | [70.4, 79.7] |
| 7 | 0.9001 | [72.3, 80.4] | [72.7, 80.2] |
| 8 | 0.9776 | [69.6, 85.4] | [71.0, 83.7] |
| 9 | 0.8654 | [73.1, 78.8] | [73.5, 77.9] |
| 10 | 0.8863 | [72.2, 78.3] | [72.5, 78.1] |

**9组件 (30W)**:
| 样本 | R² | Pred范围 | True范围 |
|------|-----|---------|---------|
| 1 | 0.6177 | [112.1, 130.5] | [112.8, 134.5] |
| 2 | 0.9175 | [114.3, 131.1] | [115.0, 132.2] |
| 3 | 0.0031 | [114.1, 129.4] | [115.8, 129.3] |
| 4 | 0.5966 | [112.3, 134.6] | [113.9, 130.7] |
| 5 | 0.0002 | [115.0, 129.5] | [117.5, 130.7] |
| 6 | 0.2485 | [113.2, 129.2] | [114.0, 132.4] |
| 7 | 0.4060 | [115.7, 126.0] | [114.5, 133.3] |
| 8 | 0.7770 | [109.6, 133.3] | [112.1, 132.2] |
| 9 | 0.1922 | [113.5, 128.7] | [115.7, 131.6] |
| 10 | 0.6897 | [114.7, 129.4] | [114.7, 134.4] |

---

## 6. 关键发现

### 6.1 物理约束的效果
- **6/7/8组件**: 稳定提升 ~2-4%
- **9组件(30W)**: 巨大提升，从R²=0.076升至0.445，提升近5倍
- 物理loss帮助模型学习边界条件的物理规律，而非单纯记忆训练分布

### 6.2 9组件仍存在的问题
- 部分样本R²极低（0.0002, 0.0031）
- 30W功率（超出训练范围2.5-13.7W约2-3倍）仍是极端外推
- 温度范围110-135°C，接近热源附近温度梯度变化剧烈

### 6.3 BC Loss vs Data Loss
- Phase2训练: Data Loss ~0.002, BC Loss ~0.14
- 物理loss量级比数据loss大约70倍
- λ_bc=0.01使两者加权后相当（BC贡献~0.0014）

---

## 7. 文件清单

```
temperature_prediction/model_v3/
├── train_plan_a_physics_v2.py          # 训练脚本（方案B主脚本）
├── predict_plan_a_physics_gen.py       # 泛化测试脚本
├── results_plan_a_physics/
│   ├── plan_a_physics_phase1.pth       # Phase1模型
│   ├── plan_a_physics_phase2.pth       # Phase2最终模型 ← 使用这个
│   ├── loss_curves.png                 # 训练/验证loss曲线
│   ├── r2_scores.png                   # R²分数分布
│   ├── scatter_pred_vs_true.png        # 预测vs真实散点图
│   ├── sample_01~06.png                # 热图对比
│   ├── run_config_latest.json          # 最终配置
│   └── summary.json                    # 训练摘要
└── models/
    └── set_fno_thermal.py              # SetFNOModel定义
```

---

## 8. 可视化图表

### 8.1 训练时生成（❌受bug影响，不代表真实性能）
位于 `results_plan_a_physics/`:
- `loss_curves.png` - 训练/验证loss曲线（log10 scale）
- `r2_scores.png` - 测试集R²分数分布（❌bug导致偏低）
- `scatter_pred_vs_true.png` - 预测vs真实温度散点图（❌bug导致偏低）
- `sample_01~06.png` - 6个样本的热图对比（❌bug导致偏低）

### 8.2 泛化测试生成（✅正确）
位于 `results_plan_a_physics/gen_comparison/`:
- `6_Component_15.7W/sample_01~10.png` - 6组件 GT|Pred|Error
- `7_Component_17.5W/sample_01~10.png` - 7组件 GT|Pred|Error
- `8_Component_20.0W/sample_01~10.png` - 8组件 GT|Pred|Error

生成脚本: `temperature_prediction/model_v3/plot_gen_comparison.py`

---

## 9. 已知Bug

### train_plan_a_physics_v2.py 评估bug
**问题**: 训练结束时计算测试集R²时未做反归一化，直接拿归一化预测值与真实温度比较。

**现象**: `summary.json` 中 `r2_mean = -168.57`，`sample_01~06.png`、`r2_scores.png`、`scatter_pred_vs_true.png` 均显示错误的低性能。

**影响范围**: 仅影响训练脚本的测试集评估输出，**不影响模型权重**。模型本身训练正常，泛化测试结果以 `predict_plan_a_physics_gen.py` 和 `gen_comparison/` 为准。

**修复方案**: 在评估时调用 `inverse_transform_temps()` 将归一化预测转为真实温度后再计算R²（已在 `predict_plan_a_physics_gen.py` 和 `plot_gen_comparison.py` 中修复）。

---

## 10. 后续优化建议

1. **修复训练脚本评估bug**: 在 `train_plan_a_physics_v2.py` 的评估部分加入反归一化
2. **尝试更小的λ_bc**: 当前λ_bc=0.01可能仍太强，可尝试0.005或0.001
3. **加入域内泛化数据**: 将6-8组件样本加入训练（不加到测试集）
4. **功率范围扩展**: 在训练中加入15-20W数据
5. **调整热源掩码阈值**: 当前0.06，可能需要针对不同组件数调整
6. **尝试方案A + dropout**: 在无物理约束的方案A上加dropout正则化

---

## 11. 结论

**物理约束作为辅助loss是提升功率外推泛化的有效方法**。方案B在保持方案A优秀架构的同时，通过显式编码边界条件的物理规律，显著提升了模型在未见高功率场景下的表现。

**核心创新**: BC方程从SOR solver的离散形式精确推导，归一化后简化为`T_edge = 0.536 · T_adj`，物理意义清晰。
