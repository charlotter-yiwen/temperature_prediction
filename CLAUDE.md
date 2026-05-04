# CLAUDE.md - 项目规则

## 环境配置
- Python 环境：`C:\anaconda3\envs\magnet2`
- 激活方式：`conda activate magnet2` 或直接用 `C:/anaconda3/envs/magnet2/python.exe`

## 版本控制
- 每次代码修改后（特别是 thermal 模型或 heatsink 算法），立即用 `git add -A && git commit -m "..."` 提交

## 模型训练
- 训练前必须提醒用户以下重要参数：
  - `--physics-norm` - 物理归一化
  - `--epochs` - 训练轮次
  - `--batch-size` - 批大小
  - `--lr` - 学习率
  - `--early-stopping` - 是否启用早停
  - `--n-runs` - 训练次数
- 确认数据集是否符合用户要求的组件数量

## 调试规则
- 当用户要求"调试"或"修复"代码时，**必须先问**：具体症状是什么？期望行为是什么？
- 对于 heatsink/tree 算法，修改后验证：trunk 宽度/高度扩展、中心位置处理、连接检查逻辑

## 训练数据生成
- 检查 JSON 文件数量和格式是否符合要求
- 确认 grid size (100x100 vs 200x200)
- 确认参数格式（绝对坐标 vs 相对位移）
- **必须使用 thermal_prediction.py 生成数据，不得修改该文件**

## 代码修改规则
- 当用户提出**较大的要求**时，先讨论方案并确认理解一致后再动手，不要直接改代码
- 只有等用户说"可以"或"开始吧"后才开始修改
- AI 写的所有代码脚本放到：`temperature_prediction/my_scripts/` 文件夹

## 可视化规则
- Loss 曲线纵坐标统一使用**log10 scale**：`plt.yscale('log')`

## 定期总结与更新
- 每8小时对当前对话做一次总结
- 将关键进展、待办事项、当前最佳超参数配置更新到 CLAUDE.md

## 当前进展

### 最佳模型：方案B Plan A + 物理Loss（推荐）
- **模型参数**: 42,830,209（约 42.8M）
- **架构**: PlanAPlusPhysics 包装器（SetFNOModel + 物理约束loss）
- **泛化结果**:
  - 6组件 (15.7W): R²=**0.9228**
  - 7组件 (17.5W): R²=**0.9101**
  - 8组件 (20.0W): R²=**0.8981**
  - 9组件 (30W): R²=**0.4449**（从0.076大幅提升）
- **物理约束设计**:
  - BC方程: `T_edge - c_adj * T_adj = 0`，其中 `c_adj = k_dx2/(k_dx2+h_dx) = 0.536`
  - PDE方程: interior non-source 点上 Laplacian = 0
  - 热源掩码: distance < 0.06 归一化单位
  - λ_pde=0.001, λ_bc=0.01
- **训练**: Phase1(500epoch数据only) → Phase2(1510epoch, early stop, best_val=0.0155)
- **文件位置**: `temperature_prediction/model_v3/results_plan_a_physics/`

### 方案A 均衡放大（hp_search/plan_a_balanced，无物理约束）
- **模型参数**: 42,830,209（约 42.8M）
- **训练R²**: 0.9775
- **泛化结果**:
  - 6组件 (15.7W): R²=0.9068
  - 7组件 (17.5W): R²=0.8797
  - 8组件 (20.0W): R²=0.8620
  - 9组件 (30W): R²=0.0760

### 方案B 超参数
| 类别 | 参数 | 值 |
|------|------|-----|
| 架构 | d_model | 256 |
| | num_heads | 8 |
| | n_sab | 4 |
| | fno_ch | 64 |
| | fno_modes | 24 |
| | n_fno | 6 |
| | dropout | 0.0 |
| 物理 | lambda_pde | 0.001 |
| | lambda_bc | 0.01 |
| | k_fr4 | 0.35 |
| | h_conv | 30.0 |
| | c_adj | 0.536 |
| 训练 | epochs | 2000 |
| | batch_size | 32 |
| | lr | 5e-5 (Phase2) |
| | weight_decay | 1e-5 |
| | val_ratio | 0.1 |
| | early_stopping | True |
| | patience | 200 |
| 数据 | n_components | 5 |
| | physics_norm | True |
| | t_ambient | 25.0 |
| | grid_size | 100 |

### 历史模型对比
| 方案 | 参数 | 6组件 | 7组件 | 8组件 | 9组件 |
|------|------|-------|-------|-------|-------|
| **B Plan A+物理** | 42.8M | **0.9228** | **0.9101** | **0.8981** | 0.4449 |
| A 均衡 | 42.8M | 0.9068 | 0.8797 | 0.8620 | 0.0760 |
| V3 two_phase | 32.6M | 0.8425 | 0.7686 | 0.5824 | 0.4843 |
| **V3 poweraug_fixed** | 32.6M | 0.7914 | 0.7093 | 0.6326 | **0.8542** |
| V3 poweraug (buggy) | 32.6M | 0.6500 | 0.6060 | 0.5651 | 0.8290 |
| B Transformer | 12.0M | 0.8743 | 0.7646 | 0.7662 | - |
| C FNO | 70.0M | 0.8209 | 0.8074 | 0.7799 | - |
| V3 A+D | 5.7M | 0.7844 | 0.7750 | 0.6468 | 0.0516 |
| 原始 | 4.4M | 0.8898 | 0.8744 | 0.8649 | - |

### V3 (Plan A + Plan D) 模型
- **模型参数**: 5,742,082（约 5.7M）
- **训练R²**: 0.9860 (1-5组件)
- **泛化结果**:
  - 6组件 (15.7W): R²=0.7844
  - 7组件 (17.5W): R²=0.7750
  - 8组件 (20.0W): R²=0.6468
  - 9组件 (30W): R²=0.0516
- **架构**: TemperatureUpsampler (Plan A) + U-Net RefineNet (Plan D)
- **注意**: V3只有Plan A的1/7大小，泛化性能更差

### 方案A 超参数（无物理约束）
| 类别 | 参数 | 值 |
|------|------|-----|
| 架构 | d_model | 256 |
| | num_heads | 8 |
| | n_sab | 4 |
| | fno_ch | 64 |
| | fno_modes | 24 |
| | n_fno | 6 |
| | dropout | 0.0 |
| 训练 | epochs | 2000 |
| | batch_size | 32 |
| | lr | 0.0001 |
| | weight_decay | 1e-5 |
| | val_ratio | 0.1 |
| 数据 | n_components | 5 |
| | physics_norm | True |
| | t_ambient | 25.0 |
| | grid_size | 100 |

### 问题分析
- 训练数据功率范围：2.5-13.7W
- 测试功率范围：6组件15.7W，7组件17.5W，8组件20W，9组件30W
- physics_norm 在功率外推时有限制
- **物理约束作为辅助loss显著提升泛化**：9组件R²从0.076→0.445
- 功率外推仍是核心问题，但物理loss帮助显著改善

## 待办事项
- [ ] 尝试更弱的物理loss系数进一步提升9组件性能
- [ ] 在训练数据中加入6-8组件样本（域内泛化）
- [ ] 尝试更高的λ_bc值观察对9组件的影响
