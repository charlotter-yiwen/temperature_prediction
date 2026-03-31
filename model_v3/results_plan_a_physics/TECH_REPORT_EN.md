# Plan A + Physics Loss - Technical Report
**Date**: 2026-03-30
**Model**: PlanAPlusPhysics (Plan A + Physics Constraint Auxiliary Loss)
**Status**: Completed

---

## 1. Research Objective

Verify whether physics equations as auxiliary loss can improve the generalization performance of the Plan A large model on power extrapolation tasks.

**Core Problem**: Training data is 1-5 components (power 2.5-13.7W), testing is 6-9 components (power 15.7-30W). Pure data-driven models show severe performance degradation at boundaries.

---

## 2. Physics Equation Derivation

### 2.1 Heat Equation (Steady-State)
```
d^2T/dx^2 + d^2T/dy^2 = 0
```
Discretized as 5-point Laplacian:
```
Laplacian = T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j] = 0
```

### 2.2 Boundary Condition (Robin BC)
```
-k * dT/dn = h * (T - T_amb)
```

**Discrete BC equation (edge node i=0)**:
```
(k/dx^2 + h/dx) * T_edge - k/dx^2 * T_adj - h/dx * T_amb = 0
```

**Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| k_fr4 | 0.35 W/(mK) | FR4 substrate thermal conductivity |
| h_conv | 30 W/(m2K) | Convection heat transfer coefficient |
| dx_norm | 1/99 = 0.0101 | Normalized grid spacing |
| k_dx2 | 3430.35 | k/dx^2 |
| h_dx | 2970.0 | h/dx |
| a_edge | 6400.35 | k_dx2 + h_dx |
| c_adj | 0.536 | Normalized BC coefficient = k_dx2/a_edge |
| c_amb | 0.464 | = h_dx/a_edge |

### 2.3 Physics Normalization
```
T_norm = (T - T_amb) / P_total
```
Under physics_norm, BC simplifies to (no ambient term):
```
T_norm_edge - c_adj * T_norm_adj = 0
```

### 2.4 Heat Source Mask
```
is_source[i,j] = 1 if min_dist(component_center, grid[i,j]) < 0.06 else 0
```
PDE loss is only computed on interior non-source points.

---

## 3. Model Architecture

### 3.1 PlanAPlusPhysics Wrapper

```python
class PlanAPlusPhysics(nn.Module):
    def __init__(self, base_model, k_fr4=0.35, h_conv=30.0, dx_norm=1.0/99.0):
        self.base = base_model  # SetFNOModel
        self.k_fr4 = k_fr4
        self.h_conv = h_conv
        k_dx2 = k_fr4 / (dx_norm ** 2)
        h_dx  = h_conv / dx_norm
        self.c_adj = k_dx2 / (k_dx2 + h_dx)  # = 0.536

    def compute_physics_loss(self, xb):
        # BC Loss: T_edge - c_adj*T_adj = 0
        # PDE Loss: Laplacian on interior non-source = 0
```

### 3.2 Base Model SetFNOModel

| Component | Parameter |
|-----------|-----------|
| d_model | 256 |
| num_heads | 8 |
| n_sab (SetAttention blocks) | 4 |
| fno_ch | 64 |
| fno_modes | 24 |
| n_fno | 6 |
| **Total Parameters** | **42,830,209 (~42.8M)** |

### 3.3 Training Configuration

**Phase 1** (data only, no physics loss):
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

**Phase 2** (+ physics constraint):
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

## 4. Training Results

### 4.1 Phase 1 (Data Only)
- Epochs: 500
- Early Stopping: No (data training only)
- Final Train Loss: ~0.04

### 4.2 Phase 2 (+Physics Loss)
- Early Stopping Epoch: 1510 (patience=200 triggered)
- Best Val Loss: 0.0155
- Loss Breakdown:
  - Data Loss: ~0.002
  - PDE Loss: ~0.03
  - BC Loss: ~0.14

### 4.3 Training Script Evaluation Bug
The training script has a bug: when computing test R2, it did not inverse-transform predictions, comparing normalized values directly with raw temperatures. This caused summary.json r2_mean = -168.57, which is NOT the true performance.

True generalization results come from the fixed test scripts predict_plan_a_physics_gen.py and plot_gen_comparison.py (with correct inverse transform applied).

---

## 5. Generalization Test Results

Test dataset: temperature_prediction/data/generation_dataset/

| Components | Power | Samples | R2 (Plan B) | R2 (Plan A no physics) | Improvement |
|------------|-------|---------|-------------|------------------------|-------------|
| 6 components | 15.7W | 10 | 0.9228 | 0.9068 | +0.016 |
| 7 components | 17.5W | 10 | 0.9101 | 0.8797 | +0.030 |
| 8 components | 20.0W | 10 | 0.8981 | 0.8620 | +0.036 |
| 9 components | 30W | 10 | 0.4449 | 0.0760 | +0.369 |

### 5.1 Per-Sample Detailed Results (6 components)

| Sample | R2 | Pred Range | True Range |
|--------|-----|-----------|------------|
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

### 5.2 Per-Sample Detailed Results (9 components)

| Sample | R2 | Pred Range | True Range |
|--------|-----|-----------|------------|
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

## 6. Key Findings

### 6.1 Effect of Physics Constraint
- 6/7/8 components: stable improvement of ~2-4%
- 9 components (30W): huge improvement, R2 from 0.076 to 0.445, ~5x improvement
- Physics loss helps model learn physical laws of boundary conditions instead of memorizing training distribution

### 6.2 Remaining Issues with 9 Components
- Some samples have extremely low R2 (0.0002, 0.0031)
- 30W power (2-3x outside training range 2.5-13.7W) is still extreme extrapolation
- Temperature range 110-135C with steep gradients near heat sources

### 6.3 BC Loss vs Data Loss
- Phase2 training: Data Loss ~0.002, BC Loss ~0.14
- Physics loss magnitude ~70x larger than data loss
- lambda_bc=0.01 makes weighted contributions comparable (BC contribution ~0.0014)

---

## 7. File Structure

```
temperature_prediction/model_v3/
├── train_plan_a_physics_v2.py          # Training script (Plan B main)
├── predict_plan_a_physics_gen.py       # Generalization test script
├── plot_gen_comparison.py              # GT vs Pred comparison generator
├── results_plan_a_physics/
│   ├── plan_a_physics_phase1.pth       # Phase1 model
│   ├── plan_a_physics_phase2.pth       # Phase2 final model - USE THIS
│   ├── gen_comparison/                # Correct GT/Pred comparison images
│   │   ├── 6_Component_15.7W/
│   │   ├── 7_Component_17.5W/
│   │   └── 8_Component_20.0W/
│   ├── loss_curves.png
│   ├── r2_scores.png
│   ├── scatter_pred_vs_true.png
│   └── TECH_REPORT.pdf
└── models/
    └── set_fno_thermal.py             # SetFNOModel definition
```

---

## 8. Visualization

### 8.1 Generated During Training (affected by bug, not true performance)
- loss_curves.png - Training/validation loss curves (log10 scale)
- r2_scores.png - Test R2 score distribution
- scatter_pred_vs_true.png - Predicted vs true temperature scatter
- sample_01~06.png - 6 sample heatmap comparisons

### 8.2 Generalization Test (CORRECT)
- gen_comparison/6_Component_15.7W/sample_01~10.png
- gen_comparison/7_Component_17.5W/sample_01~10.png
- gen_comparison/8_Component_20.0W/sample_01~10.png

---

## 9. Known Bug

### train_plan_a_physics_v2.py Evaluation Bug
Problem: Test R2 computation during training did not apply inverse transform.

Phenomenon: summary.json r2_mean = -168.57.

Impact: Only affects training script evaluation output. Model weights are unaffected. True generalization results are from predict_plan_a_physics_gen.py and gen_comparison/.

Fix: Apply inverse_transform_temps() before R2 computation (already fixed in predict_plan_a_physics_gen.py and plot_gen_comparison.py).

---

## 10. Future Optimization Suggestions

1. Fix training script evaluation: add inverse transform before R2
2. Try smaller lambda_bc: currently 0.01, try 0.005 or 0.001
3. Add in-domain generalization data: include 6-8 component samples in training (not test)
4. Expand power range: add 15-20W data to training
5. Adjust heat source mask threshold: currently 0.06
6. Try Plan A with dropout regularization

---

## 11. Conclusion

Physics constraints as auxiliary loss are an effective method to improve power extrapolation generalization. Plan B maintains Plan A's excellent architecture while significantly improving performance on unseen high-power scenarios by explicitly encoding the physical laws of boundary conditions.

Core Innovation: BC equation precisely derived from SOR solver discrete form, simplified after normalization to T_edge = 0.536 * T_adj, with clear physical meaning.
