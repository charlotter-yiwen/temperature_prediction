# Training Log

---
## POD-RBF Run — 2026-03-05 18:22:28

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-05 18:48:16

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 8 |
| Learning rate | 0.0001 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.920922 |
| Max R²              | 0.988034 |
| Min R²              | 0.754172 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.909879 |
| 02 | 0.943336 |
| 03 | 0.974522 |
| 04 | 0.946146 |
| 05 | 0.776380 |
| 06 | 0.942050 |
| 07 | 0.944139 |
| 08 | 0.895000 |
| 09 | 0.988034 |
| 10 | 0.754172 |
| 11 | 0.911586 |
| 12 | 0.849632 |
| 13 | 0.951850 |
| 14 | 0.934957 |
| 15 | 0.942628 |
| 16 | 0.973933 |
| 17 | 0.859027 |
| 18 | 0.985813 |
| 19 | 0.949967 |
| 20 | 0.985388 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:31:53

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | multiquadric |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882342 |
| Max R²              | 0.979605 |
| Min R²              | 0.676289 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922506 |
| 02 | 0.876766 |
| 03 | 0.952608 |
| 04 | 0.900092 |
| 05 | 0.676289 |
| 06 | 0.876414 |
| 07 | 0.859170 |
| 08 | 0.831309 |
| 09 | 0.979605 |
| 10 | 0.817379 |
| 11 | 0.846535 |
| 12 | 0.751178 |
| 13 | 0.953843 |
| 14 | 0.900018 |
| 15 | 0.840629 |
| 16 | 0.956774 |
| 17 | 0.860940 |
| 18 | 0.955790 |
| 19 | 0.917406 |
| 20 | 0.971583 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:32:03

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | gaussian |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | -0.006624 |
| Max R²              | 0.446634 |
| Min R²              | -0.388129 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.169032 |
| 02 | -0.082441 |
| 03 | 0.133627 |
| 04 | 0.049226 |
| 05 | -0.173303 |
| 06 | 0.056204 |
| 07 | 0.059592 |
| 08 | -0.273613 |
| 09 | 0.031742 |
| 10 | 0.446634 |
| 11 | -0.388129 |
| 12 | -0.368372 |
| 13 | 0.100751 |
| 14 | 0.140502 |
| 15 | 0.063156 |
| 16 | 0.182095 |
| 17 | -0.277575 |
| 18 | -0.109133 |
| 19 | 0.071037 |
| 20 | 0.036494 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:32:13

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | multiquadric |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882342 |
| Max R²              | 0.979605 |
| Min R²              | 0.676289 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922506 |
| 02 | 0.876766 |
| 03 | 0.952608 |
| 04 | 0.900092 |
| 05 | 0.676289 |
| 06 | 0.876414 |
| 07 | 0.859170 |
| 08 | 0.831309 |
| 09 | 0.979605 |
| 10 | 0.817379 |
| 11 | 0.846535 |
| 12 | 0.751178 |
| 13 | 0.953843 |
| 14 | 0.900018 |
| 15 | 0.840629 |
| 16 | 0.956774 |
| 17 | 0.860940 |
| 18 | 0.955790 |
| 19 | 0.917406 |
| 20 | 0.971583 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:32:15

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | linear |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882285 |
| Max R²              | 0.979585 |
| Min R²              | 0.676153 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922480 |
| 02 | 0.876691 |
| 03 | 0.952569 |
| 04 | 0.900056 |
| 05 | 0.676153 |
| 06 | 0.876370 |
| 07 | 0.859120 |
| 08 | 0.831210 |
| 09 | 0.979585 |
| 10 | 0.817326 |
| 11 | 0.846409 |
| 12 | 0.751083 |
| 13 | 0.953818 |
| 14 | 0.899969 |
| 15 | 0.840574 |
| 16 | 0.956753 |
| 17 | 0.860869 |
| 18 | 0.955750 |
| 19 | 0.917372 |
| 20 | 0.971547 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:32:17

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:32:36

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | linear |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882285 |
| Max R²              | 0.979585 |
| Min R²              | 0.676153 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922480 |
| 02 | 0.876691 |
| 03 | 0.952569 |
| 04 | 0.900056 |
| 05 | 0.676153 |
| 06 | 0.876370 |
| 07 | 0.859120 |
| 08 | 0.831210 |
| 09 | 0.979585 |
| 10 | 0.817326 |
| 11 | 0.846409 |
| 12 | 0.751083 |
| 13 | 0.953818 |
| 14 | 0.899969 |
| 15 | 0.840574 |
| 16 | 0.956753 |
| 17 | 0.860869 |
| 18 | 0.955750 |
| 19 | 0.917372 |
| 20 | 0.971547 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:33:05

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:00

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:03

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:05

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 10 |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.915218 |
| Max R²              | 0.974742 |
| Min R²              | 0.796685 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.937554 |
| 02 | 0.872119 |
| 03 | 0.950358 |
| 04 | 0.947739 |
| 05 | 0.796685 |
| 06 | 0.943465 |
| 07 | 0.945080 |
| 08 | 0.852269 |
| 09 | 0.974742 |
| 10 | 0.874319 |
| 11 | 0.886921 |
| 12 | 0.872182 |
| 13 | 0.959260 |
| 14 | 0.917818 |
| 15 | 0.932249 |
| 16 | 0.928471 |
| 17 | 0.879487 |
| 18 | 0.964176 |
| 19 | 0.927808 |
| 20 | 0.941661 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:08

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 15 |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.925696 |
| Max R²              | 0.984332 |
| Min R²              | 0.822222 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941678 |
| 02 | 0.915655 |
| 03 | 0.963153 |
| 04 | 0.953994 |
| 05 | 0.822222 |
| 06 | 0.948365 |
| 07 | 0.945517 |
| 08 | 0.868663 |
| 09 | 0.984332 |
| 10 | 0.850053 |
| 11 | 0.905421 |
| 12 | 0.864106 |
| 13 | 0.963841 |
| 14 | 0.937061 |
| 15 | 0.930937 |
| 16 | 0.959997 |
| 17 | 0.882763 |
| 18 | 0.977770 |
| 19 | 0.932918 |
| 20 | 0.965478 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:10

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.926490 |
| Max R²              | 0.985565 |
| Min R²              | 0.804299 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.945778 |
| 02 | 0.919941 |
| 03 | 0.962438 |
| 04 | 0.958992 |
| 05 | 0.804299 |
| 06 | 0.948220 |
| 07 | 0.944253 |
| 08 | 0.881373 |
| 09 | 0.985565 |
| 10 | 0.850138 |
| 11 | 0.905750 |
| 12 | 0.842980 |
| 13 | 0.961163 |
| 14 | 0.946607 |
| 15 | 0.931718 |
| 16 | 0.962868 |
| 17 | 0.885084 |
| 18 | 0.979488 |
| 19 | 0.934933 |
| 20 | 0.978202 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:13

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 30 |
| RBF kernel  | auto |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.924736 |
| Max R²              | 0.988072 |
| Min R²              | 0.791840 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.942915 |
| 02 | 0.919609 |
| 03 | 0.963306 |
| 04 | 0.955810 |
| 05 | 0.791840 |
| 06 | 0.944406 |
| 07 | 0.943088 |
| 08 | 0.886685 |
| 09 | 0.988072 |
| 10 | 0.845837 |
| 11 | 0.913405 |
| 12 | 0.826718 |
| 13 | 0.962485 |
| 14 | 0.942895 |
| 15 | 0.931041 |
| 16 | 0.961787 |
| 17 | 0.883352 |
| 18 | 0.977353 |
| 19 | 0.933007 |
| 20 | 0.981099 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:15

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:17

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.926490 |
| Max R²              | 0.985565 |
| Min R²              | 0.804299 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.945778 |
| 02 | 0.919941 |
| 03 | 0.962438 |
| 04 | 0.958992 |
| 05 | 0.804299 |
| 06 | 0.948220 |
| 07 | 0.944253 |
| 08 | 0.881373 |
| 09 | 0.985565 |
| 10 | 0.850138 |
| 11 | 0.905750 |
| 12 | 0.842980 |
| 13 | 0.961163 |
| 14 | 0.946607 |
| 15 | 0.931718 |
| 16 | 0.962868 |
| 17 | 0.885084 |
| 18 | 0.979488 |
| 19 | 0.934933 |
| 20 | 0.978202 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:20

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | multiquadric |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882342 |
| Max R²              | 0.979605 |
| Min R²              | 0.676289 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922506 |
| 02 | 0.876766 |
| 03 | 0.952608 |
| 04 | 0.900092 |
| 05 | 0.676289 |
| 06 | 0.876414 |
| 07 | 0.859170 |
| 08 | 0.831309 |
| 09 | 0.979605 |
| 10 | 0.817379 |
| 11 | 0.846535 |
| 12 | 0.751178 |
| 13 | 0.953843 |
| 14 | 0.900018 |
| 15 | 0.840629 |
| 16 | 0.956774 |
| 17 | 0.860940 |
| 18 | 0.955790 |
| 19 | 0.917406 |
| 20 | 0.971583 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:22

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | multiquadric |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882255 |
| Max R²              | 0.977712 |
| Min R²              | 0.683862 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922396 |
| 02 | 0.871723 |
| 03 | 0.951498 |
| 04 | 0.901412 |
| 05 | 0.683862 |
| 06 | 0.877275 |
| 07 | 0.859000 |
| 08 | 0.825315 |
| 09 | 0.977712 |
| 10 | 0.818712 |
| 11 | 0.840001 |
| 12 | 0.755987 |
| 13 | 0.954070 |
| 14 | 0.901055 |
| 15 | 0.840599 |
| 16 | 0.958012 |
| 17 | 0.862967 |
| 18 | 0.957932 |
| 19 | 0.917293 |
| 20 | 0.968277 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:25

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | auto |
| RBF kernel  | cubic |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.921167 |
| Max R²              | 0.988303 |
| Min R²              | 0.815334 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.936634 |
| 02 | 0.929083 |
| 03 | 0.959591 |
| 04 | 0.935344 |
| 05 | 0.815334 |
| 06 | 0.946508 |
| 07 | 0.954405 |
| 08 | 0.873803 |
| 09 | 0.988303 |
| 10 | 0.826987 |
| 11 | 0.906160 |
| 12 | 0.824158 |
| 13 | 0.948536 |
| 14 | 0.932547 |
| 15 | 0.940282 |
| 16 | 0.959874 |
| 17 | 0.841503 |
| 18 | 0.979819 |
| 19 | 0.940271 |
| 20 | 0.984208 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:27

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.922882 |
| Max R²              | 0.987959 |
| Min R²              | 0.778077 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.941205 |
| 02 | 0.926770 |
| 03 | 0.962758 |
| 04 | 0.951805 |
| 05 | 0.778077 |
| 06 | 0.944464 |
| 07 | 0.942642 |
| 08 | 0.885512 |
| 09 | 0.987959 |
| 10 | 0.843845 |
| 11 | 0.910869 |
| 12 | 0.821061 |
| 13 | 0.958368 |
| 14 | 0.943862 |
| 15 | 0.929733 |
| 16 | 0.961063 |
| 17 | 0.875951 |
| 18 | 0.975812 |
| 19 | 0.933993 |
| 20 | 0.981895 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-07 09:56:30

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | randomized_svd |
| POD rank    | auto |
| RBF kernel  | multiquadric |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.882342 |
| Max R²              | 0.979605 |
| Min R²              | 0.676289 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.922506 |
| 02 | 0.876766 |
| 03 | 0.952608 |
| 04 | 0.900092 |
| 05 | 0.676289 |
| 06 | 0.876414 |
| 07 | 0.859170 |
| 08 | 0.831309 |
| 09 | 0.979605 |
| 10 | 0.817379 |
| 11 | 0.846535 |
| 12 | 0.751178 |
| 13 | 0.953843 |
| 14 | 0.900018 |
| 15 | 0.840629 |
| 16 | 0.956774 |
| 17 | 0.860940 |
| 18 | 0.955790 |
| 19 | 0.917406 |
| 20 | 0.971583 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---

## POD Parameter Sweep -- 2026-03-07 09:56:30

| No | Avg R2 | Max R2 | Min R2 | Configuration |
|---|---|---|---|---|
| 8 | 0.92649 | 0.985565 | 0.804299 | svd, thin_plate_spline, rank=20 |
| 5 | 0.92649 | 0.985565 | 0.804299 | svd, auto kernel, rank=20 |
| 4 | 0.925696 | 0.984332 | 0.822222 | svd, auto kernel, rank=15 |
| 6 | 0.924736 | 0.988072 | 0.79184 | svd, auto kernel, rank=30 |
| 12 | 0.922882 | 0.987959 | 0.778077 | randomized_svd, thin_plate_spline, rank=auto |
| 7 | 0.922882 | 0.987959 | 0.778077 | svd, thin_plate_spline, rank=auto |
| 1 | 0.922882 | 0.987959 | 0.778077 | baseline (randomized_svd, auto, rank=auto) |
| 2 | 0.922882 | 0.987959 | 0.778077 | svd, auto kernel, rank=auto |
| 11 | 0.921167 | 0.988303 | 0.815334 | svd, cubic, rank=auto |
| 3 | 0.915218 | 0.974742 | 0.796685 | svd, auto kernel, rank=10 |
| 9 | 0.882342 | 0.979605 | 0.676289 | svd, multiquadric, rank=auto |
| 13 | 0.882342 | 0.979605 | 0.676289 | randomized_svd, multiquadric, rank=auto |
| 10 | 0.882255 | 0.977712 | 0.683862 | svd, multiquadric, rank=20 |

**Best:** `python pod.py --svd-method svd --rank 20 --rbf-kernel thin_plate_spline` -- Avg R2 = 0.92649


---
## cGAN-CNN Run — 2026-03-08 08:58:43

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 8 |
| Learning rate | 0.0001 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.916529 |
| Max R²              | 0.984703 |
| Min R²              | 0.748423 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.930210 |
| 02 | 0.927555 |
| 03 | 0.963908 |
| 04 | 0.945221 |
| 05 | 0.748423 |
| 06 | 0.941167 |
| 07 | 0.957796 |
| 08 | 0.880040 |
| 09 | 0.984703 |
| 10 | 0.789468 |
| 11 | 0.897895 |
| 12 | 0.825625 |
| 13 | 0.939801 |
| 14 | 0.938653 |
| 15 | 0.955938 |
| 16 | 0.972131 |
| 17 | 0.840405 |
| 18 | 0.980193 |
| 19 | 0.934742 |
| 20 | 0.976698 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:00:13

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 8 |
| Learning rate | 0.0005 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.914433 |
| Max R²              | 0.988522 |
| Min R²              | 0.725262 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.930297 |
| 02 | 0.931085 |
| 03 | 0.969572 |
| 04 | 0.934255 |
| 05 | 0.725262 |
| 06 | 0.949844 |
| 07 | 0.970784 |
| 08 | 0.866265 |
| 09 | 0.988522 |
| 10 | 0.762047 |
| 11 | 0.893810 |
| 12 | 0.796687 |
| 13 | 0.948257 |
| 14 | 0.938071 |
| 15 | 0.968744 |
| 16 | 0.971880 |
| 17 | 0.831195 |
| 18 | 0.985863 |
| 19 | 0.949932 |
| 20 | 0.976282 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:01:43

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 8 |
| Learning rate | 5e-05 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.901215 |
| Max R²              | 0.980648 |
| Min R²              | 0.719288 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.913083 |
| 02 | 0.910410 |
| 03 | 0.966215 |
| 04 | 0.935589 |
| 05 | 0.763660 |
| 06 | 0.930233 |
| 07 | 0.931837 |
| 08 | 0.839679 |
| 09 | 0.980648 |
| 10 | 0.719288 |
| 11 | 0.866814 |
| 12 | 0.795513 |
| 13 | 0.945554 |
| 14 | 0.944074 |
| 15 | 0.928536 |
| 16 | 0.965988 |
| 17 | 0.804915 |
| 18 | 0.977495 |
| 19 | 0.932186 |
| 20 | 0.972593 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:02:46

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 16 |
| Learning rate | 0.0001 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.902825 |
| Max R²              | 0.980740 |
| Min R²              | 0.762779 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.884453 |
| 02 | 0.908208 |
| 03 | 0.958042 |
| 04 | 0.936989 |
| 05 | 0.764616 |
| 06 | 0.943304 |
| 07 | 0.956065 |
| 08 | 0.829290 |
| 09 | 0.979577 |
| 10 | 0.782492 |
| 11 | 0.872643 |
| 12 | 0.823382 |
| 13 | 0.944837 |
| 14 | 0.903194 |
| 15 | 0.954351 |
| 16 | 0.973758 |
| 17 | 0.762779 |
| 18 | 0.980740 |
| 19 | 0.924649 |
| 20 | 0.973136 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:05:16

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 4 |
| Learning rate | 0.0001 |
| Adv weight    | 0.01 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.914824 |
| Max R²              | 0.985461 |
| Min R²              | 0.740914 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.929845 |
| 02 | 0.933365 |
| 03 | 0.968916 |
| 04 | 0.945408 |
| 05 | 0.796505 |
| 06 | 0.941841 |
| 07 | 0.945140 |
| 08 | 0.857595 |
| 09 | 0.985329 |
| 10 | 0.740914 |
| 11 | 0.878953 |
| 12 | 0.837857 |
| 13 | 0.940497 |
| 14 | 0.948336 |
| 15 | 0.942244 |
| 16 | 0.972879 |
| 17 | 0.816253 |
| 18 | 0.985461 |
| 19 | 0.947247 |
| 20 | 0.981904 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:06:46

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 8 |
| Learning rate | 0.0001 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.916485 |
| Max R²              | 0.987905 |
| Min R²              | 0.778983 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.939401 |
| 02 | 0.935641 |
| 03 | 0.971015 |
| 04 | 0.951130 |
| 05 | 0.802889 |
| 06 | 0.937303 |
| 07 | 0.948589 |
| 08 | 0.863597 |
| 09 | 0.987905 |
| 10 | 0.778983 |
| 11 | 0.870117 |
| 12 | 0.821808 |
| 13 | 0.938701 |
| 14 | 0.939567 |
| 15 | 0.946582 |
| 16 | 0.973383 |
| 17 | 0.803726 |
| 18 | 0.987559 |
| 19 | 0.946795 |
| 20 | 0.985004 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:08:16

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 8 |
| Learning rate | 0.0001 |
| Adv weight    | 0.1 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.901476 |
| Max R²              | 0.979108 |
| Min R²              | 0.701334 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.914708 |
| 02 | 0.923454 |
| 03 | 0.965547 |
| 04 | 0.932213 |
| 05 | 0.701334 |
| 06 | 0.938981 |
| 07 | 0.945069 |
| 08 | 0.862315 |
| 09 | 0.979108 |
| 10 | 0.728940 |
| 11 | 0.876134 |
| 12 | 0.818911 |
| 13 | 0.930576 |
| 14 | 0.925722 |
| 15 | 0.943432 |
| 16 | 0.969666 |
| 17 | 0.807151 |
| 18 | 0.973495 |
| 19 | 0.925287 |
| 20 | 0.967478 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:09:19

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 16 |
| Learning rate | 0.0005 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.913259 |
| Max R²              | 0.988726 |
| Min R²              | 0.661758 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.920919 |
| 02 | 0.933075 |
| 03 | 0.973329 |
| 04 | 0.943143 |
| 05 | 0.784146 |
| 06 | 0.936640 |
| 07 | 0.931920 |
| 08 | 0.908324 |
| 09 | 0.988726 |
| 10 | 0.661758 |
| 11 | 0.910760 |
| 12 | 0.810508 |
| 13 | 0.940721 |
| 14 | 0.929773 |
| 15 | 0.923011 |
| 16 | 0.967887 |
| 17 | 0.878185 |
| 18 | 0.981213 |
| 19 | 0.955361 |
| 20 | 0.985780 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 09:11:50

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 500 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.919495 |
| Max R²              | 0.988411 |
| Min R²              | 0.752466 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.923939 |
| 02 | 0.928234 |
| 03 | 0.969806 |
| 04 | 0.946478 |
| 05 | 0.784793 |
| 06 | 0.949250 |
| 07 | 0.963116 |
| 08 | 0.863218 |
| 09 | 0.988411 |
| 10 | 0.752466 |
| 11 | 0.901976 |
| 12 | 0.862422 |
| 13 | 0.948414 |
| 14 | 0.941836 |
| 15 | 0.955987 |
| 16 | 0.977168 |
| 17 | 0.819743 |
| 18 | 0.984729 |
| 19 | 0.946450 |
| 20 | 0.981461 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---

## cGAN Parameter Sweep (epochs=500) -- 2026-03-08 09:11:50

| No | Avg R2 | Max R2 | Min R2 | Configuration |
|---|---|---|---|---|
| 9 | 0.919495 | 0.988411 | 0.752466 | lr=5e-5, bs=4,  adv=0.001 |
| 1 | 0.916529 | 0.984703 | 0.748423 | baseline (lr=1e-4, bs=8, adv=0.01) |
| 6 | 0.916485 | 0.987905 | 0.778983 | lr=1e-4, bs=8,  adv=0.001 |
| 5 | 0.914824 | 0.985461 | 0.740914 | lr=1e-4, bs=4,  adv=0.01 |
| 2 | 0.914433 | 0.988522 | 0.725262 | lr=5e-4, bs=8, adv=0.01 |
| 8 | 0.913259 | 0.988726 | 0.661758 | lr=5e-4, bs=16, adv=0.001 |
| 4 | 0.902825 | 0.98074 | 0.762779 | lr=1e-4, bs=16, adv=0.01 |
| 7 | 0.901476 | 0.979108 | 0.701334 | lr=1e-4, bs=8,  adv=0.1 |
| 3 | 0.901215 | 0.980648 | 0.719288 | lr=5e-5, bs=8, adv=0.01 |

**Best (sweep):** `python cgan_cnn.py --lr 0.00005 --batch-size 4 --adv-weight 0.001` -- Avg R2 = 0.919495


---
## cGAN-CNN Run — 2026-03-08 09:21:41

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.923324 |
| Max R²              | 0.988381 |
| Min R²              | 0.768857 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.936118 |
| 02 | 0.934314 |
| 03 | 0.973428 |
| 04 | 0.951142 |
| 05 | 0.768857 |
| 06 | 0.949171 |
| 07 | 0.958212 |
| 08 | 0.887099 |
| 09 | 0.988381 |
| 10 | 0.787633 |
| 11 | 0.898214 |
| 12 | 0.836712 |
| 13 | 0.947902 |
| 14 | 0.947552 |
| 15 | 0.957404 |
| 16 | 0.975712 |
| 17 | 0.850106 |
| 18 | 0.986534 |
| 19 | 0.944703 |
| 20 | 0.987279 |

### File Paths
- Train params: `thermal_analysis_output/training data/params_training.npy`
- Train temps:  `thermal_analysis_output/training data/temps_training.npy`
- Test params:  `thermal_analysis_output/test data/params_testing.npy`
- Test temps:   `thermal_analysis_output/test data/temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 09:56:47

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 32 |
| Test samples  | 8  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.851620 |
| Max R²              | 0.956333 |
| Min R²              | 0.718170 |
| Valid samples       | 8/8 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.917407 |
| 02 | 0.718170 |
| 03 | 0.744923 |
| 04 | 0.933102 |
| 05 | 0.831435 |
| 06 | 0.822917 |
| 07 | 0.888676 |
| 08 | 0.956333 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 10:03:16

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 32 |
| Test samples    | 8  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.795458 |
| Max R²              | 0.952558 |
| Min R²              | 0.476109 |
| Valid samples       | 8/8 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.867452 |
| 02 | 0.476109 |
| 03 | 0.756224 |
| 04 | 0.864758 |
| 05 | 0.697987 |
| 06 | 0.860523 |
| 07 | 0.888049 |
| 08 | 0.952558 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_40\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 10:15:14

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 48 |
| Test samples  | 12  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.931414 |
| Max R²              | 0.988226 |
| Min R²              | 0.869214 |
| Valid samples       | 12/12 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.938178 |
| 02 | 0.947705 |
| 03 | 0.941275 |
| 04 | 0.968839 |
| 05 | 0.908589 |
| 06 | 0.937924 |
| 07 | 0.975119 |
| 08 | 0.988226 |
| 09 | 0.875544 |
| 10 | 0.877243 |
| 11 | 0.949108 |
| 12 | 0.869214 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 10:24:57

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 48 |
| Test samples    | 12  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.930983 |
| Max R²              | 0.990191 |
| Min R²              | 0.782760 |
| Valid samples       | 12/12 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.937148 |
| 02 | 0.966022 |
| 03 | 0.952499 |
| 04 | 0.972966 |
| 05 | 0.916370 |
| 06 | 0.938598 |
| 07 | 0.977980 |
| 08 | 0.990191 |
| 09 | 0.782760 |
| 10 | 0.911577 |
| 11 | 0.948806 |
| 12 | 0.876877 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_60\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 10:40:45

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 64 |
| Test samples  | 16  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.943732 |
| Max R²              | 0.996724 |
| Min R²              | 0.851293 |
| Valid samples       | 16/16 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.890916 |
| 02 | 0.948672 |
| 03 | 0.987921 |
| 04 | 0.944896 |
| 05 | 0.906554 |
| 06 | 0.923362 |
| 07 | 0.851293 |
| 08 | 0.993527 |
| 09 | 0.992918 |
| 10 | 0.990892 |
| 11 | 0.928147 |
| 12 | 0.882964 |
| 13 | 0.955019 |
| 14 | 0.948723 |
| 15 | 0.996724 |
| 16 | 0.957191 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 10:53:41

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 64 |
| Test samples    | 16  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.931468 |
| Max R²              | 0.995943 |
| Min R²              | 0.654203 |
| Valid samples       | 16/16 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.893361 |
| 02 | 0.935218 |
| 03 | 0.989490 |
| 04 | 0.947344 |
| 05 | 0.864663 |
| 06 | 0.929362 |
| 07 | 0.654203 |
| 08 | 0.993277 |
| 09 | 0.995755 |
| 10 | 0.993736 |
| 11 | 0.941072 |
| 12 | 0.888806 |
| 13 | 0.958721 |
| 14 | 0.972201 |
| 15 | 0.995943 |
| 16 | 0.950334 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_80\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 11:13:27

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 80 |
| Test samples  | 20  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.944807 |
| Max R²              | 0.993594 |
| Min R²              | 0.883489 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.924823 |
| 02 | 0.953383 |
| 03 | 0.904781 |
| 04 | 0.883489 |
| 05 | 0.911057 |
| 06 | 0.969035 |
| 07 | 0.909161 |
| 08 | 0.946842 |
| 09 | 0.929470 |
| 10 | 0.971882 |
| 11 | 0.908930 |
| 12 | 0.898952 |
| 13 | 0.971341 |
| 14 | 0.964047 |
| 15 | 0.990939 |
| 16 | 0.939530 |
| 17 | 0.985479 |
| 18 | 0.993594 |
| 19 | 0.974782 |
| 20 | 0.964618 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 11:29:36

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 80 |
| Test samples    | 20  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.892085 |
| Max R²              | 0.980080 |
| Min R²              | 0.782558 |
| Valid samples       | 20/20 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.919774 |
| 02 | 0.908590 |
| 03 | 0.900212 |
| 04 | 0.903326 |
| 05 | 0.858049 |
| 06 | 0.851221 |
| 07 | 0.839962 |
| 08 | 0.891861 |
| 09 | 0.844230 |
| 10 | 0.833153 |
| 11 | 0.782558 |
| 12 | 0.906314 |
| 13 | 0.937181 |
| 14 | 0.868281 |
| 15 | 0.977588 |
| 16 | 0.879877 |
| 17 | 0.980080 |
| 18 | 0.938174 |
| 19 | 0.955124 |
| 20 | 0.866156 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_100\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 11:53:26

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 96 |
| Test samples  | 24  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.956387 |
| Max R²              | 0.991755 |
| Min R²              | 0.801238 |
| Valid samples       | 24/24 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.976296 |
| 02 | 0.952750 |
| 03 | 0.986826 |
| 04 | 0.868446 |
| 05 | 0.964605 |
| 06 | 0.991652 |
| 07 | 0.801238 |
| 08 | 0.984069 |
| 09 | 0.985057 |
| 10 | 0.958390 |
| 11 | 0.916955 |
| 12 | 0.927239 |
| 13 | 0.972448 |
| 14 | 0.940912 |
| 15 | 0.959440 |
| 16 | 0.943237 |
| 17 | 0.927728 |
| 18 | 0.979268 |
| 19 | 0.990788 |
| 20 | 0.982507 |
| 21 | 0.984155 |
| 22 | 0.985103 |
| 23 | 0.982416 |
| 24 | 0.991755 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 12:12:51

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 96 |
| Test samples    | 24  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.938397 |
| Max R²              | 0.998812 |
| Min R²              | 0.591849 |
| Valid samples       | 24/24 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.978895 |
| 02 | 0.912781 |
| 03 | 0.989770 |
| 04 | 0.801693 |
| 05 | 0.967270 |
| 06 | 0.987504 |
| 07 | 0.591849 |
| 08 | 0.977163 |
| 09 | 0.979795 |
| 10 | 0.910388 |
| 11 | 0.871673 |
| 12 | 0.920438 |
| 13 | 0.972508 |
| 14 | 0.928792 |
| 15 | 0.973750 |
| 16 | 0.961408 |
| 17 | 0.878782 |
| 18 | 0.975849 |
| 19 | 0.989318 |
| 20 | 0.986845 |
| 21 | 0.987835 |
| 22 | 0.998812 |
| 23 | 0.989731 |
| 24 | 0.988672 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_120\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 12:40:31

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 112 |
| Test samples  | 28  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.943492 |
| Max R²              | 0.996076 |
| Min R²              | 0.709680 |
| Valid samples       | 28/28 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.932324 |
| 02 | 0.982169 |
| 03 | 0.977510 |
| 04 | 0.944316 |
| 05 | 0.965336 |
| 06 | 0.984719 |
| 07 | 0.979321 |
| 08 | 0.920236 |
| 09 | 0.926179 |
| 10 | 0.961896 |
| 11 | 0.922964 |
| 12 | 0.836321 |
| 13 | 0.978674 |
| 14 | 0.709680 |
| 15 | 0.910901 |
| 16 | 0.930117 |
| 17 | 0.979655 |
| 18 | 0.923181 |
| 19 | 0.988115 |
| 20 | 0.911186 |
| 21 | 0.959143 |
| 22 | 0.971134 |
| 23 | 0.970826 |
| 24 | 0.988931 |
| 25 | 0.919681 |
| 26 | 0.996076 |
| 27 | 0.967070 |
| 28 | 0.980122 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\test data\temps_testing.npy`

---
## cGAN-CNN Run — 2026-03-08 13:03:08

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 112 |
| Test samples    | 28  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.940818 |
| Max R²              | 0.997305 |
| Min R²              | 0.566118 |
| Valid samples       | 28/28 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.939589 |
| 02 | 0.989468 |
| 03 | 0.985998 |
| 04 | 0.899266 |
| 05 | 0.948663 |
| 06 | 0.986310 |
| 07 | 0.983257 |
| 08 | 0.931539 |
| 09 | 0.882964 |
| 10 | 0.972943 |
| 11 | 0.947668 |
| 12 | 0.805818 |
| 13 | 0.989105 |
| 14 | 0.566118 |
| 15 | 0.923561 |
| 16 | 0.952704 |
| 17 | 0.987544 |
| 18 | 0.950564 |
| 19 | 0.991312 |
| 20 | 0.890475 |
| 21 | 0.969716 |
| 22 | 0.980986 |
| 23 | 0.981133 |
| 24 | 0.991743 |
| 25 | 0.946813 |
| 26 | 0.997305 |
| 27 | 0.973839 |
| 28 | 0.976498 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_140\test data\temps_testing.npy`

---
## POD-RBF Run — 2026-03-08 13:34:38

### Hyperparameters
| Param | Value |
|---|---|
| SVD method  | svd |
| POD rank    | 20 |
| RBF kernel  | thin_plate_spline |

### Dataset
| Item | Value |
|---|---|
| Train samples | 128 |
| Test samples  | 32  |
| Param dim     | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.958493 |
| Max R²              | 0.995127 |
| Min R²              | 0.743582 |
| Valid samples       | 32/32 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.963292 |
| 02 | 0.967339 |
| 03 | 0.957690 |
| 04 | 0.944643 |
| 05 | 0.984150 |
| 06 | 0.956264 |
| 07 | 0.959854 |
| 08 | 0.971346 |
| 09 | 0.973524 |
| 10 | 0.912164 |
| 11 | 0.961438 |
| 12 | 0.924297 |
| 13 | 0.974864 |
| 14 | 0.970604 |
| 15 | 0.936465 |
| 16 | 0.951881 |
| 17 | 0.743582 |
| 18 | 0.991358 |
| 19 | 0.992262 |
| 20 | 0.978434 |
| 21 | 0.995127 |
| 22 | 0.938991 |
| 23 | 0.986039 |
| 24 | 0.991074 |
| 25 | 0.993675 |
| 26 | 0.983094 |
| 27 | 0.987496 |
| 28 | 0.940661 |
| 29 | 0.937680 |
| 30 | 0.992026 |
| 31 | 0.941394 |
| 32 | 0.969070 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\test data\temps_testing.npy`

---
## Sample-Size Summary — 2026-03-08

### Fixed Hyperparameters
- POD: `--svd-method svd --rank 20 --rbf-kernel thin_plate_spline`
- cGAN: `--epochs 2000 --lr 0.00005 --batch-size 4 --adv-weight 0.001`
- Split ratio: `train/test = 0.8/0.2`

### Consolidated Results

| Total Samples | Train | Test | POD Avg R² | cGAN Avg R² |
|---:|---:|---:|---:|---:|
| 40  | 32  | 8  | 0.851620 | 0.795458 |
| 60  | 48  | 12 | 0.931414 | 0.930983 |
| 80  | 64  | 16 | 0.943732 | 0.931468 |
| 100 | 80  | 20 | 0.944807 | 0.892085 |
| 120 | 96  | 24 | 0.956387 | 0.938397 |
| 140 | 112 | 28 | 0.943492 | 0.940818 |
| 160 | 128 | 32 | 0.958493 | 0.951346 |

### Best by Model
- Best POD: `160 samples` with Avg R² = `0.958493`
- Best cGAN: `160 samples` with Avg R² = `0.951346`


---
## cGAN-CNN Run — 2026-03-08 14:00:27

### Hyperparameters
| Param | Value |
|---|---|
| Epochs        | 2000 |
| Batch size    | 4 |
| Learning rate | 5e-05 |
| Adv weight    | 0.001 |
| Device        | cuda |

### Dataset
| Item | Value |
|---|---|
| Train samples   | 128 |
| Test samples    | 32  |
| Param dim       | 4 |
| Temp field size | (40000,) |

### Results
| Metric | Value |
|---|---|
| Average R² (finite) | 0.951346 |
| Max R²              | 0.997623 |
| Min R²              | 0.563369 |
| Valid samples       | 32/32 |
| NaN samples         | 0 |

### Per-sample R²

| Sample | R² |
|---|---|
| 01 | 0.982375 |
| 02 | 0.925448 |
| 03 | 0.958907 |
| 04 | 0.968568 |
| 05 | 0.981376 |
| 06 | 0.922819 |
| 07 | 0.968556 |
| 08 | 0.977771 |
| 09 | 0.947782 |
| 10 | 0.840644 |
| 11 | 0.971317 |
| 12 | 0.952117 |
| 13 | 0.957409 |
| 14 | 0.967514 |
| 15 | 0.952525 |
| 16 | 0.964120 |
| 17 | 0.563369 |
| 18 | 0.994752 |
| 19 | 0.994124 |
| 20 | 0.967981 |
| 21 | 0.995959 |
| 22 | 0.945358 |
| 23 | 0.997623 |
| 24 | 0.990397 |
| 25 | 0.993658 |
| 26 | 0.986161 |
| 27 | 0.989742 |
| 28 | 0.952931 |
| 29 | 0.937373 |
| 30 | 0.993714 |
| 31 | 0.935421 |
| 32 | 0.965257 |

### File Paths
- Train params: `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\training data\params_training.npy`
- Train temps:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\training data\temps_training.npy`
- Test params:  `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\test data\params_testing.npy`
- Test temps:   `C:\Users\mkale\Desktop\hou_capita\temperature\thermal_analysis_output_sets\samples_160\test data\temps_testing.npy`

---
## Set-FNO Run v1 (FAILED) — 2026-03-23 23:51

### Model
| Param | Value |
|---|---|
| Architecture | Set Transformer + FNO 2D |
| d_model | 128 |
| num_heads | 4 |
| SAB layers | 2 |
| FNO channels | 32 |
| FNO modes | 12 |
| FNO layers | 4 |
| Total params | 4,425,473 |
| Device | cuda |

### Hyperparameters
| Param | Value |
|---|---|
| Epochs | 2000 (early stopped at 420) |
| Batch size | 8 |
| Learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Early stopping | Yes (patience=300) |

### Dataset (Count Sweep, 1–5 components)
| Item | Value |
|---|---|
| Total samples | 200 (5 groups × 40) |
| Train samples | 144 |
| Val samples | 16 |
| Test samples | 40 |
| Param dim | (N, 5, 3) — [x, y, power] per component, NaN→0 padding |
| Grid size | 100×100 |

### Test Results (1–5 components, seen during training)
| Metric | Value |
|---|---|
| Average R² | 0.5083 |
| Max R² | 0.9961 |
| Min R² | -1.5516 |

### Generalization Results (6/7/8 components, unseen)
| Component Count | Samples | Avg R² |
|---|---|---|
| 6 | 20 | -126.20 |
| 7 | 20 | -181.02 |
| 8 | 20 | -216.82 |
| **Overall** | **60** | **-174.68** |

### Failure Analysis
1. **Training instability**: Loss crashed from 0.006 → 1.01 at epoch 400, early stopping triggered at epoch 420
2. **Model too large for data**: 4.4M parameters vs only 144 training samples — severe overfitting risk
3. **Learning rate too high**: 5e-4 caused gradient explosion mid-training

### File Paths
- Results: `data/set_fno_results/`

---
## Set-FNO Run v2 — 2026-03-24 00:00

### Model
| Param | Value |
|---|---|
| Architecture | Set Transformer + FNO 2D |
| d_model | 64 |
| num_heads | 4 |
| SAB layers | 1 |
| FNO channels | 16 |
| FNO modes | 8 |
| FNO layers | 2 |
| Device | cuda |

### Hyperparameters
| Param | Value |
|---|---|
| Epochs | 3000 (ran all) |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping | Yes (patience=500, not triggered) |

### Dataset (Count Sweep, 1–5 components)
| Item | Value |
|---|---|
| Total samples | 200 (5 groups × 40) |
| Train samples | 144 |
| Val samples | 16 |
| Test samples | 40 |
| Param dim | (N, 5, 3) — [x, y, power] per component, NaN→0 padding |
| Grid size | 100×100 |

### Test Results (1–5 components, seen during training)
| Metric | Value |
|---|---|
| Average R² | **0.9727** |
| Max R² | 0.9961 |
| Min R² | 0.8803 |
| All 40 samples | R² > 0.88 |

### Per-sample R² (Test)

| Sample | R² | Sample | R² | Sample | R² | Sample | R² |
|---|---|---|---|---|---|---|---|
| 01 | 0.9887 | 11 | 0.9896 | 21 | 0.9926 | 31 | 0.9929 |
| 02 | 0.9926 | 12 | 0.9775 | 22 | 0.9858 | 32 | 0.8803 |
| 03 | 0.9915 | 13 | 0.9935 | 23 | 0.9926 | 33 | 0.9961 |
| 04 | 0.9817 | 14 | 0.9847 | 24 | 0.9406 | 34 | 0.9957 |
| 05 | 0.9638 | 15 | 0.9858 | 25 | 0.9960 | 35 | 0.9613 |
| 06 | 0.9662 | 16 | 0.9954 | 26 | 0.9855 | 36 | 0.9203 |
| 07 | 0.9740 | 17 | 0.9893 | 27 | 0.9939 | 37 | 0.9618 |
| 08 | 0.9279 | 18 | 0.9905 | 28 | 0.9941 | 38 | 0.9378 |
| 09 | 0.9692 | 19 | 0.9597 | 29 | 0.9835 | 39 | 0.9874 |
| 10 | 0.9798 | 20 | 0.9896 | 30 | 0.9029 | 40 | 0.9169 |

### Generalization Results (6/7/8 components, unseen) — FAILED
| Component Count | Samples | Avg R² |
|---|---|---|
| 6 | 20 | -107.56 |
| 7 | 20 | -219.37 |
| 8 | 20 | -215.68 |
| **Overall** | **60** | **-180.87** |

### Failure Analysis — Generalization
1. **Distribution shift**: Training data has 1–5 components (total power 2.2–13.7W); test data has 6–8 components (total power ~16–23W). The FNO decoder learned temperature magnitude patterns specific to the training power range.
2. **Temperature scale mismatch**: StandardScaler was fitted on 1–5 component temperature fields (25–76°C). Inverse transform for 6–8 component inputs (which should produce 80–100+°C) outputs wildly wrong values.
3. **Zero-padding semantics**: Absent components are zero-padded (x=0, y=0, p=0). When the model sees 6+ components all with non-zero values, the input structure differs from what was seen during training where many slots were always zero.
4. **Set Transformer limitation**: While theoretically invariant to set size, the PMA pooling with a single learned seed vector may compress the extra component signals in ways not generalizable.

### Conclusion
- **Within-distribution (1–5 components): Excellent (R²=0.97)**
- **Out-of-distribution (6–8 components): Total failure**
- **Next steps**: Train on 1–7 components, test on 8; or use physics-informed normalization (divide temp field by total power)

### File Paths
- Train/Test params: `data/thermal_analysis_output_count_sweep/params_count_sweep.npy`
- Train/Test temps: `data/thermal_analysis_output_count_sweep/temps_count_sweep.npy`
- Gen-test params: `data/thermal_analysis_output_count_sweep_gen/params_count_sweep.npy`
- Gen-test temps: `data/thermal_analysis_output_count_sweep_gen/temps_count_sweep.npy`
- Results v1: `data/set_fno_results/`
- Results v2: `data/set_fno_results_v2/`

---
## Set-FNO Run v3 (Physics Norm) — 2026-03-24

### Key Changes from v2
1. **Physics-informed normalization**: `T_norm = (T - 25°C) / P_total` — thermal impedance map (°C/W), making different component counts comparable
2. **2.5× more training data**: 500 samples (5 groups × 100) vs 200 (5 groups × 40)

### Model
| Param | Value |
|---|---|
| Architecture | Set Transformer + FNO 2D |
| d_model | 64 |
| num_heads | 4 |
| SAB layers | 1 |
| FNO channels | 16 |
| FNO modes | 8 |
| FNO layers | 2 |
| Device | cuda |

### Hyperparameters
| Param | Value |
|---|---|
| Epochs | 3000 (early stopped at 1901) |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping | Yes (patience=500) |
| Physics norm | **Yes** — `T_norm = (T - T_amb) / P_total` |
| T_ambient | 25.0 °C |

### Dataset (Count Sweep, 1–5 components)
| Item | Value |
|---|---|
| Total samples | **500** (5 groups × 100) |
| Train samples | 360 |
| Val samples | 40 |
| Test samples | 100 |
| Param dim | (N, 5, 3) — [x, y, power] per component, NaN→0 padding |
| Grid size | 100×100 |

### Test Results (1–5 components, seen during training)
| Metric | Value |
|---|---|
| Average R² | **0.9910** |
| Min R² | 0.9389 |
| Max R² | 0.9998 |
| All 100 samples | R² > 0.93 |

### Generalization Results (6/7/8 components, unseen) — **SUCCESS**
| Component Count | Samples | Avg R² |
|---|---|---|
| 6 | 20 | **0.9128** |
| 7 | 20 | **0.9291** |
| 8 | 20 | **0.8420** |
| **Overall** | **60** | **0.8946** |

### Improvement vs v2
| Metric | v2 (no physics norm, 200 samples) | v3 (physics norm, 500 samples) |
|---|---|---|
| Test R² (1–5) | 0.9727 | **0.9910** |
| Gen R² (6–8) | -180.87 | **0.8946** |

### Analysis
- **Physics normalization was the key breakthrough**: By dividing `(T - T_ambient)` by total power, the model learns thermal impedance patterns that naturally scale with any number of components. The thermal impedance field has a consistent range (~0–5 °C/W) regardless of component count.
- **More data helped**: 500 vs 200 samples improved in-distribution R² from 0.9727 to 0.9910.
- **8-component prediction is hardest**: R²=0.842 for 8 components vs 0.913/0.929 for 6/7. This makes sense as 8 components create more complex interaction patterns the model hasn't seen.
- **Early stopping at epoch 1901** (patience=500) indicates good convergence with no overfitting.

### File Paths
- Train/Test params: `data/thermal_analysis_output_count_sweep/params_count_sweep.npy` (500 samples)
- Train/Test temps: `data/thermal_analysis_output_count_sweep/temps_count_sweep.npy`
- Gen-test params: `data/thermal_analysis_output_count_sweep_gen/params_count_sweep.npy` (60 samples)
- Gen-test temps: `data/thermal_analysis_output_count_sweep_gen/temps_count_sweep.npy`
- Results: `data/set_fno_results_v3/`
- Model: `set_fno_model_v3.pth`

---
## Set-FNO Data Pipeline & Train/Test Split Notes

### Data Pipeline Comparison

| Model | Data Flow |
|---|---|
| POD-RBF / cGAN-CNN | JSON files → `process_json_to_grid.py` (scipy.griddata interpolation) → `.npy` → Model |
| Set-FNO | `thermal_prediction_error.py --count-sweep` (SOR solver direct 100×100 grid output) → `.npy` → Model |

**Set-FNO does NOT use `process_json_to_grid.py`**. The simulation script (`thermal_prediction_error.py`) directly outputs the temperature field as a 100×100 grid from the SOR solver, saved as `params_count_sweep.npy` and `temps_count_sweep.npy`. No additional preprocessing or interpolation is needed.

### Train/Test Split Strategy (inside `set_fno_thermal.py`)

All splitting is done automatically inside `load_count_sweep_data()` using `sklearn.model_selection.train_test_split`:

| Split | Ratio | Method | Seed |
|---|---|---|---|
| Test set | 20% of total | Stratified by component count | 42 |
| Validation set | 10% of training set | Stratified by component count | 42 |
| Training set | Remaining ~72% | — | — |

**Stratified sampling** ensures each component count (1, 2, 3, 4, 5) is proportionally represented in every split.

For the v3 run (500 samples, 100 per component count):

| Split | Approx. Samples | Purpose |
|---|---|---|
| Training | ~360 | Model weight updates |
| Validation | ~40 | Early stopping criterion |
| Test (in-distribution) | 100 | Evaluate on 1–5 components |
| Generalization test | 60 (separate file) | Evaluate on unseen 6–8 components |

The **generalization test set** (6–8 components, 20 each) is loaded from a completely separate `.npy` file via `--gen-test-params` / `--gen-test-temps` and never participates in training or validation.
