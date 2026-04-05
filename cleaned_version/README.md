# Cleaned Version - Temperature Prediction Project

## Folder Structure

```
cleaned_version/
├── models/
│   ├── shared/                    # Shared model architectures
│   │   └── set_fno_thermal.py     # SetFNO model definition
│   ├── v1/                        # Version 1: PINN-based models (March 2026)
│   │   ├── train_pinn_physics_fix.py
│   │   ├── train_plan_a_physics.py
│   │   ├── train_plan_a_physics_v2.py
│   │   ├── predict_pinn_gen_physics_fix.py
│   │   ├── predict_plan_a_physics_gen.py
│   │   └── plot_gen_comparison.py
│   └── v2/                        # Version 2: SetFNO models (April 2026)
│       ├── train_setfno_30w.py    # Main training script for 30W dataset
│       ├── train_pinn_30w.py
│       ├── train_pinn_phase2.py
│       ├── pinn_v3_physics_fix.py
│       └── test_generalization.py
│
├── data/
│   ├── v1/                        # Version 1 data (1-5 components)
│   │   ├── training/              # 342 JSON files
│   │   ├── test/
│   │   └── validation/
│   └── v2/                        # Version 2 data (30W JSON format)
│       ├── training/              # 600 JSON files (1-5 components)
│       └── generalization/        # 180 JSON files (6-8 components)
│
├── outputs/
│   ├── v1/
│   │   ├── model_v3/              # Model v3 training results
│   │   │   ├── results_bc_0_0005_10k/
│   │   │   └── results_plan_a_physics/
│   │   └── hp_search/             # Hyperparameter search results
│   │       ├── plan_a_balanced/   # Best v1 model (42.8M params)
│   │       ├── plan_b_transfomer/
│   │       ├── plan_c_fno/
│   │       └── ...
│   └── v2/
│       └── model_30w/             # Model 30W training results
│           ├── results_setfno_phase2/  # Best v2 model
│           ├── generalization_results/
│           └── ...
│
├── my_scripts/
│   ├── v1/                        # Version 1 utility scripts
│   │   ├── compare_methods.py
│   │   ├── generate_random_6comp.py
│   │   ├── generate_random_9comp.py
│   │   ├── predict_7comp.py
│   │   ├── predict_8comp.py
│   │   └── ...
│   ├── v2/                        # Version 2 utility scripts
│   │   ├── generate_1to5comp_random_power.py
│   │   ├── simulate_copper_via_temp.py
│   │   ├── visualize_copper_via.py
│   │   └── ...
│   └── IEEE_Writing_Style/        # IEEE paper template
│
├── simulation/
│   └── thermal_prediction.py      # Thermal simulation code
│
├── preprocessing/
│   └── process_json_to_grid.py    # Data preprocessing
│
└── scripts/
    └── *.ps1                      # PowerShell scripts
```

## Version Differences

| Feature | v1 (March 2026) | v2 (April 2026) |
|---------|-----------------|-----------------|
| Model | PINN-based | SetFNO (SetTransformer + FNO) |
| Parameters | 5.7M - 42.8M | 42.8M |
| Training Data | 1-5 components | 1-5 components (30W format) |
| Power Range | 2.5-13.7W | 2.5-6W per component |
| Data Format | NumPy | JSON |
| Generalization | 6-9 components | 6-8 components |

## Best Models

### v1: Plan A Balanced + Physics Loss
- Location: `outputs/v1/model_v3/results_plan_a_physics/`
- Parameters: 42.8M
- Generalization R²: 6-comp=0.92, 7-comp=0.91, 8-comp=0.90, 9-comp=0.44

### v2: SetFNO Phase 2
- Location: `outputs/v2/model_30w/results_setfno_phase2/`
- Parameters: 42.8M
- Generalization R²: 6-comp=0.93, 7-comp=0.93, 8-comp=0.91

## Usage

### Training v2 Model
```bash
cd models/v2
python train_setfno_30w.py --data-dir ../../data/v2/training --epochs 2000 --physics-norm
```

### Testing Generalization
```bash
cd models/v2
python test_generalization.py --model-path ../../outputs/v2/model_30w/results_setfno_phase2/setfno_30w_phase2.pth --data-dir ../../data/v2/generalization --component-count 6
```
