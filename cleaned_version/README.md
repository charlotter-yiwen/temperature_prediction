# Cleaned Version - Temperature Prediction Project

## Folder Structure

```
cleaned_version/
├── models/
│   ├── shared/                    # Shared model architectures
│   │   └── set_fno_thermal.py     # SetFNO model definition
│   ├── v1/                        # V1: PINN-based models (March 2026)
│   ├── v2/                        # V2: SetFNO models (April 2026)
│   └── v3/                        # V3: SetFNO V3 2ch/3ch (May 2026)
│       ├── train_setfno_v3.py         # V3 2-channel training
│       ├── train_setfno_v3_3ch.py     # V3 3-channel training (latest)
│       ├── gen_test_v3.py             # Generalization test 6-10 comp
│       ├── gen_test_plan_a.py         # PlanA distill generalization test
│       └── generate_random_10comp.py  # Generate 10-component test data
│
├── data/
│   ├── v1/                        # V1 data (1-5 components)
│   │   └── training/              # 342 JSON files
│   └── v2/                        # V2 data (30W JSON format)
│       ├── training/              # 600 JSON files (1-5 components)
│       └── generalization/        # 180 JSON files (6-8 components)
│
├── outputs/
│   ├── v1/
│   │   ├── model_v3/              # Model v3 training results
│   │   └── hp_search/             # Hyperparameter search (16 experiments)
│   ├── v2/
│   │   └── model_30w/             # Model 30W results + generalization
│   └── v3/                        # V3 results (10 experiment variants)
│       ├── results_v3/            # Base V3
│       ├── results_v3_3ch_balanced/   # 3ch balanced (R²=0.9898)
│       ├── results_v3_poweraug_fixed/ # Power augmentation (fixed)
│       ├── results_v3_targeted_aug/   # Targeted augmentation
│       ├── results_v3_two_phase/      # Two-phase training
│       ├── results_v3_distill/        # Teacher distillation
│       └── ...                        # More variants
│
├── my_scripts/
│   ├── v1/                        # V1 utility scripts
│   ├── v2/                        # V2 utility scripts (copper via etc.)
│   └── v3/                        # V3 utility scripts
│       ├── run_training.py            # Training wrapper
│       └── run_3ch_balanced.bat       # Batch launcher
│
├── simulation/
│   └── thermal_prediction.py
├── preprocessing/
│   └── process_json_to_grid.py
└── scripts/
```

## Version Comparison

| Feature | v1 (March) | v2 (April) | v3 (May) |
|---------|-----------|-----------|----------|
| Model | PINN | SetFNO | SetFNO V3 (2ch/3ch) |
| Parameters | 5.7-42.8M | 42.8M | 32.6M |
| Input | Coordinates + power | Coordinates + power | Heat source map + grid |
| Key Innovation | Physics loss | Physics norm | Power aug, distillation |
| Training Data | 1-5 comp | 1-5 comp (30W) | 1-5 comp + augmented |
| Generalization | 6-9 comp | 6-8 comp | 6-10 comp |

## Best Results per Version

### v1: Plan A Balanced + Physics Loss
- Location: `outputs/v1/hp_search/plan_a_balanced/`
- 6-comp R²=0.91, 7-comp=0.88, 8-comp=0.86

### v2: SetFNO Phase 2
- Location: `outputs/v2/model_30w/results_setfno_phase2/`
- 6-comp R²=0.93, 7-comp=0.93, 8-comp=0.91

### v3: Distillation (Best V3 so far)
- Location: `outputs/v3/results_v3_distill/`
- 6-comp R²=0.90, 7-comp=0.87, 8-comp=0.84, 9-comp=0.88, 10-comp=0.74
- 3ch Balanced training R²=0.9898 (on training data)

## Usage

### Training V3 Model
```bash
cd models/v3
python train_setfno_v3_3ch.py --epochs 2000 --batch-size 32 --lr 5e-5
```

### Testing V3 Generalization
```bash
cd models/v3
python gen_test_v3.py --model-path ../../outputs/v3/results_v3_distill/setfno_v3.pth
```

### Training V2 Model (30W)
```bash
cd models/v2
python train_setfno_30w.py --data-dir ../../data/v2/training --epochs 2000 --physics-norm
```
