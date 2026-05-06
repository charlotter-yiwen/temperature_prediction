@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
python my_scripts/train_setfno_v3_3ch.py ^
    --params training_data/params_count_sweep.npy ^
    --temps  training_data/temps_count_sweep.npy ^
    --epochs 2000 --batch-size 32 --lr 5e-5 ^
    --lambda-bc 1e-3 --lambda-pde 1e-4 ^
    --power-aug-copies 2 --power-aug-min 0.8 --power-aug-max 2.0 ^
    --early-stopping --patience 200 ^
    --out-dir my_scripts/results_v3_3ch_balanced
