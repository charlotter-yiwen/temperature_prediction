"""Wrapper to run training with logging"""
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set args
sys.argv = [
    "train_setfno_v3_3ch.py",
    "--params", "training_data/params_count_sweep.npy",
    "--temps",  "training_data/temps_count_sweep.npy",
    "--epochs", "2000",
    "--batch-size", "32",
    "--lr", "5e-5",
    "--lambda-bc", "0.001",
    "--lambda-pde", "0.0001",
    "--power-aug-copies", "2",
    "--power-aug-min", "0.8",
    "--power-aug-max", "2.0",
    "--early-stopping",
    "--patience", "200",
    "--out-dir", "my_scripts/results_v3_3ch_balanced",
]

# Redirect stdout to log file
import io

log_path = "my_scripts/results_v3_3ch_balanced_log.txt"
os.makedirs("my_scripts", exist_ok=True)

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, data):
        for w in self.writers:
            w.write(data)
            w.flush()
    def flush(self):
        for w in self.writers:
            w.flush()

log_file = open(log_path, "w", buffering=1)
tee = TeeWriter(sys.__stdout__, log_file)
sys.stdout = tee
sys.stderr = tee

print(f"Starting training, logging to {log_path}")

from my_scripts.train_setfno_v3_3ch import parse_args, train, test_only

args = parse_args()
print(f"Args: {vars(args)}")

if args.test_only:
    test_only(args)
else:
    train(args)

log_file.close()
