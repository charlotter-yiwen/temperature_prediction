"""
convert_json_to_npy.py
======================
Convert JSON thermal data files to .npy format for set_fno_thermal.py
- Parse component positions from filenames (absolute coordinates)
- Split into train/val/test sets (80/10/10)
- Save as .npy files ready for training
"""

import os
import json
import re
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\data"
OUTPUT_DIR = r"c:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\training_data"
GRID_SIZE = 100  # Grid size is 100x100 based on JSON data (10000 points)
BOARD_SIZE = 100.0  # mm, PCB board size
# 组件功率分布：U1=2.5W, U2=2.2W, U3=3.0W, U4=2.8W, U5=3.2W
POWERS = [2.5, 2.2, 3.0, 2.8, 3.2]


def parse_filename_get_positions(filename):
    """Extract component count and positions from filename.

    Filename format: countN_idxMM_X1,Y1,X2,Y2,...,.json
    Returns: (n_components, [[x1,y1,p1], [x2,y2,p2], ...])
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Match countN_idxNN
    count_match = re.match(r'count(\d+)_idx\d+', name)
    if not count_match:
        return None, None

    n_components = int(count_match.group(1))

    # Extract the coordinate part after the last underscore
    # Format: X1,Y1,X2,Y2,... (integers representing mm)
    parts = name.split('_')
    coord_part = parts[-1]

    # Parse coordinates (X,Y pairs)
    coords_str = coord_part.split(',')
    positions = []
    for i in range(0, len(coords_str), 2):
        if i + 1 < len(coords_str):
            x = int(coords_str[i])
            y = int(coords_str[i + 1])
            # Use actual power for each component
            positions.append([x, y, POWERS[i // 2]])

    return n_components, positions


def load_json_temp_field(json_path):
    """Load temperature field from JSON file and reshape to 100x100 grid."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract temperatures and reshape to GRID_SIZE x GRID_SIZE
    temps = np.array([item['temperature'] for item in data])
    temps = temps.reshape(GRID_SIZE, GRID_SIZE)
    return temps


def find_max_components(data_dir):
    """Find the maximum number of components across all files."""
    max_comp = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            n_comp, _ = parse_filename_get_positions(filename)
            if n_comp is not None:
                max_comp = max(max_comp, n_comp)
    return max_comp


def main():
    # Find all JSON files (exclude non-data JSON files)
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')
                  and not f.endswith('_manifest.json')
                  and not f.endswith('_schema.json')]
    print(f"Found {len(json_files)} JSON files")

    # Find max components to set array dimensions
    max_components = find_max_components(DATA_DIR)
    print(f"Max components in dataset: {max_components}")

    # Arrays to store data
    n_samples = len(json_files)
    d_per_comp = 3  # x, y, power

    params = np.full((n_samples, max_components, d_per_comp), np.nan, dtype=np.float32)
    temps = np.zeros((n_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    comp_counts = np.zeros(n_samples, dtype=np.int32)
    valid_flags = np.zeros(n_samples, dtype=bool)

    print("Loading data from JSON files...")

    for idx, filename in enumerate(json_files):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{n_samples}...")

        json_path = os.path.join(DATA_DIR, filename)

        # Get component positions from filename
        n_comp, positions = parse_filename_get_positions(filename)
        if n_comp is None:
            print(f"  Warning: Could not parse {filename}")
            continue

        # Fill in params (NaN for unused component slots)
        for i, pos in enumerate(positions):
            params[idx, i, 0] = pos[0]  # x (mm)
            params[idx, i, 1] = pos[1]  # y (mm)
            params[idx, i, 2] = pos[2]  # power (W)

        comp_counts[idx] = n_comp
        valid_flags[idx] = True

        # Load temperature field
        temps[idx] = load_json_temp_field(json_path)

    print(f"Loaded {n_samples} samples total")

    # Filter out invalid samples
    valid_mask = valid_flags
    n_valid = valid_mask.sum()
    print(f"Valid samples: {n_valid} out of {n_samples}")

    if n_valid < n_samples:
        params = params[valid_mask]
        temps = temps[valid_mask]
        comp_counts = comp_counts[valid_mask]
        json_files = [f for f, v in zip(json_files, valid_mask) if v]

    n_samples = n_valid

    # Split: first 80% train, 10% val, 10% test
    # Using stratified split on component count

    # Check which component counts have enough samples for stratification
    unique, counts = np.unique(comp_counts, return_counts=True)
    print(f"Component count distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # For small classes, we can't stratify - use regular split
    min_samples_per_class = 2
    can_stratify = all(c >= min_samples_per_class for c in counts)

    if can_stratify:
        # First split: 90% train+val, 10% test
        train_val_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.1,
            random_state=42,
            stratify=comp_counts
        )

        # Second split: 88.89% of train_val = 80% of total, 11.11% = 10% of total
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.111,  # 0.1 / 0.9 = 0.111...
            random_state=42,
            stratify=comp_counts[train_val_idx]
        )
    else:
        # Fallback: non-stratified split
        print("Warning: Some component classes have too few samples, using non-stratified split")
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_test = int(0.1 * n_samples)
        n_val = int(0.1 * n_samples)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

    print(f"\nDataset split:")
    print(f"  Training samples:   {len(train_idx)} ({100*len(train_idx)/n_samples:.1f}%)")
    print(f"  Validation samples: {len(val_idx)} ({100*len(val_idx)/n_samples:.1f}%)")
    print(f"  Test samples:       {len(test_idx)} ({100*len(test_idx)/n_samples:.1f}%)")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as .npy files (count-sweep format for set_fno_thermal)
    # File format: params_count_sweep.npy, temps_count_sweep.npy
    params_path = os.path.join(OUTPUT_DIR, "params_count_sweep.npy")
    temps_path = os.path.join(OUTPUT_DIR, "temps_count_sweep.npy")

    np.save(params_path, params)
    # Save temps as 1D array (n_samples, H*W) as expected by set_fno_thermal.py
    temps_flat = temps.reshape(n_samples, -1)
    np.save(temps_path, temps_flat)

    print(f"\nSaved:")
    print(f"  {params_path} -> shape {params.shape}")
    print(f"  {temps_path} -> shape {temps_flat.shape}  (flattened: n_samples x H*W)")

    # Also save train/val/test indices for reference
    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist()
    }
    splits_path = os.path.join(OUTPUT_DIR, "data_splits.json")
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"  {splits_path}")

    print("\n" + "="*60)
    print("Data conversion complete!")
    print("="*60)
    print(f"\nTo train the model, run:")
    print(f"  python models/set_fno_thermal.py \\")
    print(f"      --count-sweep-params \"{params_path}\" \\")
    print(f"      --count-sweep-temps \"{temps_path}\" \\")
    print(f"      --n-components {max_components} \\")
    print(f"      --d-per-comp 3 \\")
    print(f"      --epochs 500 \\")
    print(f"      --batch-size 8 \\")
    print(f"      --val-ratio 0.1 \\")
    print(f"      --out-dir set_fno_results")


if __name__ == "__main__":
    main()