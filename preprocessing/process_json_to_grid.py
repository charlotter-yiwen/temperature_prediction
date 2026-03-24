import os
import json
import re
import shutil
import numpy as np
from scipy.interpolate import griddata
import argparse
import matplotlib.pyplot as plt
#python process_json_to_grid.py "thermal_analysis_output" --out temps.npy --params-out params.npy

def process_folder(folder_path, out_name="square_fin_test.npy", params_out_name="params.npy", xi_count=200, yi_count=300, subsample=2):
    # gather json files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    temperature_matrix = []
    params_matrix = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize to list of points
        points = None
        if isinstance(data, dict):
            # try common keys
            for key in ('data', 'points', 'temperatures', 'temp_data'):
                if key in data and isinstance(data[key], list):
                    points = data[key]
                    break
            # if not found, maybe the dict itself is a single record list
            if points is None:
                # try to detect list-like values
                if all(isinstance(v, (int, float, str)) for v in data.values()):
                    # single point
                    points = [data]
        elif isinstance(data, list):
            points = data

        if points is None or len(points) == 0:
            print(f"Skipping {file_name}: no point list found")
            continue

        # extract x,y,(z),temperature
        xs = []
        ys = []
        zs = []
        temps = []
        for p in points:
            # p can be list/tuple or dict
            if isinstance(p, (list, tuple)):
                # try to map as [z,x,y,temp] or [x,y,temp]
                if len(p) >= 4:
                    z_val = float(p[0]); x_val = float(p[1]); y_val = float(p[2]); t_val = float(p[3])
                elif len(p) >= 3:
                    z_val = 0.0; x_val = float(p[0]); y_val = float(p[1]); t_val = float(p[2])
                else:
                    continue
            elif isinstance(p, dict):
                x_val = p.get('x') if 'x' in p else p.get('X') if 'X' in p else p.get('pos_x') if 'pos_x' in p else None
                y_val = p.get('y') if 'y' in p else p.get('Y') if 'Y' in p else p.get('pos_y') if 'pos_y' in p else None
                t_val = p.get('temperature') if 'temperature' in p else p.get('temp') if 'temp' in p else p.get('T') if 'T' in p else None
                z_val = p.get('z') if 'z' in p else 0.0
                if x_val is None or y_val is None or t_val is None:
                    # attempt to find numeric values in dict values
                    vals = [v for v in p.values() if isinstance(v, (int, float))]
                    if len(vals) >= 3:
                        x_val, y_val, t_val = vals[0], vals[1], vals[2]
                    else:
                        continue
            else:
                continue

            xs.append(float(x_val))
            ys.append(float(y_val))
            zs.append(float(z_val))
            temps.append(float(t_val))

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        temps = np.array(temps)

        if xs.size == 0:
            print(f"Skipping {file_name}: no numeric points extracted")
            continue

        # subsample
        xs = xs[::subsample]
        ys = ys[::subsample]
        temps = temps[::subsample]

        # create grid
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xi = np.linspace(x_min, x_max, xi_count)
        yi = np.linspace(y_min, y_max, yi_count)
        xi, yi = np.meshgrid(xi, yi)

        try:
            temperature_grid = griddata((xs, ys), temps, (xi, yi), method='nearest')
        except Exception as e:
            print(f"Interpolation failed for {file_name}: {e}")
            continue

        # save heatmap PNG next to the JSON file
        try:
            fig = plt.figure(figsize=(8, 4))
            im = plt.imshow(temperature_grid, origin='lower', cmap='jet',
                            extent=[x_min, x_max, y_min, y_max], aspect='auto')
            plt.title(file_name)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            cbar = plt.colorbar(im)
            cbar.set_label('Temperature')
            png_path = os.path.splitext(file_path)[0] + '.png'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Failed to save PNG for {file_name}: {e}")

        temperature_matrix.append(temperature_grid.flatten(order='C'))
        # parse filename parameters like "a,b,c,d.json"
        try:
            stem = os.path.splitext(file_name)[0]
            parts = stem.split(',')
            nums = [float(p) for p in parts]
            if len(nums) != 4:
                raise ValueError("expected 4 numbers in filename")
            params_matrix.append(nums)
        except Exception:
            params_matrix.append([np.nan, np.nan, np.nan, np.nan])
            print(f"Warning: could not parse 4 numeric params from {file_name}; filled with NaN")
        print(f"Processed file: {file_name} -> saved {png_path}")

    if len(temperature_matrix) == 0:
        print("No valid JSON files processed.")
        return

    temperature_matrix = np.array(temperature_matrix)
    params_matrix = np.array(params_matrix)
    print(temperature_matrix.shape)
    print(params_matrix.shape)
    try:
        np.save(out_name, temperature_matrix)
        print(f"Saved matrix to {out_name}")
        np.save(params_out_name, params_matrix)
        print(f"Saved parameters to {params_out_name}")
    except PermissionError:
        fallback = os.path.join(os.path.expanduser('~'), 'Documents', out_name)
        try:
            np.save(fallback, temperature_matrix)
            print(f"Permission denied writing {out_name}; saved to {fallback} instead")
            params_fallback = os.path.join(os.path.expanduser('~'), 'Documents', params_out_name)
            np.save(params_fallback, params_matrix)
            print(f"Saved parameters to {params_fallback}")
        except Exception as e:
            print(f"Failed to save to fallback path {fallback}: {e}")
            raise


def split_and_process(folder_path, train_ratio=0.8, xi_count=200, yi_count=300, subsample=2):
    """Split JSON files into training/test subfolders and process both.
    Saves temps_training.npy / params_training.npy under training data,
    temps_testing.npy / params_testing.npy under test data.
    """
    # Only include files whose stem is purely numeric parameters, e.g. "1,2,-3,4.json"
    _param_re = re.compile(r'^-?\d+(?:,-?\d+)*$')
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.json') and _param_re.match(os.path.splitext(f)[0])
    ]
    files.sort()
    if not files:
        print("No JSON files to split.")
        return

    train_dir = os.path.join(folder_path, 'training data')
    test_dir = os.path.join(folder_path, 'test data')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # clear old jsons in subfolders to avoid mixing
    for target_dir in (train_dir, test_dir):
        for name in os.listdir(target_dir):
            if name.lower().endswith('.json'):
                try:
                    os.remove(os.path.join(target_dir, name))
                except OSError:
                    pass

    # deterministic split
    n_total = len(files)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_total
    train_files = files[:n_train]
    test_files = files[n_train:]
    if not test_files and n_total > 1:
        # ensure at least one test sample
        test_files = [train_files.pop()]

    for name in train_files:
        shutil.copy2(os.path.join(folder_path, name), os.path.join(train_dir, name))
    for name in test_files:
        shutil.copy2(os.path.join(folder_path, name), os.path.join(test_dir, name))

    print(f"Split {n_total} files -> train {len(train_files)}, test {len(test_files)}")

    process_folder(train_dir,
                   out_name=os.path.join(train_dir, 'temps_training.npy'),
                   params_out_name=os.path.join(train_dir, 'params_training.npy'),
                   xi_count=xi_count, yi_count=yi_count, subsample=subsample)

    process_folder(test_dir,
                   out_name=os.path.join(test_dir, 'temps_testing.npy'),
                   params_out_name=os.path.join(test_dir, 'params_testing.npy'),
                   xi_count=xi_count, yi_count=yi_count, subsample=subsample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON temperature files into interpolated grids')
    parser.add_argument('folder', help='Folder containing JSON files')
    parser.add_argument('--out', default='square_fin_test.npy', help='Output .npy filename')
    parser.add_argument('--params-out', default='params.npy', help='Output .npy filename for parsed parameters')
    parser.add_argument('--xi', type=int, default=200, help='Grid x resolution')
    parser.add_argument('--yi', type=int, default=200, help='Grid y resolution')
    parser.add_argument('--subsample', type=int, default=2, help='Subsample step for input points')
    parser.add_argument('--split', action='store_true', help='Split JSONs into training/test subsets and process both')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training split ratio (0-1)')
    args = parser.parse_args()
    if args.split:
        split_and_process(args.folder, train_ratio=args.train_ratio, xi_count=args.xi, yi_count=args.yi, subsample=args.subsample)
    else:
        process_folder(args.folder, out_name=args.out, params_out_name=args.params_out, xi_count=args.xi, yi_count=args.yi, subsample=args.subsample)
