"""
Generate 1-5 component thermal maps with random power (3-6W) per component.
Each sample produces one JSON and one PNG file.
"""
import sys
sys.path.insert(0, r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\simulation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import random
import math
import time

from thermal_prediction_error import (
    pcb_2d_thermal_simulation_sor_optimized,
    plot_temperature,
    generate_temperature_json_data,
    rects_overlap
)

# Configuration
GRID_SIZE = 100
AMBIENT_TEMP = 25.0
PCB_SIZE_MM = 100.0
COMPONENT_SIZE_MM = 8.0  # 8x8mm components
POWER_MIN = 3.0  # Watts
POWER_MAX = 6.0  # Watts
MAX_ITERATIONS = 50000
TOLERANCE = 1e-9
OMEGA = 1.98
OUTPUT_DIR = r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test'

# Samples per component count
SAMPLES_PER_COUNT = {
    1: 20,
    2: 20,
    3: 20,
    4: 20,
    5: 20
}

def random_power():
    """Generate random power between POWER_MIN and POWER_MAX"""
    return random.uniform(POWER_MIN, POWER_MAX)

def generate_non_overlapping_layout(num_components, pcb_size=100.0, comp_size=8.0, max_attempts=500):
    """
    Generate non-overlapping component layout.
    Returns list of dicts with x_min, x_max, y_min, y_max, power.
    """
    components = []
    margin = 5.0  # margin from PCB edges

    for i in range(num_components):
        attempts = 0
        while attempts < max_attempts:
            x = random.uniform(margin, pcb_size - margin - comp_size)
            y = random.uniform(margin, pcb_size - margin - comp_size)

            new_rect = (x, x + comp_size, y, y + comp_size)

            # Check overlap with existing components
            overlap = False
            for existing in components:
                existing_rect = (existing['x_min'], existing['x_max'],
                               existing['y_min'], existing['y_max'])
                if rects_overlap(new_rect, existing_rect):
                    overlap = True
                    break

            if not overlap:
                components.append({
                    'name': f'C{i}',
                    'x_min': new_rect[0],
                    'x_max': new_rect[1],
                    'y_min': new_rect[2],
                    'y_max': new_rect[3],
                    'power': random_power()
                })
                break
            attempts += 1

        if attempts >= max_attempts:
            # Fallback: place component at random position even if overlapping
            components.append({
                'name': f'C{i}',
                'x_min': new_rect[0],
                'x_max': new_rect[1],
                'y_min': new_rect[2],
                'y_max': new_rect[3],
                'power': random_power()
            })

    return components

def run_single_simulation(components, grid_size=GRID_SIZE):
    """Run thermal simulation for given components"""
    T, comps_out, _, _ = pcb_2d_thermal_simulation_sor_optimized(
        grid_size=grid_size,
        ambient_temp=AMBIENT_TEMP,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        omega=OMEGA,
        components_mm=components,
        pcb_dimensions_mm=(PCB_SIZE_MM, PCB_SIZE_MM)
    )
    return T, comps_out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Configuration: grid={GRID_SIZE}, ambient={AMBIENT_TEMP}C, power={POWER_MIN}-{POWER_MAX}W")
    print("-" * 60)

    total_samples = sum(SAMPLES_PER_COUNT.values())
    print(f"Total samples to generate: {total_samples}")

    sample_idx = 0
    start_time = time.time()

    for num_comp in range(1, 6):
        num_samples = SAMPLES_PER_COUNT[num_comp]
        print(f"\n=== Generating {num_samples} samples with {num_comp} component(s) ===")

        for i in range(1, num_samples + 1):
            sample_idx += 1
            sample_start = time.time()

            # Generate random layout
            components = generate_non_overlapping_layout(num_comp)

            # Build filename based on component positions and powers
            pos_parts = []
            power_parts = []
            for c in components:
                cx = (c['x_min'] + c['x_max']) / 2
                cy = (c['y_min'] + c['y_max']) / 2
                pos_parts.append(f"{int(cx)},{int(cy)}")
                power_parts.append(f"{c['power']:.2f}")

            pos_tag = '_'.join(pos_parts)
            power_tag = '_'.join(power_parts)
            filename = f"comp{num_comp}_s{i:02d}_{pos_tag}_p{power_tag}"

            # Run simulation
            T, comps_out = run_single_simulation(components)

            # Get temperature stats
            t_min = T.min()
            t_max = T.max()

            # Save JSON
            temp_data = generate_temperature_json_data(T, (PCB_SIZE_MM, PCB_SIZE_MM), GRID_SIZE)
            json_data = {
                'num_components': num_comp,
                'sample_index': i,
                'filename_base': filename,
                'components': [{
                    'name': c['name'],
                    'x_range_mm': [c['x_min'], c['x_max']],
                    'y_range_mm': [c['y_min'], c['y_max']],
                    'center_mm': [(c['x_min']+c['x_max'])/2, (c['y_min']+c['y_max'])/2],
                    'power_W': c['power']
                } for c in components],
                'temperature_range_C': [t_min, t_max],
                'simulation_params': {
                    'grid_size': GRID_SIZE,
                    'ambient_temp_C': AMBIENT_TEMP,
                    'max_iterations': MAX_ITERATIONS,
                    'tolerance': TOLERANCE,
                    'omega': OMEGA,
                    'pcb_size_mm': PCB_SIZE_MM
                },
                'temperature_data': temp_data
            }

            json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Save PNG
            png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
            fig = plot_temperature(T, comps_out, (PCB_SIZE_MM, PCB_SIZE_MM),
                                 save_path=png_path, show_plot=False)
            plt.close(fig)

            elapsed = time.time() - sample_start
            print(f"  [{sample_idx:02d}/{total_samples}] {filename}: "
                  f"T={t_min:.1f}-{t_max:.1f}C ({elapsed:.1f}s)")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"DONE! Generated {total_samples} samples in {total_elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()