"""
generate_random_10comp.py
========================
用 thermal_prediction.py 生成 10 组件随机位置的温度数据。
前9个组件功率2.5W，第10个组件功率10W，总功率32.5W。
每个样本的 10 个组件位置随机生成（不重叠）。
"""
import os
import sys
import json
import numpy as np

TP_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(TP_DIR, "data", "generation_dataset")

BOARD_MM = 100.0
GRID_SIZE = 100
AMBIENT_TEMP = 25.0
COMPONENT_SIZE_MM = 8.0
# 前9个组件2.5W，第10个组件10W
POWERS = [2.5] * 9 + [10.0]
NUM_SAMPLES = 10


def rects_overlap(r1, r2):
    x1_min, x1_max, y1_min, y1_max = r1
    x2_min, x2_max, y2_min, y2_max = r2
    return not (x1_max <= x2_min or x2_max <= x1_min or
                y1_max <= y2_min or y2_max <= y1_min)


def random_component_positions(n=10, board_mm=100.0, comp_size_mm=8.0, powers=None, seed=None):
    if powers is None:
        powers = [2.0] * n
    if seed is not None:
        np.random.seed(seed)
    half = comp_size_mm / 2.0
    margin = 2.0
    comps = []
    for i in range(n):
        attempts = 0
        while attempts < 1000:
            cx = np.random.uniform(margin + half, board_mm - margin - half)
            cy = np.random.uniform(margin + half, board_mm - margin - half)
            x_min, x_max = cx - half, cx + half
            y_min, y_max = cy - half, cy + half
            overlaps = False
            for existing in comps:
                if rects_overlap((x_min, x_max, y_min, y_max),
                                 (existing["x_min"], existing["x_max"],
                                  existing["y_min"], existing["y_max"])):
                    overlaps = True
                    break
            if not overlaps:
                comps.append({
                    "name": f"C{i+1}",
                    "x_min": round(x_min, 2), "x_max": round(x_max, 2),
                    "y_min": round(y_min, 2), "y_max": round(y_max, 2),
                    "power": powers[i],
                })
                break
            attempts += 1
        if attempts >= 1000:
            raise RuntimeError(f"Could not place component {i+1} without overlap")
    return comps


def build_filename(comps, n_comp):
    flat = []
    for comp in comps:
        cx = int((comp["x_min"] + comp["x_max"]) / 2)
        cy = int((comp["y_min"] + comp["y_max"]) / 2)
        flat.append(str(cx))
        flat.append(str(cy))
    return f"count{n_comp}_idx_" + "_".join(flat) + ".json"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Generating {NUM_SAMPLES} random 10-component samples...")
    print(f"Powers: C1-C9 = 2.5W, C10 = 10W, Total = {sum(POWERS):.1f}W")

    for i in range(NUM_SAMPLES):
        n_comp = 10
        print(f"\n--- Sample {i+1}/{NUM_SAMPLES} ---")
        comps = random_component_positions(n=n_comp, board_mm=BOARD_MM,
                                           comp_size_mm=COMPONENT_SIZE_MM,
                                           powers=POWERS, seed=i*42 + 1337)
        total_p = sum(c["power"] for c in comps)
        for c in comps:
            cx = (c["x_min"] + c["x_max"]) / 2
            cy = (c["y_min"] + c["y_max"]) / 2
            print(f"  {c['name']}: center=({cx:.1f}, {cy:.1f}), power={c['power']}W")
        print(f"  Total power: {total_p:.1f}W")

        print(f"  Running thermal simulation...")
        T, comps_out, k_matrix, convergence = pcb_2d_thermal_simulation_sor_optimized(
            grid_size=GRID_SIZE,
            ambient_temp=AMBIENT_TEMP,
            max_iterations=100000,
            tolerance=1e-12,
            omega=1.98,
            components_mm=comps,
            pcb_dimensions_mm=(BOARD_MM, BOARD_MM),
        )

        temp_data = generate_temperature_json_data(T, (BOARD_MM, BOARD_MM), GRID_SIZE)
        temps = [d["temperature"] for d in temp_data]
        print(f"  Temp range: [{min(temps):.2f}, {max(temps):.2f}] °C")

        fname = build_filename(comps, n_comp=10)
        json_path = os.path.join(OUTPUT_DIR, fname)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(temp_data, f, indent=2)
        print(f"  Saved: {fname}")

    print(f"\n\nTotal saved: {NUM_SAMPLES} files in {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(TP_DIR, "simulation"))
    from thermal_prediction import (
        pcb_2d_thermal_simulation_sor_optimized,
        generate_temperature_json_data,
    )
    main()
