"""
plot_gen_json.py
将 generation_dataset 里的 .json 文件打印成 PNG 图像。
"""
import os
import json
import sys

TP_DIR = os.path.dirname(os.path.dirname(__file__))
GEN_DIR = os.path.join(TP_DIR, "data", "generation_dataset")


def main():
    json_files = sorted([f for f in os.listdir(GEN_DIR) if f.endswith(".json")])
    print(f"Found {len(json_files)} JSON files")

    for fname in json_files:
        json_path = os.path.join(GEN_DIR, fname)
        with open(json_path, "r", encoding="utf-8") as f:
            temp_data = json.load(f)

        temps = [d["temperature"] for d in temp_data]
        print(f"  {fname}: temp=[{min(temps):.2f}, {max(temps):.2f}] °C")

    print("\nAll files found. Use plot_temperature from thermal_prediction.py to visualize.")

    # 导入并绘图
    sys.path.insert(0, os.path.join(TP_DIR, "simulation"))
    from thermal_prediction import plot_temperature, pcb_2d_thermal_simulation_sor_optimized, generate_temperature_json_data
    import numpy as np
    import matplotlib.pyplot as plt

    for fname in json_files:
        json_path = os.path.join(GEN_DIR, fname)
        with open(json_path, "r", encoding="utf-8") as f:
            temp_data = json.load(f)

        temps = np.array([d["temperature"] for d in temp_data], dtype=np.float32)
        T = temps.reshape(100, 100)

        png_path = json_path.replace(".json", ".png")

        # 从文件名解析组件数量和位置
        name = fname.replace(".json", "")
        parts = name.split("_")
        nums = [p for p in parts if p.lstrip("-").isdigit()]
        # 从文件名提取组件数量: count7_idx_... -> n_comp=7
        count_match = name.split("_")[0]  # "count7"
        n_comp = int(count_match.replace("count", ""))
        comps = []
        for i in range(n_comp):
            cx = int(nums[i*2])
            cy = int(nums[i*2+1])
            half = 4.0
            comps.append({
                "name": f"C{i+1}",
                "x_min": cx - half, "x_max": cx + half,
                "y_min": cy - half, "y_max": cy + half,
                "x_min_mm": cx - half, "x_max_mm": cx + half,
                "y_min_mm": cy - half, "y_max_mm": cy + half,
            })

        fig = plot_temperature(T, comps, (100.0, 100.0),
                               save_path=png_path, show_plot=False)
        plt.close(fig)
        print(f"  Saved: {png_path}")

    print(f"\nDone! {len(json_files)} PNGs saved.")


if __name__ == "__main__":
    main()
