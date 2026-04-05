"""
Verify hotspot position matches component position
"""
import sys
sys.path.insert(0, r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\simulation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Single component config: center at (50, 50)mm, power 5W
comps = [{
    'name': 'C0',
    'x_min': 46.0, 'x_max': 54.0,  # 8x8mm component centered at 50mm
    'y_min': 46.0, 'y_max': 54.0,
    'power': 5.0
}]

from thermal_prediction_error import pcb_2d_thermal_simulation_sor_optimized

# Run simulation
print("=== Running thermal simulation ===")
T, comps_out, _, _ = pcb_2d_thermal_simulation_sor_optimized(
    grid_size=100,
    ambient_temp=25.0,
    max_iterations=100000,
    tolerance=1e-12,
    omega=1.98,
    components_mm=comps,
    pcb_dimensions_mm=(100.0, 100.0)
)

print(f"\n=== Simulation Results ===")
print(f"T shape: {T.shape}")
print(f"T range: {T.min():.2f}C - {T.max():.2f}C")

# Find hotspot position - use argmax directly then convert
flat_idx = np.argmax(T)
hotspot_grid_x = flat_idx // T.shape[1]
hotspot_grid_y = flat_idx % T.shape[1]
hotspot_temp = T[hotspot_grid_x, hotspot_grid_y]

# Calculate hotspot physical position (mm)
mm_per_grid = 100.0 / 100  # PCB 100mm / 100 grid = 1mm per grid
hotspot_x_mm = hotspot_grid_x * mm_per_grid
hotspot_y_mm = hotspot_grid_y * mm_per_grid

print(f"\n=== Hotspot Position ===")
print(f"Hotspot grid: ({hotspot_grid_x}, {hotspot_grid_y})")
print(f"Hotspot physical: ({hotspot_x_mm:.1f}mm, {hotspot_y_mm:.1f}mm)")
print(f"Hotspot temperature: {hotspot_temp:.2f}C")

# Component position
comp = comps[0]
comp_center_x = (comp['x_min'] + comp['x_max']) / 2
comp_center_y = (comp['y_min'] + comp['y_max']) / 2
print(f"\n=== Component Position ===")
print(f"Component name: {comp['name']}")
print(f"Component range: x=[{comp['x_min']}, {comp['x_max']}]mm, y=[{comp['y_min']}, {comp['y_max']}]mm")
print(f"Component center: ({comp_center_x:.1f}mm, {comp_center_y:.1f}mm)")
print(f"Component power: {comp['power']}W")

# Position offset
diff_x = abs(hotspot_x_mm - comp_center_x)
diff_y = abs(hotspot_y_mm - comp_center_y)
print(f"\n=== Position Offset ===")
print(f"X offset: {diff_x:.2f}mm")
print(f"Y offset: {diff_y:.2f}mm")
print(f"Total offset: {np.sqrt(diff_x**2 + diff_y**2):.2f}mm")

# Conclusion
if diff_x <= 2 and diff_y <= 2:
    print("\n[OK] Hotspot position matches component position!")
else:
    print("\n[FAIL] Hotspot position does NOT match component position!")

# Generate output
output_dir = r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test'
os.makedirs(output_dir, exist_ok=True)

# Extract temperature data for points above ambient+0.5C
temp_data = []
for i in range(100):
    for j in range(100):
        x_mm = i * mm_per_grid
        y_mm = j * mm_per_grid
        temp = T[i, j]
        if temp > 25.5:
            temp_data.append({
                'x_mm': round(x_mm, 2),
                'y_mm': round(y_mm, 2),
                'temperature': round(float(temp), 4)
            })

json_data = {
    'description': 'Single component temperature distribution data',
    'component': {
        'name': comp['name'],
        'x_range': [comp['x_min'], comp['x_max']],
        'y_range': [comp['y_min'], comp['y_max']],
        'center': [comp_center_x, comp_center_y],
        'power_W': comp['power']
    },
    'hotspot': {
        'grid_position': [int(hotspot_grid_x), int(hotspot_grid_y)],
        'physical_position_mm': [round(hotspot_x_mm, 2), round(hotspot_y_mm, 2)],
        'temperature_C': round(float(hotspot_temp), 4)
    },
    'temperature_points_count': len(temp_data),
    'temperature_points': temp_data,
    'ambient_temp_C': 25.0,
    'mm_per_grid': mm_per_grid
}

json_path = os.path.join(output_dir, 'single_component_temps.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved to: {json_path}")

# Generate PNG visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Temperature distribution
vmin = max(25.0, T.min() - 2)
vmax = T.max() + 2
im = ax.imshow(T, cmap='hot', origin='lower',
               extent=[0, 100, 0, 100], vmin=vmin, vmax=vmax)

# Component rectangle (cyan border)
rect = plt.Rectangle((comp['x_min'], comp['y_min']),
                     comp['x_max'] - comp['x_min'], comp['y_max'] - comp['y_min'],
                     linewidth=3, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)

# Hotspot marker (green X)
ax.plot(hotspot_grid_x, hotspot_grid_y, 'gX', markersize=15, markeredgewidth=2,
        label=f'Hotspot ({hotspot_x_mm:.1f}mm, {hotspot_y_mm:.1f}mm)')

# Component center (white circle)
ax.plot(comp_center_x, comp_center_y, 'wo', markersize=10, markerfacecolor='none',
        markeredgewidth=2, label=f'Component center ({comp_center_x:.1f}mm, {comp_center_y:.1f}mm)')

ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)
ax.set_title(f'Single Component Thermal Map\nHotspot: ({hotspot_x_mm:.1f}mm, {hotspot_y_mm:.1f}mm) {hotspot_temp:.1f}C\nComponent: ({comp_center_x:.1f}mm, {comp_center_y:.1f}mm)', fontsize=11)
ax.legend(loc='upper right')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Temperature (C)', fontsize=11)

plt.tight_layout()
png_path = os.path.join(output_dir, 'single_component_verification.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"PNG saved to: {png_path}")

print("\n=== Verification Complete ===")