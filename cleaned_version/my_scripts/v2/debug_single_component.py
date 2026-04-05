import sys
sys.path.insert(0, r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\simulation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PCB_SIZE_MM = 100.0
AMBIENT_TEMP = 25.0

comps = [{
    'name': 'C0',
    'x_min': 46.0, 'x_max': 54.0,
    'y_min': 46.0, 'y_max': 54.0,
    'power': 5.0
}]

from thermal_prediction_error import pcb_2d_thermal_simulation_sor_optimized
T, comps_out, _, _ = pcb_2d_thermal_simulation_sor_optimized(
    grid_size=100, ambient_temp=25.0,
    max_iterations=100000, tolerance=1e-12, omega=1.98,
    components_mm=comps,
    pcb_dimensions_mm=(100.0, 100.0)
)

print('T type:', type(T), 'T shape:', T.shape, 'T dtype:', T.dtype)
print('T.max():', T.max())

max_idx = np.unravel_index(np.argmax(T), T)
print(f'Hotspot grid=({max_idx[0]},{max_idx[1]}) T={T[max_idx]:.4f}')

out_dir = r'C:\Users\jkong\Documents\power brain_new\yiwen version\training_data_30W_test'
os.makedirs(out_dir, exist_ok=True)

# Save temperature data to text for verification
np.savetxt(os.path.join(out_dir, 'T_single_component.txt'), T, fmt='%.4f')
print('Saved T matrix to txt')

# Plot center cross-section
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(T[50, :], range(100))
ax.set_xlabel('Y (grid index)')
ax.set_ylabel('Temperature (C)')
ax.set_title(f'T at X=50 (row 50), max={T[50,:].max():.2f}')
ax.axhline(y=T[50,:].max(), color='r', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(range(100), T[:, 50])
ax.set_xlabel('X (grid index)')
ax.set_ylabel('Temperature (C)')
ax.set_title(f'T at Y=50 (col 50), max={T[:,50].max():.2f}')
ax.axhline(y=T[:,50].max(), color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'single_component_cross_section.png'), dpi=150)
plt.close()
print('Saved cross-section plot')

# Generate PNG
fig, ax = plt.subplots(figsize=(5, 4))
vmin = max(AMBIENT_TEMP, T.min() - 3)
vmax = T.max() + 3
print(f'vmin={vmin}, vmax={vmax}')

im = ax.imshow(T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax,
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM])
c = comps[0]
rect = plt.Rectangle((c['x_min'], c['y_min']),
                     c['x_max'] - c['x_min'], c['y_max'] - c['y_min'],
                     linewidth=2, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)
ax.text(50, 50, '5W', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Single Component: center=(50,50)mm')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'single_component_test.png'), dpi=150)
plt.close()
print('Saved PNG')
print('DONE')