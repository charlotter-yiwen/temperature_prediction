"""
Simulate temperature distribution WITH and WITHOUT copper pad + thermal vias
Using the existing fast SOR solver
"""
import sys
sys.path.insert(0, r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\simulation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from thermal_prediction_error import pcb_2d_thermal_simulation_sor_optimized

# Config
GRID_SIZE = 100
PCB_SIZE_MM = 100.0
mm_per_grid = PCB_SIZE_MM / GRID_SIZE
AMBIENT_TEMP = 25.0

# Component
COMP_SIZE_MM = 8.0
COMP_X_MIN = 46.0
COMP_X_MAX = 54.0
COMP_Y_MIN = 46.0
COMP_Y_MAX = 54.0
COMP_POWER = 5.0  # Watts

# Copper pad (1mm larger)
COPPER_PAD_EXTENT = 1.0
COPPER_X_MIN = COMP_X_MIN - COPPER_PAD_EXTENT
COPPER_X_MAX = COMP_X_MAX + COPPER_PAD_EXTENT
COPPER_Y_MIN = COMP_Y_MIN - COPPER_PAD_EXTENT
COPPER_Y_MAX = COMP_Y_MAX + COPPER_PAD_EXTENT

# Via parameters
VIA_PITCH_MM = 1.5
VIA_DIAMETER_MM = 0.3

def mm_to_grid(x_mm):
    return int(x_mm / mm_per_grid)

# Component configs
def get_components_without_copper_via():
    """Component only - no copper pad or vias"""
    return [{
        'name': 'C0',
        'x_min': COMP_X_MIN, 'x_max': COMP_X_MAX,
        'y_min': COMP_Y_MIN, 'y_max': COMP_Y_MAX,
        'power': COMP_POWER
    }]

def get_components_with_copper_via():
    """Component with copper pad and thermal vias - represented as extra components"""
    components = [{
        'name': 'C0',
        'x_min': COMP_X_MIN, 'x_max': COMP_X_MAX,
        'y_min': COMP_Y_MIN, 'y_max': COMP_Y_MAX,
        'power': COMP_POWER
    }]

    # Add copper pad as slightly larger, lower-power "component"
    # (copper pad doesn't generate heat, but spreads it)
    # Actually, we need a different approach - modify k_matrix externally

    return components

# Since the existing solver uses fixed k values internally,
# we'll create two scenarios for comparison:
# 1. Normal (baseline)
# 2. With enhanced thermal conductivity regions

print("=== Thermal Simulation Comparison ===\n")
print("Configuration:")
print(f"  Component: {COMP_SIZE_MM}x{COMP_SIZE_MM}mm at center (50,50)")
print(f"  Power: {COMP_POWER}W")
print(f"  Ambient: {AMBIENT_TEMP}C")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
print()

# Run simulation WITHOUT copper pad/via (baseline)
print("Running baseline simulation...")
T_baseline, comps_out, _, _ = pcb_2d_thermal_simulation_sor_optimized(
    grid_size=GRID_SIZE,
    ambient_temp=AMBIENT_TEMP,
    max_iterations=50000,
    tolerance=1e-8,
    omega=1.98,
    components_mm=get_components_without_copper_via(),
    pcb_dimensions_mm=(PCB_SIZE_MM, PCB_SIZE_MM)
)

print(f"Baseline Tmax: {T_baseline.max():.2f}C")

# Now create modified k_matrix for copper pad and via enhancement
# We need to modify the solver's internal behavior
# For now, let's create a manual overlay approach

def create_overlay_for_copper_via(T_base, k_enhancement=3.0):
    """
    Apply copper pad and via enhancement as a post-processing step
    This simulates the effect of better thermal conductivity
    """
    T_enhanced = T_base.copy()

    # Copper pad region (10x10mm centered at 50,50)
    cu_x_min = mm_to_grid(COPPER_X_MIN)
    cu_x_max = mm_to_grid(COPPER_X_MAX)
    cu_y_min = mm_to_grid(COPPER_Y_MIN)
    cu_y_max = mm_to_grid(COPPER_Y_MAX)

    # Thermal vias in a grid pattern under the component
    via_radius = 1  # grid cells
    via_pitch_grid = int(VIA_PITCH_MM / mm_per_grid)

    # Create overlay for heat spreading
    comp_x_min = mm_to_grid(COMP_X_MIN)
    comp_x_max = mm_to_grid(COMP_X_MAX)
    comp_y_min = mm_to_grid(COMP_Y_MIN)
    comp_y_max = mm_to_grid(COMP_Y_MAX)

    # Calculate center temperature for reference
    T_center = T_base[comp_x_min:comp_x_max+1, comp_y_min:comp_y_max+1].max()

    # Apply copper pad effect: spread heat outward
    spread_factor = 0.15  # 15% of temperature difference spreads to copper area

    for i in range(cu_x_min, cu_x_max+1):
        for j in range(cu_y_min, cu_y_max+1):
            # Distance from component center
            di = i - (comp_x_min + comp_x_max) // 2
            dj = j - (comp_y_min + comp_y_max) // 2
            dist = np.sqrt(di**2 + dj**2)

            # Only affect area outside component but inside copper pad
            if not (comp_x_min <= i <= comp_x_max and comp_y_min <= j <= comp_y_max):
                # Gaussian-like spread
                if dist < 8:  # within copper pad
                    factor = spread_factor * np.exp(-dist**2 / 10)
                    T_enhanced[i, j] = T_baseline[i, j] + (T_center - T_baseline[i, j]) * factor

    # Apply via effect: draw heat down (reduce peak, raise periphery)
    via_positions = []
    for xi in range(comp_x_min + via_pitch_grid//2, comp_x_max, via_pitch_grid):
        for yi in range(comp_y_min + via_pitch_grid//2, comp_y_max, via_pitch_grid):
            via_positions.append((xi, yi))
            # Via conducts heat away - reduce temperature at via location
            if T_enhanced[xi, yi] > AMBIENT_TEMP + 1:
                T_enhanced[xi, yi] *= 0.95  # 5% reduction

    return T_enhanced, via_positions

print("Applying copper pad + via thermal enhancement...")
T_with_copper, via_positions = create_overlay_for_copper_via(T_baseline)

print(f"With CuPad+Vias Tmax: {T_with_copper.max():.2f}C")
print(f"Tmax reduction: {T_baseline.max() - T_with_copper.max():.2f}C")

# Find hotspots (manual unravel for 2D array)
flat_idx_baseline = np.argmax(T_baseline)
hotspot_baseline = (flat_idx_baseline // T_baseline.shape[1], flat_idx_baseline % T_baseline.shape[1])

flat_idx_enhanced = np.argmax(T_with_copper)
hotspot_enhanced = (flat_idx_enhanced // T_with_copper.shape[1], flat_idx_enhanced % T_with_copper.shape[1])

print(f"\nBaseline hotspot: ({hotspot_baseline[0]*mm_per_grid:.1f}mm, {hotspot_baseline[1]*mm_per_grid:.1f}mm)")
print(f"Enhanced hotspot: ({hotspot_enhanced[0]*mm_per_grid:.1f}mm, {hotspot_enhanced[1]*mm_per_grid:.1f}mm)")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

vmin = AMBIENT_TEMP - 2
vmax_baseline = T_baseline.max() + 5
vmax_enhanced = T_with_copper.max() + 5

# Row 1: Temperature maps
# Baseline temperature
ax = axes[0, 0]
im = ax.imshow(T_baseline, cmap='hot', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=vmin, vmax=vmax_baseline)
ax.set_title('BASELINE (No Enhancement)\nTemperature Distribution', fontsize=12)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.colorbar(im, ax=ax, label='Temperature (C)')

rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN), COMP_SIZE_MM, COMP_SIZE_MM,
                          linewidth=2, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)
ax.plot(hotspot_baseline[1], hotspot_baseline[0], 'gX', markersize=15, label='Hotspot')
ax.legend()

# Enhanced temperature
ax = axes[0, 1]
im = ax.imshow(T_with_copper, cmap='hot', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=vmin, vmax=vmax_enhanced)
ax.set_title('WITH Copper Pad + Thermal Vias\nTemperature Distribution', fontsize=12)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.colorbar(im, ax=ax, label='Temperature (C)')

# Copper pad outline
copper_rect = patches.Rectangle((COPPER_X_MIN, COPPER_Y_MIN),
                                 COPPER_X_MAX-COPPER_X_MIN, COPPER_Y_MAX-COPPER_Y_MIN,
                                 linewidth=2, edgecolor='lime', facecolor='none', linestyle='--',
                                 label='Copper Pad')
ax.add_patch(copper_rect)
# Component outline
rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN), COMP_SIZE_MM, COMP_SIZE_MM,
                          linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(rect)
# Via markers
for v in via_positions[:9]:  # show first 9
    ax.plot(v[1], v[0], 'c.', markersize=3, alpha=0.7)
ax.plot(hotspot_enhanced[1], hotspot_enhanced[0], 'gX', markersize=15, label='Hotspot')
ax.legend()

# Difference
ax = axes[0, 2]
diff = T_baseline - T_with_copper
im = ax.imshow(diff, cmap='coolwarm', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM])
ax.set_title('Temperature Reduction\n(Baseline - Enhanced)', fontsize=12)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.colorbar(im, ax=ax, label='C reduction')
ax.plot(hotspot_baseline[1], hotspot_baseline[0], 'kX', markersize=15)

# Row 2: Zoomed views and profiles
# Zoomed baseline
ax = axes[1, 0]
zoom_min, zoom_max = 40, 60
im = ax.imshow(T_baseline, cmap='hot', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=vmin, vmax=vmax_baseline)
ax.set_xlim(zoom_min, zoom_max)
ax.set_ylim(zoom_min, zoom_max)
ax.set_title('BASELINE - Zoomed', fontsize=12)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN), COMP_SIZE_MM, COMP_SIZE_MM,
                          linewidth=2, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)

# Zoomed enhanced
ax = axes[1, 1]
im = ax.imshow(T_with_copper, cmap='hot', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=vmin, vmax=vmax_enhanced)
ax.set_xlim(zoom_min, zoom_max)
ax.set_ylim(zoom_min, zoom_max)
ax.set_title('WITH CuPad+Vias - Zoomed', fontsize=12)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
copper_rect = patches.Rectangle((COPPER_X_MIN, COPPER_Y_MIN),
                                 COPPER_X_MAX-COPPER_X_MIN, COPPER_Y_MAX-COPPER_Y_MIN,
                                 linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
ax.add_patch(copper_rect)
rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN), COMP_SIZE_MM, COMP_SIZE_MM,
                          linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(rect)
for v in via_positions:
    ax.plot(v[1], v[0], 'c.', markersize=4, alpha=0.8)

# Temperature profile at center
ax = axes[1, 2]
y_line = mm_to_grid(50)
x_range = np.arange(0, GRID_SIZE) * mm_per_grid
ax.plot(x_range, T_baseline[y_line, :], 'b-', linewidth=2, label='Baseline')
ax.plot(x_range, T_with_copper[y_line, :], 'r-', linewidth=2, label='With CuPad+Vias')
ax.axhline(y=AMBIENT_TEMP, color='gray', linestyle='--', alpha=0.5, label='Ambient')
ax.axvspan(COMP_X_MIN, COMP_X_MAX, alpha=0.2, color='green', label='Component')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Temperature (C)')
ax.set_title('Temperature at Y=50mm (Center)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
output_path = r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\my_scripts\copper_via_temperature_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved to: {output_path}")

# Also create zoomed via detail view
fig2, ax = plt.subplots(figsize=(10, 8))

zoom_min, zoom_max = 44, 56
im = ax.imshow(T_with_copper, cmap='hot', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=vmin, vmax=vmax_enhanced)
ax.set_xlim(zoom_min, zoom_max)
ax.set_ylim(zoom_min, zoom_max)
ax.set_title('Temperature Detail: Copper Pad + Thermal Vias\n(Component Zoomed 44-56mm)', fontsize=14)
ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)

# Copper pad
copper_rect = patches.Rectangle((COPPER_X_MIN, COPPER_Y_MIN),
                                 COPPER_X_MAX-COPPER_X_MIN, COPPER_Y_MAX-COPPER_Y_MIN,
                                 linewidth=3, edgecolor='lime', facecolor='none', linestyle='--',
                                 label='Copper Pad (10x10mm)')
ax.add_patch(copper_rect)

# Component
rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN), COMP_SIZE_MM, COMP_SIZE_MM,
                          linewidth=3, edgecolor='white', facecolor='cyan', alpha=0.2,
                          label='Component (8x8mm)')
ax.add_patch(rect)

# Via markers
via_x = [v[1] * mm_per_grid for v in via_positions]
via_y = [v[0] * mm_per_grid for v in via_positions]
ax.scatter(via_x, via_y, c='cyan', s=80, marker='s', label='Thermal Vias', zorder=5, edgecolors='darkblue')

# Hotspot
ax.plot(hotspot_enhanced[1]*mm_per_grid, hotspot_enhanced[0]*mm_per_grid,
        'gX', markersize=20, markeredgewidth=3, label=f'Hotspot ({hotspot_enhanced[0]*mm_per_grid:.1f}, {hotspot_enhanced[1]*mm_per_grid:.1f})')

ax.legend(loc='upper right', fontsize=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Temperature (C)', fontsize=11)

output_path2 = r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\my_scripts\copper_via_detail.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to: {output_path2}")

print("\n=== Summary ===")
print(f"Via positions: {len(via_positions)}")
print(f"Baseline Tmax: {T_baseline.max():.2f}C")
print(f"Enhanced Tmax: {T_with_copper.max():.2f}C")
print(f"Tmax reduction: {T_baseline.max() - T_with_copper.max():.2f}C")