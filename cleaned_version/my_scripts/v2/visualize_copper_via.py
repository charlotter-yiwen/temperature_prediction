"""
Visualize copper layer and thermal via modeling concept - CORRECTED
"""
import sys
sys.path.insert(0, r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\simulation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# Config
GRID_SIZE = 100
PCB_SIZE_MM = 100.0
mm_per_grid = PCB_SIZE_MM / GRID_SIZE

# Component position (center at 50,50)
COMP_SIZE_MM = 8.0
COMP_X_MIN = 50 - COMP_SIZE_MM/2  # 46mm
COMP_X_MAX = 50 + COMP_SIZE_MM/2  # 54mm
COMP_Y_MIN = 50 - COMP_SIZE_MM/2  # 46mm
COMP_Y_MAX = 50 + COMP_SIZE_MM/2  # 54mm

# Copper pad (1mm larger on each side)
COPPER_PAD_EXTENT = 1.0
COPPER_X_MIN = COMP_X_MIN - COPPER_PAD_EXTENT
COPPER_X_MAX = COMP_X_MAX + COPPER_PAD_EXTENT
COPPER_Y_MIN = COMP_Y_MIN - COPPER_PAD_EXTENT
COPPER_Y_MAX = COMP_Y_MAX + COPPER_PAD_EXTENT

# Via parameters
VIA_PITCH_MM = 1.5
VIA_DIAMETER_MM = 0.3

# Create thermal conductivity map
k_map = np.full((GRID_SIZE, GRID_SIZE), 0.35)  # FR4 baseline

# Helper function: mm to grid index
def mm_to_grid(x_mm):
    return int(x_mm / mm_per_grid)

# Mark copper pad region
copper_x_min_i = mm_to_grid(COPPER_X_MIN)
copper_x_max_i = mm_to_grid(COPPER_X_MAX)
copper_y_min_i = mm_to_grid(COPPER_Y_MIN)
copper_y_max_i = mm_to_grid(COPPER_Y_MAX)

k_map[copper_x_min_i:copper_x_max_i+1, copper_y_min_i:copper_y_max_i+1] = 400.0

# Mark component region (aluminum)
comp_x_min_i = mm_to_grid(COMP_X_MIN)
comp_x_max_i = mm_to_grid(COPPER_X_MAX)
comp_y_min_i = mm_to_grid(COMP_Y_MIN)
comp_y_max_i = mm_to_grid(COPPER_Y_MAX)

k_map[comp_x_min_i:comp_x_max_i+1, comp_y_min_i:comp_y_max_i+1] = 180.0

# Create via mask
via_positions = []
via_pitch_grid = int(VIA_PITCH_MM / mm_per_grid)
via_radius_grid = max(1, int(VIA_DIAMETER_MM / mm_per_grid / 2))

for xi in range(comp_x_min_i + via_pitch_grid//2, comp_x_max_i, via_pitch_grid):
    for yi in range(comp_y_min_i + via_pitch_grid//2, comp_y_max_i, via_pitch_grid):
        via_positions.append((xi, yi))
        for dx in range(-via_radius_grid, via_radius_grid+1):
            for dy in range(-via_radius_grid, via_radius_grid+1):
                if 0 <= xi+dx < GRID_SIZE and 0 <= yi+dy < GRID_SIZE:
                    k_map[xi+dx, yi+dy] = 100.0

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Full PCB thermal conductivity map
ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(k_map, cmap='YlOrRd', origin='lower',
                 extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=0, vmax=450)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_title('Thermal Conductivity Map (k, W/(m·K))\nFull PCB View', fontsize=12)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Thermal Conductivity (W/(m·K))')

legend_elements = [
    patches.Patch(facecolor='#FF6B35', label='Component (k=180)'),
    patches.Patch(facecolor='#FFD700', label='Copper Pad (k=400)'),
    patches.Patch(facecolor='#FF4500', label='Thermal Via (k=100)'),
    patches.Patch(facecolor='#FFFF80', label='FR4 Substrate (k=0.35)'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Plot 2: Zoomed view of component area
ax2 = fig.add_subplot(2, 2, 2)
zoom_min = 40
zoom_max = 60
im2 = ax2.imshow(k_map, cmap='YlOrRd', origin='lower',
                 extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM], vmin=0, vmax=450)
ax2.set_xlim(zoom_min, zoom_max)
ax2.set_ylim(zoom_min, zoom_max)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_title('Zoomed View: Component + Copper Pad + Vias', fontsize=12)

comp_rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN),
                               COMP_SIZE_MM, COMP_SIZE_MM,
                               linewidth=2, edgecolor='blue', facecolor='none',
                               label='Component (8x8mm)')
ax2.add_patch(comp_rect)

copper_rect = patches.Rectangle((COPPER_X_MIN, COPPER_Y_MIN),
                                  COPPER_X_MAX - COPPER_X_MIN,
                                  COPPER_Y_MAX - COPPER_Y_MIN,
                                  linewidth=2, edgecolor='green', facecolor='none',
                                  linestyle='--', label='Copper Pad')
ax2.add_patch(copper_rect)

via_x = [v[0] * mm_per_grid for v in via_positions]
via_y = [v[1] * mm_per_grid for v in via_positions]
ax2.scatter(via_x, via_y, c='red', s=30, marker='o', label='Thermal Vias', zorder=5)

ax2.legend(loc='upper right', fontsize=8)

# Plot 3: CORRECTED Cross-section (TOP to BOTTOM)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)

# ===== CORRECTED LAYER ORDER (TOP to BOTTOM) =====

# Layer 1: AIR (above component)
ax3.add_patch(patches.Rectangle((0.5, 6.8), 9, 0.5, facecolor='#E6F3FF', edgecolor='black'))
ax3.text(5, 7.05, 'Air / Ambient (h=30 W/(m²·K))', ha='center', va='center', fontsize=8)

# Layer 2: COMPONENT (TOP of PCB)
ax3.add_patch(patches.Rectangle((3.5, 5.5), 3, 1.2, facecolor='#404040', edgecolor='black'))
ax3.text(5, 6.1, 'Component\n(Heat Source)', ha='center', va='center', fontsize=9, color='white')

# Arrow showing heat flow from component top
ax3.annotate('', xy=(5, 6.8), xytext=(5, 6.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(5.3, 6.75, 'Heat\n(Convection)', ha='left', va='center', fontsize=7, color='red')

# Layer 3: THERMAL INTERFACE MATERIAL (TIM) / Solder paste
ax3.add_patch(patches.Rectangle((3.5, 5.2), 3, 0.3, facecolor='#FFD700', edgecolor='black'))
ax3.text(5, 5.35, 'TIM / Solder', ha='center', va='center', fontsize=7)

# Layer 4: TOP COPPER PAD (directly under component)
ax3.add_patch(patches.Rectangle((3.3, 4.7), 3.4, 0.5, facecolor='#B87333', edgecolor='black'))
ax3.text(5, 4.95, 'Top Copper Pad (k=400)', ha='center', va='center', fontsize=8, color='white')

# Layer 5: TOP COPPER TRACES (sides)
ax3.add_patch(patches.Rectangle((0.5, 4.7), 2.5, 0.5, facecolor='#CD7F32', edgecolor='black'))
ax3.add_patch(patches.Rectangle((7, 4.7), 2.5, 0.5, facecolor='#CD7F32', edgecolor='black'))
ax3.text(1.5, 4.95, 'Top Cu', ha='center', va='center', fontsize=7, color='white')
ax3.text(8.5, 4.95, 'Top Cu', ha='center', va='center', fontsize=7, color='white')

# Layer 6: THERMAL VIAS (going through FR4)
via_start_y = 1.2
via_end_y = 4.7
via_positions_x = [4.2, 5, 5.8]
for x in via_positions_x:
    ax3.add_patch(patches.Rectangle((x-0.1, via_start_y), 0.2, via_end_y - via_start_y,
                                     facecolor='#B87333', edgecolor='black', alpha=0.8))

ax3.text(5, 3, 'Thermal\nVias\n(k=100)', ha='center', va='center', fontsize=7)

# Arrow showing heat flow down through vias
ax3.annotate('', xy=(5, via_start_y), xytext=(5, 4.6),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax3.text(5.4, 3.5, 'Heat\n(Down)', ha='left', va='center', fontsize=7, color='orange')

# Layer 7: FR4 SUBSTRATE
ax3.add_patch(patches.Rectangle((0.5, 1.2), 9, 2.5, facecolor='#DEB887', edgecolor='black'))
ax3.text(5, 2.45, 'FR4 Substrate (k=0.35)', ha='center', va='center', fontsize=9)

# Layer 8: BOTTOM COPPER GROUND PLANE
ax3.add_patch(patches.Rectangle((0.5, 0.7), 9, 0.5, facecolor='#B87333', edgecolor='black'))
ax3.text(5, 0.95, 'Bottom Copper Ground Plane (k=400)', ha='center', va='center', fontsize=7, color='white')

# Layer 9: AIR (below PCB)
ax3.add_patch(patches.Rectangle((0.5, 0), 9, 0.7, facecolor='#E6F3FF', edgecolor='black'))
ax3.text(5, 0.35, 'Air / Bottom Convection', ha='center', va='center', fontsize=7)

# Add side labels for layer names
ax3.text(-0.5, 7.3, 'TOP', ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)
ax3.text(-0.5, 0.35, 'BOTTOM', ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

ax3.set_title('Corrected Cross-Section (TOP to BOTTOM)', fontsize=12)
ax3.axis('off')

# Plot 4: Legend and explanation
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

explanation_text = """
Thermal Flow Path:

  [Component Heat Source]
           │
           │  ↑ Convection to air (h=30)
           │
    [TIM / Solder Paste]
           │
    [Top Copper Pad] ← Lateral heat spread (k=400)
           │
    [Thermal Vias]  ← Vertical heat path (k=100)
           │
    [FR4 Substrate] ← Poor conductor (k=0.35)
           │
    [Bottom Copper] ← Heat spread and convection
           │
           ↓  ↑ Convection from bottom

In 2D Top-View Simulation:
- Component region: k=180 (aluminum)
- Copper pad area: k=400 (high lateral spread)
- Via locations: k=100 (vertical channels)
- FR4 background: k=0.35 (baseline)
"""
ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\my_scripts\copper_via_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to: {output_path}")

# Detailed via array visualization
fig2, ax = plt.subplots(figsize=(8, 8))

zoom_min = 44
zoom_max = 56

im = ax.imshow(k_map, cmap='viridis', origin='lower',
               extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM],
               vmin=0, vmax=450)

ax.set_xlim(zoom_min, zoom_max)
ax.set_ylim(zoom_min, zoom_max)

comp_rect = patches.Rectangle((COMP_X_MIN, COMP_Y_MIN),
                               COMP_SIZE_MM, COMP_SIZE_MM,
                               linewidth=3, edgecolor='white', facecolor='cyan',
                               alpha=0.3, label='Component (8x8mm)')
ax.add_patch(comp_rect)

copper_rect = patches.Rectangle((COPPER_X_MIN, COPPER_Y_MIN),
                                  COPPER_X_MAX - COPPER_X_MIN,
                                  COPPER_Y_MAX - COPPER_Y_MIN,
                                  linewidth=2, edgecolor='yellow', facecolor='none',
                                  linestyle='--', label='Copper Pad (10x10mm)')
ax.add_patch(copper_rect)

via_x = [v[0] * mm_per_grid for v in via_positions]
via_y = [v[1] * mm_per_grid for v in via_positions]
ax.scatter(via_x, via_y, c='red', s=100, marker='s', label='Thermal Vias', zorder=5, edgecolors='darkred')

ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)
ax.set_title('Thermal Via Array Under Component\n(Via Pitch = 1.5mm, Diameter = 0.3mm)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Thermal Conductivity k (W/(m·K))', fontsize=10)

output_path2 = r'C:\Users\jkong\Documents\power brain_new\yiwen version\temperature_prediction\my_scripts\via_array_detail.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to: {output_path2}")

print("\n=== Summary ===")
print(f"Via positions count: {len(via_positions)}")
print(f"Via pitch: {VIA_PITCH_MM}mm = {via_pitch_grid} grid cells")
print(f"Component area: {COMP_SIZE_MM}x{COMP_SIZE_MM}mm = {comp_x_max_i-comp_x_min_i}x{comp_y_max_i-comp_y_min_i} grid cells")
print(f"Copper pad area: {COPPER_X_MAX-COPPER_X_MIN:.1f}x{COPPER_Y_MAX-COPPER_Y_MIN:.1f}mm")