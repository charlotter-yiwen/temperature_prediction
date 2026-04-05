import numpy as np
import matplotlib.pyplot as plt
import time
import numba
import json
import os
import argparse
import random
import math


def build_shift_filename(shifts):
    """Build a filename stem like dx1,dy1,dx2,dy2,... from shifts."""
    flat = []
    for dx, dy in shifts:
        flat.append(str(int(dx)))
        flat.append(str(int(dy)))
    return ','.join(flat) if flat else '0'


def build_position_filename(shifted_components):
    """Build a filename stem like cx1,cy1,cx2,cy2,... from component center positions."""
    flat = []
    for comp in shifted_components:
        cx = int(round(0.5 * (comp['x_min'] + comp['x_max'])))
        cy = int(round(0.5 * (comp['y_min'] + comp['y_max'])))
        flat.append(str(cx))
        flat.append(str(cy))
    return ','.join(flat) if flat else '0'


def remove_legacy_output_files(output_folder):
    """Remove old generic files so only shift-based names remain."""
    legacy_files = [
        'pcb_layout_thermal_map.png',
        'pcb_layout_temperatures.json',
        'variant_thermal_map.png',
        'variant_temperatures.json',
    ]
    for filename in legacy_files:
        file_path = os.path.join(output_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

# Using numba to accelerate the SOR solver core computation
@numba.njit(parallel=True)
def sor_iteration_kernel(T, k_matrix, Q, hx, hy, h, ambient_temp, omega, grid_size):
    """
    SOR iteration solver core computation, accelerated with Numba
    """
    max_diff = 0.0
    
    # Using red-black ordering for SOR iteration
    # First update "red" points (i+j is even)
    for i in numba.prange(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:  # Only process "red" points
                # Current point thermal conductivity
                k_curr = k_matrix[i, j]
                
                # Heat source
                q_source = Q[i, j]
                
                # Calculate neighbor influence
                sum_neighbor = 0.0
                count_neighbor = 0.0
                
                # x direction
                if i > 0:  # Left
                    k_interface = 2 * k_curr * k_matrix[i-1, j] / (k_curr + k_matrix[i-1, j] + 1e-10)
                    sum_neighbor += T[i-1, j] * hx * k_interface
                    count_neighbor += hx * k_interface
                else:  # Left boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                if i < grid_size - 1:  # Right
                    k_interface = 2 * k_curr * k_matrix[i+1, j] / (k_curr + k_matrix[i+1, j] + 1e-10)
                    sum_neighbor += T[i+1, j] * hx * k_interface
                    count_neighbor += hx * k_interface
                else:  # Right boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                # y direction
                if j > 0:  # Down
                    k_interface = 2 * k_curr * k_matrix[i, j-1] / (k_curr + k_matrix[i, j-1] + 1e-10)
                    sum_neighbor += T[i, j-1] * hy * k_interface
                    count_neighbor += hy * k_interface
                else:  # Bottom boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                if j < grid_size - 1:  # Up
                    k_interface = 2 * k_curr * k_matrix[i, j+1] / (k_curr + k_matrix[i, j+1] + 1e-10)
                    sum_neighbor += T[i, j+1] * hy * k_interface
                    count_neighbor += hy * k_interface
                else:  # Top boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                # Vertical heat dissipation (top surface to environment)
                sum_neighbor += ambient_temp * h
                count_neighbor += h
                
                # Calculate new temperature using SOR method
                if count_neighbor > 0:
                    T_new_val = (sum_neighbor + q_source) / count_neighbor
                    # Apply relaxation factor
                    T_old = T[i, j]
                    T[i, j] = T_old + omega * (T_new_val - T_old)
                    # Calculate change
                    diff = abs(T[i, j] - T_old)
                    max_diff = max(max_diff, diff)
    
    # Then update "black" points (i+j is odd)
    for i in numba.prange(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 1:  # Only process "black" points
                # Current point thermal conductivity
                k_curr = k_matrix[i, j]
                
                # Heat source
                q_source = Q[i, j]
                
                # Calculate neighbor influence
                sum_neighbor = 0.0
                count_neighbor = 0.0
                
                # x direction - use latest temperature values
                if i > 0:  # Left
                    k_interface = 2 * k_curr * k_matrix[i-1, j] / (k_curr + k_matrix[i-1, j] + 1e-10)
                    sum_neighbor += T[i-1, j] * hx * k_interface
                    count_neighbor += hx * k_interface
                else:  # Left boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                if i < grid_size - 1:  # Right
                    k_interface = 2 * k_curr * k_matrix[i+1, j] / (k_curr + k_matrix[i+1, j] + 1e-10)
                    sum_neighbor += T[i+1, j] * hx * k_interface
                    count_neighbor += hx * k_interface
                else:  # Right boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                # y direction
                if j > 0:  # Down
                    k_interface = 2 * k_curr * k_matrix[i, j-1] / (k_curr + k_matrix[i, j-1] + 1e-10)
                    sum_neighbor += T[i, j-1] * hy * k_interface
                    count_neighbor += hy * k_interface
                else:  # Bottom boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                if j < grid_size - 1:  # Up
                    k_interface = 2 * k_curr * k_matrix[i, j+1] / (k_curr + k_matrix[i, j+1] + 1e-10)
                    sum_neighbor += T[i, j+1] * hy * k_interface
                    count_neighbor += hy * k_interface
                else:  # Top boundary
                    sum_neighbor += ambient_temp * h
                    count_neighbor += h
                
                # Vertical heat dissipation (top surface to environment)
                sum_neighbor += ambient_temp * h
                count_neighbor += h
                
                # Calculate new temperature using SOR method
                if count_neighbor > 0:
                    T_new_val = (sum_neighbor + q_source) / count_neighbor
                    # Apply relaxation factor
                    T_old = T[i, j]
                    T[i, j] = T_old + omega * (T_new_val - T_old)
                    # Calculate change
                    diff = abs(T[i, j] - T_old)
                    max_diff = max(max_diff, diff)
            
    return max_diff

# Optimized version of SOR thermal simulation function
def pcb_2d_thermal_simulation_sor_optimized(grid_size, ambient_temp, 
                                          max_iterations, tolerance,
                                          omega, components_mm, pcb_dimensions_mm):
    """
    Optimized version of SOR iteration method to simulate PCB top layer temperature distribution
    """
    start_time = time.time()
    
    # Initialize temperature field to ambient temperature
    T = np.ones((grid_size, grid_size)) * ambient_temp
    
    # Material properties
    k_aluminum = 180.0  # Aluminum thermal conductivity (W/(m·K))
    k_fr4 = 0.35        # FR-4 thermal conductivity (W/(m·K))
    
    # PCB dimensions (mm)
    length_mm, width_mm = pcb_dimensions_mm
    
    # Convert to meters for calculation
    length = length_mm / 1000.0  # Length (m)
    width = width_mm / 1000.0    # Width (m)
    
    # Grid spacing
    dx = length / grid_size  # x direction grid spacing (m)
    dy = width / grid_size   # y direction grid spacing (m)
    
    # Convection heat transfer coefficient (W/(m²·K))
    h = 30.0  # Heat transfer coefficient
    
    # Convert mm coordinates to grid indices
    components = []
    mm_per_grid_x = length_mm / grid_size
    mm_per_grid_y = width_mm / grid_size
    
    # Preprocess component positions
    for comp in components_mm:
        grid_comp = {
            "name": comp["name"],
            "x_min": int(comp["x_min"] / mm_per_grid_x),
            "x_max": int(comp["x_max"] / mm_per_grid_x),
            "y_min": int(comp["y_min"] / mm_per_grid_y),
            "y_max": int(comp["y_max"] / mm_per_grid_y),
            "power": comp["power"],
            # Store original mm coordinates for reference
            "x_min_mm": comp["x_min"],
            "x_max_mm": comp["x_max"],
            "y_min_mm": comp["y_min"],
            "y_max_mm": comp["y_max"]
        }
        components.append(grid_comp)
    
    # Create heat source matrix and thermal conductivity matrix
    Q = np.zeros((grid_size, grid_size))
    k_matrix = np.ones((grid_size, grid_size)) * k_fr4
    
    # Set component heat sources and thermal conductivity
    for comp in components:
        x_min, x_max = comp["x_min"], comp["x_max"]
        y_min, y_max = comp["y_min"], comp["y_max"]
        area = (x_max - x_min) * (y_max - y_min) * dx * dy  # Component area (m²)
        if area > 0:
            power_density = comp["power"] / area  # Heat power density (W/m²)
            
            # Add heat source in component area and set aluminum thermal conductivity
            Q[x_min:x_max, y_min:y_max] = power_density
            k_matrix[x_min:x_max, y_min:y_max] = k_aluminum
    
    # Precalculate constant coefficients
    hx = 1.0 / (dx * dx)
    hy = 1.0 / (dy * dy)
    
    # Convergence monitoring variables
    prev_max_temp = ambient_temp
    last_improvements = np.ones(5) * 1e10
    convergence_history = []
    
    # Calculate theoretical optimal relaxation factor (if not specified)
    if omega <= 0:
        rho = np.cos(np.pi / grid_size)
        omega = 2.0 / (1.0 + np.sqrt(1.0 - rho * rho))
    
    # Main iteration loop
    iteration = 0
    max_diff = tolerance * 10  # Initialize to value greater than tolerance
    
    while iteration < max_iterations and max_diff > tolerance:
        # Use Numba-accelerated SOR kernel
        max_diff = sor_iteration_kernel(T, k_matrix, Q, hx, hy, h, ambient_temp, omega, grid_size)
        
        # Adaptive convergence check - check every 500 iterations
        if iteration % 500 == 0:
            current_max_temp = np.max(T)
            temp_change = current_max_temp - prev_max_temp
            convergence_history.append((iteration, max_diff, current_max_temp))
            
            # Adaptive termination condition
            last_improvements = np.roll(last_improvements, 1)
            last_improvements[0] = abs(temp_change)
            
            # If average temperature change in last 5 checks is very small, terminate early
            if iteration > 2000 and np.mean(last_improvements) < tolerance * 10:
                break
                
            # Check model stability
            if current_max_temp > ambient_temp + 200:
                print("Warning: Temperature too high, model may be unstable.")
                break
                
            prev_max_temp = current_max_temp
        
        iteration += 1
    
    end_time = time.time()
    print(f"Solution completed in {iteration} iterations. Time used: {end_time - start_time:.2f} seconds")
    
    # Simple energy balance check
    total_power_in = np.sum(Q) * dx * dy
    total_power_out = np.sum(h * (T - ambient_temp)) * dx * dy
    
    return T, components, k_matrix, convergence_history

def plot_temperature(T, components, pcb_dimensions_mm, save_path=None, show_plot=False):
    """
    Plot temperature distribution and optionally save to file
    """
    plt.figure(figsize=(10, 8))
    
    # Plot temperature distribution
    vmin = np.min(T)
    vmax = np.max(T)
    
    im_temp = plt.imshow(T.T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, 
                       extent=[0, pcb_dimensions_mm[0], 0, pcb_dimensions_mm[1]])
    
    # Show component positions
    for comp in components:
        x_min, x_max = comp["x_min_mm"], comp["x_max_mm"]
        y_min, y_max = comp["y_min_mm"], comp["y_max_mm"]
        rect = plt.Rectangle((x_min, y_min), 
                          x_max-x_min, y_max-y_min,
                          linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text((x_min+x_max)/2, (y_min+y_max)/2, comp["name"],
               ha='center', va='center', color='white', fontsize=10)
    
    title = f'PCB Temperature Distribution\nTemperature Range: {np.min(T):.1f}°C - {np.max(T):.1f}°C'
    
    plt.title(title)
    plt.xlabel('X coordinate (mm)')
    plt.ylabel('Y coordinate (mm)')
    
    cbar = plt.colorbar(im_temp)
    cbar.set_label('Temperature (°C)')
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temperature map saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return plt.gcf()

def analyze_temperature(T, components, silent=True):
    """
    Analyze temperature results statistics
    """
    if not silent:
        print(f"\n=== Temperature Analysis Results ===")
        print(f"Overall temperature range: {np.min(T):.2f}°C to {np.max(T):.2f}°C")
        print(f"Average temperature: {np.mean(T):.2f}°C")
    
    # Create a dictionary to store component temperature data
    component_temps = {}
    
    # Analyze component temperatures
    for comp in components:
        x_min, x_max = comp["x_min"], comp["x_max"]
        y_min, y_max = comp["y_min"], comp["y_max"]
        
        # Component area temperature
        comp_temp_data = T[x_min:x_max, y_min:y_max]
        
        # Store temperature data
        component_temps[comp['name']] = {
            'max': np.max(comp_temp_data),
            'avg': np.mean(comp_temp_data),
            'power': comp['power']
        }
        
        if not silent:
            print(f"{comp['name']} (power: {comp['power']}W): "
                  f"Max temperature: {np.max(comp_temp_data):.2f}°C, "
                  f"Average temperature: {np.mean(comp_temp_data):.2f}°C")
    
    return component_temps

def generate_temperature_json_data(T, pcb_dimensions_mm, grid_size):
    """
    Generate temperature data in JSON format
    """
    data = []
    # Convert mm to cm for output
    length_cm = pcb_dimensions_mm[0] / 10.0
    width_cm = pcb_dimensions_mm[1] / 10.0
    
    # Calculate grid spacing in cm
    dx_cm = length_cm / grid_size
    dy_cm = width_cm / grid_size
    
    # Generate temperature data points
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate position in cm (center of grid cell)
            x_cm = (i + 0.5) * dx_cm
            y_cm = (j + 0.5) * dy_cm
            
            # Get temperature at this position
            temperature = T[i, j]
            
            # Add data point to list
            data.append({
                "x": round(x_cm, 2),
                "y": round(y_cm, 2),
                "temperature": round(temperature, 4)
            })
    
    return data

def run_pcb_thermal_analysis(components_mm, 
                           output_folder="thermal_analysis_output",
                           filename_prefix="0",
                           grid_size=200, 
                           ambient_temp=25.0,
                           max_iterations=100000, 
                           tolerance=1e-12,
                           omega=1.98,
                           pcb_dimensions_mm=(100.0, 100.0),
                           show_plot=False,
                           save_outputs=True):
    """
    Run PCB thermal analysis for a single component layout
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Run PCB thermal simulation
    T, components, k_matrix, convergence_history = pcb_2d_thermal_simulation_sor_optimized(
        grid_size=grid_size,
        ambient_temp=ambient_temp,
        max_iterations=max_iterations,
        tolerance=tolerance,
        omega=omega,
        components_mm=components_mm,
        pcb_dimensions_mm=pcb_dimensions_mm
    )
    
    # Analyze temperature results (silent mode)
    component_temps = analyze_temperature(T, components, silent=True)
    
    # Plot temperature distribution and save to file
    image_path = None
    if save_outputs:
        image_path = os.path.join(output_folder, f"{filename_prefix}.png")
        fig = plot_temperature(T, components, pcb_dimensions_mm, save_path=image_path, show_plot=show_plot)
        plt.close(fig)
    elif show_plot:
        fig = plot_temperature(T, components, pcb_dimensions_mm, save_path=None, show_plot=show_plot)
        plt.close(fig)
    
    # Generate and save temperature data as JSON
    temp_data = generate_temperature_json_data(T, pcb_dimensions_mm, grid_size)
    if save_outputs:
        json_path = os.path.join(output_folder, f"{filename_prefix}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2)
        print(f"Temperature data saved to: {json_path}")
    
    # Return results
    return T, components, component_temps, temp_data


def rects_overlap(a, b):
    """Return True if rect a overlaps rect b.
    Rect is (x_min, x_max, y_min, y_max)
    """
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    overlap_x = not (ax1 <= bx0 or bx1 <= ax0)
    overlap_y = not (ay1 <= by0 or by1 <= ay0)
    return overlap_x and overlap_y


def create_shifted_components(components_mm, shifts, pcb_dimensions_mm):
    """Return a new components list shifted by shifts [(dx,dy), ...].
    Assumes shifts length == len(components_mm).
    """
    length_mm, width_mm = pcb_dimensions_mm
    new_comps = []
    for comp, (dx, dy) in zip(components_mm, shifts):
        # Apply shift directly relative to the original component coordinates.
        new_comp = {
            'name': comp['name'],
            'x_min': comp['x_min'] + dx,
            'x_max': comp['x_max'] + dx,
            'y_min': comp['y_min'] + dy,
            'y_max': comp['y_max'] + dy,
            'power': comp['power']
        }
        new_comps.append(new_comp)
    return new_comps


def generate_non_overlapping_shifts(components_mm,
                                    pcb_dimensions_mm,
                                    num_variants=15,
                                    max_shift=50,
                                    max_attempts=200,
                                    include_zero_variant=True,
                                    min_shift_distance=0.0):
    """Generate a list of shift-lists. Each shift-list is [(dx,dy), ...] for each component.
    The first variant is always the original layout with zero shifts.
    Remaining variants use random integer shifts in [-max_shift, max_shift] (mm)
    while enforcing board bounds and no overlap.
    """
    variants = []
    n = len(components_mm)
    length_mm, width_mm = pcb_dimensions_mm

    # original rects
    orig_rects = []
    for c in components_mm:
        orig_rects.append((c['x_min'], c['x_max'], c['y_min'], c['y_max']))

    seen = set()
    if include_zero_variant:
        zero_variant = tuple((0, 0) for _ in range(n))
        variants.append(list(zero_variant))
        seen.add(zero_variant)

    tries = 0
    while len(variants) < num_variants and tries < max_attempts * num_variants:
        tries += 1
        shifts = []
        for i in range(n):
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            if math.sqrt(dx * dx + dy * dy) < min_shift_distance:
                # force larger movement when requested
                dx = random.choice([-1, 1]) * random.randint(int(min_shift_distance), max_shift)
                dy = random.choice([-1, 1]) * random.randint(0, max_shift)
            shifts.append((dx, dy))

        shift_key = tuple(shifts)
        if shift_key in seen:
            continue

        # build shifted rects and check bounds and overlaps
        shifted_rects = []
        ok = True
        for (rect, (dx, dy)) in zip(orig_rects, shifts):
            x0, x1, y0, y1 = rect
            sx0, sx1 = x0 + dx, x1 + dx
            sy0, sy1 = y0 + dy, y1 + dy
            # clamp within board
            if sx0 < 0 or sy0 < 0 or sx1 > length_mm or sy1 > width_mm:
                ok = False; break
            shifted_rects.append((sx0, sx1, sy0, sy1))

        if not ok:
            continue

        # check pairwise overlap
        for i in range(n):
            for j in range(i+1, n):
                if rects_overlap(shifted_rects[i], shifted_rects[j]):
                    ok = False; break
            if not ok:
                break

        if ok:
            variants.append(shifts)
            seen.add(shift_key)

    return variants


def generate_and_save_variants(components_mm, pcb_dimensions_mm=(100.0,100.0), num_variants=15, max_shift=50, grid_size=200, output_folder='thermal_analysis_output'):
    """Generate layouts, run simulation for each, and save files named by shifts.
    The first saved layout is always the base layout named like 0,0,0,0.*
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    variants = generate_non_overlapping_shifts(components_mm, pcb_dimensions_mm, num_variants=num_variants, max_shift=max_shift)
    saved = []
    for shifts in variants:
        # create shifted components
        new_comps = create_shifted_components(components_mm, shifts, pcb_dimensions_mm)

        # Run analysis without intermediate generic filenames.
        T, comps, comp_temps, temp_data = run_pcb_thermal_analysis(
            components_mm=new_comps,
            output_folder=output_folder,
            filename_prefix='variant',
            grid_size=grid_size,
            ambient_temp=25.0,
            max_iterations=100000,
            tolerance=1e-12,
            omega=1.98,
            pcb_dimensions_mm=pcb_dimensions_mm,
            show_plot=False,
            save_outputs=False
        )

        name = build_shift_filename(shifts)
        json_path = os.path.join(output_folder, f"{name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2)

        # Save a PNG heatmap with the same base name as the JSON
        png_path = os.path.join(output_folder, f"{name}.png")
        fig = plot_temperature(T, comps, pcb_dimensions_mm, save_path=png_path, show_plot=False)
        plt.close(fig)

        saved.append({
            'name': name,
            'json_path': json_path,
            'png_path': png_path,
            'shifts': [list(pair) for pair in shifts]
        })

    return saved


def generate_component_sweep_dataset(components_mm,
                                     pcb_dimensions_mm=(100.0, 100.0),
                                     positions_per_component=40,
                                     max_shift=50,
                                     grid_size=200,
                                     output_folder='thermal_analysis_output',
                                     ambient_temp=25.0,
                                     max_iterations=30000,
                                     tolerance=1e-9,
                                     omega=1.98,
                                     max_attempts_per_position=300):
    """
    Generate dataset by moving ONE component at a time.

    For each component i:
      - sample `positions_per_component` valid shifts (dx,dy)
      - keep other components fixed
      - run thermal simulation and save PNG/JSON

    Also save:
      - params_sweep.npy : [x1,y1,p1, x2,y2,p2, ..., xM,yM,pM] per sample
        where M = total number of components, (x,y) are center coords in mm,
        p is power in W. All components are always present (moved one stays
        shifted, others keep original positions).
      - temps_sweep.npy  : flattened temperature field
      - sweep_manifest.json
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    n_comp = len(components_mm)
    if n_comp < 1:
        raise ValueError("components_mm must contain at least one component")

    length_mm, width_mm = pcb_dimensions_mm

    # original rects for bound/overlap checks
    base_rects = []
    for comp in components_mm:
        base_rects.append((comp['x_min'], comp['x_max'], comp['y_min'], comp['y_max']))

    params_rows = []
    temps_rows = []
    manifest = []

    def build_xyp_param_vector(shifted_components, n_total):
        """Build [x1,y1,p1, x2,y2,p2, ...] vector from shifted components."""
        vec = np.full((n_total * 3,), np.nan, dtype=np.float32)
        for idx, comp in enumerate(shifted_components):
            cx = 0.5 * (comp['x_min'] + comp['x_max'])
            cy = 0.5 * (comp['y_min'] + comp['y_max'])
            vec[3 * idx] = float(cx)
            vec[3 * idx + 1] = float(cy)
            vec[3 * idx + 2] = float(comp['power'])
        return vec

    for comp_idx in range(n_comp):
        chosen_shifts = set()
        generated = 0
        attempts = 0

        while generated < positions_per_component and attempts < positions_per_component * max_attempts_per_position:
            attempts += 1
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            if dx == 0 and dy == 0:
                continue

            key = (dx, dy)
            if key in chosen_shifts:
                continue

            # Move only selected component, keep others fixed
            moved_rects = list(base_rects)
            x0, x1, y0, y1 = base_rects[comp_idx]
            sx0, sx1 = x0 + dx, x1 + dx
            sy0, sy1 = y0 + dy, y1 + dy

            # board bounds
            if sx0 < 0 or sy0 < 0 or sx1 > length_mm or sy1 > width_mm:
                continue

            moved_rects[comp_idx] = (sx0, sx1, sy0, sy1)

            # overlap check
            valid = True
            for i in range(n_comp):
                for j in range(i + 1, n_comp):
                    if rects_overlap(moved_rects[i], moved_rects[j]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            # build shifted components
            shifts = [(0, 0) for _ in range(n_comp)]
            shifts[comp_idx] = (dx, dy)
            new_comps = create_shifted_components(components_mm, shifts, pcb_dimensions_mm)

            T, comps, comp_temps, temp_data = run_pcb_thermal_analysis(
                components_mm=new_comps,
                output_folder=output_folder,
                filename_prefix='variant',
                grid_size=grid_size,
                ambient_temp=ambient_temp,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                pcb_dimensions_mm=pcb_dimensions_mm,
                show_plot=False,
                save_outputs=False
            )

            name = f"comp{comp_idx+1}_dx{int(dx)}_dy{int(dy)}"
            json_path = os.path.join(output_folder, f"{name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(temp_data, f, indent=2)

            png_path = os.path.join(output_folder, f"{name}.png")
            fig = plot_temperature(T, comps, pcb_dimensions_mm, save_path=png_path, show_plot=False)
            plt.close(fig)

            move_distance = math.sqrt(dx * dx + dy * dy)
            xyp_param = build_xyp_param_vector(new_comps, n_comp)
            params_rows.append(xyp_param)
            temps_rows.append(T.flatten(order='C'))
            manifest.append({
                'name': name,
                'component_index': int(comp_idx + 1),
                'component_name': components_mm[comp_idx]['name'],
                'dx_mm': int(dx),
                'dy_mm': int(dy),
                'move_distance_mm': float(move_distance),
                'params_schema': [feat
                                  for i in range(n_comp)
                                  for feat in (f'x{i+1}', f'y{i+1}', f'p{i+1}')],
                'params': xyp_param.tolist(),
                'json_path': json_path,
                'png_path': png_path
            })

            chosen_shifts.add(key)
            generated += 1

        if generated < positions_per_component:
            raise RuntimeError(
                f"Component {comp_idx+1} only generated {generated}/{positions_per_component} valid positions. "
                f"Try reducing --max-shift or changing layout."
            )

    params_matrix = np.array(params_rows, dtype=np.float32)
    temps_matrix = np.array(temps_rows, dtype=np.float32)

    params_path = os.path.join(output_folder, 'params_sweep.npy')
    temps_path = os.path.join(output_folder, 'temps_sweep.npy')
    manifest_path = os.path.join(output_folder, 'sweep_manifest.json')

    np.save(params_path, params_matrix)
    np.save(temps_path, temps_matrix)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    schema_path = os.path.join(output_folder, 'params_sweep_schema.json')
    schema = {
        'description': 'Per-sample parameter vector: [x1,y1,p1, x2,y2,p2, ...] — center XY (mm) and power (W) for each component.',
        'num_components': int(n_comp),
        'feature_order': [feat
                          for i in range(n_comp)
                          for feat in (f'x{i+1}', f'y{i+1}', f'p{i+1}')],
        'shape': [int(params_matrix.shape[0]), int(params_matrix.shape[1])]
    }
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Saved sweep params to: {params_path}")
    print(f"Saved sweep temps to: {temps_path}")
    print(f"Saved sweep manifest to: {manifest_path}")
    print(f"Saved sweep schema to: {schema_path}")
    print(f"Sweep samples: {params_matrix.shape[0]} (components={n_comp}, per_component={positions_per_component})")

    return {
        'params_path': params_path,
        'temps_path': temps_path,
        'manifest_path': manifest_path,
        'num_samples': int(params_matrix.shape[0]),
        'num_components': int(n_comp),
        'positions_per_component': int(positions_per_component),
    }


def generate_count_sweep_dataset(components_mm,
                                 pcb_dimensions_mm=(100.0, 100.0),
                                 positions_per_count=40,
                                 max_components=5,
                                 min_components=1,
                                 max_shift=25,
                                 grid_size=100,
                                 output_folder='thermal_analysis_output_count_sweep',
                                 ambient_temp=25.0,
                                 max_iterations=25000,
                                 tolerance=1e-8,
                                 omega=1.98,
                                 max_attempts=600,
                                 min_shift_distance=12.0):
    """
    Generate thermal maps by component COUNT:
      min_components component -> positions_per_count maps
      ...
      max_components components -> positions_per_count maps

        Saved params format per sample:
            [x1, y1, p1, x2, y2, p2, ..., xM, yM, pM]
        where M=max_components and unused entries are NaN.
        (x, y) are component center coordinates in mm after shifting,
        p is the component power in W.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    max_components = min(max_components, len(components_mm))
    min_components = max(min_components, 1)
    if max_components < min_components:
        raise ValueError(f"max_components ({max_components}) < min_components ({min_components})")

    params_rows = []
    temps_rows = []
    manifest = []

    def build_xyp_param_vector(shifted_components, max_comp_count):
        vec = np.full((max_comp_count * 3,), np.nan, dtype=np.float32)
        for idx, comp in enumerate(shifted_components):
            cx = 0.5 * (comp['x_min'] + comp['x_max'])
            cy = 0.5 * (comp['y_min'] + comp['y_max'])
            vec[3 * idx] = float(cx)
            vec[3 * idx + 1] = float(cy)
            vec[3 * idx + 2] = float(comp['power'])
        return vec

    for k in range(min_components, max_components + 1):
        comps_k = components_mm[:k]
        variants = generate_non_overlapping_shifts(
            comps_k,
            pcb_dimensions_mm,
            num_variants=positions_per_count,
            max_shift=max_shift,
            max_attempts=max_attempts,
            include_zero_variant=False,
            min_shift_distance=min_shift_distance
        )

        if len(variants) < positions_per_count:
            raise RuntimeError(
                f"Only generated {len(variants)}/{positions_per_count} valid layouts for component_count={k}."
            )

        print(f"component_count={k}: generating {positions_per_count} maps")

        for idx, shifts in enumerate(variants[:positions_per_count], start=1):
            shifted = create_shifted_components(comps_k, shifts, pcb_dimensions_mm)

            T, comps, _, temp_data = run_pcb_thermal_analysis(
                components_mm=shifted,
                output_folder=output_folder,
                filename_prefix='variant',
                grid_size=grid_size,
                ambient_temp=ambient_temp,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                pcb_dimensions_mm=pcb_dimensions_mm,
                show_plot=False,
                save_outputs=False
            )

            move_distances = [math.sqrt(dx * dx + dy * dy) for dx, dy in shifts]
            pos_tag = build_position_filename(shifted)
            name = f"count{k}_idx{idx:02d}_{pos_tag}"

            json_path = os.path.join(output_folder, f"{name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(temp_data, f, indent=2)

            png_path = os.path.join(output_folder, f"{name}.png")
            fig = plot_temperature(T, comps, pcb_dimensions_mm, save_path=png_path, show_plot=False)
            plt.close(fig)

            xyp_param = build_xyp_param_vector(shifted, max_components)
            params_rows.append(xyp_param)
            temps_rows.append(T.flatten(order='C'))

            xy_positions = []
            for comp in shifted:
                xy_positions.append([
                    float(0.5 * (comp['x_min'] + comp['x_max'])),
                    float(0.5 * (comp['y_min'] + comp['y_max'])),
                    float(comp['power']),
                ])

            manifest.append({
                'name': name,
                'component_count': int(k),
                'sample_index_in_count': int(idx),
                'move_distances_mm': [float(v) for v in move_distances],
                'shifts': [list(pair) for pair in shifts],
                'xyp_positions': xy_positions,
                'params_schema': [feat
                                  for i in range(max_components)
                                  for feat in (f'x{i+1}', f'y{i+1}', f'p{i+1}')],
                'params': xyp_param.tolist(),
                'json_path': json_path,
                'png_path': png_path
            })

    params_matrix = np.array(params_rows, dtype=np.float32)
    temps_matrix = np.array(temps_rows, dtype=np.float32)

    params_path = os.path.join(output_folder, 'params_count_sweep.npy')
    temps_path = os.path.join(output_folder, 'temps_count_sweep.npy')
    manifest_path = os.path.join(output_folder, 'count_sweep_manifest.json')

    np.save(params_path, params_matrix)
    np.save(temps_path, temps_matrix)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    schema_path = os.path.join(output_folder, 'params_count_sweep_schema.json')
    schema = {
        'description': 'Per-sample parameter vector: [x1,y1,p1, x2,y2,p2, ...] — center XY (mm) and power (W), padded with NaN for missing components.',
        'max_components': int(max_components),
        'feature_order': [feat
                          for i in range(max_components)
                          for feat in (f'x{i+1}', f'y{i+1}', f'p{i+1}')],
        'shape': [int(params_matrix.shape[0]), int(params_matrix.shape[1])]
    }
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Saved count-sweep params to: {params_path}")
    print(f"Saved count-sweep temps to: {temps_path}")
    print(f"Saved count-sweep manifest to: {manifest_path}")
    print(f"Saved count-sweep schema to: {schema_path}")
    print(f"Count sweep samples: {params_matrix.shape[0]} ({max_components} groups x {positions_per_count})")

    return {
        'params_path': params_path,
        'temps_path': temps_path,
        'manifest_path': manifest_path,
        'num_samples': int(params_matrix.shape[0]),
        'max_components': int(max_components),
        'positions_per_count': int(positions_per_count),
    }


def load_components_from_json(path):
    """Load component list from a JSON file.
    The file should contain a list of components with keys:
    name, x_min, x_max, y_min, y_max, power
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# =============================================================================
# 9.  Random-power count-sweep dataset generator
#    For each sample: random positions + random power split, total 2-30W
# =============================================================================

PCB_SIZE_MM = 100.0
GRID_SIZE = 100
AMBIENT_TEMP = 25.0
COMPONENT_SIZE_MM = 8.0
MIN_POWER_PER_COMP = 1.5
MAX_POWER_PER_COMP = 10.0


def random_position_mm():
    """Return (x_center, y_center) within PCB margins."""
    margin = COMPONENT_SIZE_MM / 2 + 1
    x = random.uniform(margin, PCB_SIZE_MM - margin)
    y = random.uniform(margin, PCB_SIZE_MM - margin)
    return x, y


def generate_random_components(n, total_power):
    """Generate n non-overlapping components with random positions and power split.
    Returns list of dicts: {name, x_min, x_max, y_min, y_max, power}
    """
    for _ in range(2000):
        ratios = np.random.rand(n)
        ratios /= ratios.sum()
        powers = ratios * total_power
        powers = np.clip(powers, MIN_POWER_PER_COMP, MAX_POWER_PER_COMP)
        powers *= total_power / powers.sum()

        positions = []
        comps = []
        ok = True
        for i in range(n):
            cx, cy = random_position_mm()
            x_min = cx - COMPONENT_SIZE_MM / 2
            x_max = cx + COMPONENT_SIZE_MM / 2
            y_min = cy - COMPONENT_SIZE_MM / 2
            y_max = cy + COMPONENT_SIZE_MM / 2
            rect = (x_min, x_max, y_min, y_max)
            for (px0, px1, py0, py1) in positions:
                if rects_overlap(rect, (px0, px1, py0, py1)):
                    ok = False; break
            if not ok: break
            positions.append(rect)
            comps.append({
                'name': f'C{i}',
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'power': float(powers[i])
            })
        if ok:
            return comps
    raise RuntimeError(f'Failed to generate {n} non-overlapping components')


def build_position_filename(comps):
    """cx1_cy1_cx2_cy2_... from component list (absolute positions)."""
    flat = []
    for c in comps:
        cx = int(round((c['x_min'] + c['x_max']) / 2))
        cy = int(round((c['y_min'] + c['y_max']) / 2))
        flat.append(f"{cx}_{cy}")
    return '_'.join(flat)


def build_xyp_param_vector(comps, max_comp=5):
    """[x1,y1,p1, x2,y2,p2, ...] for up to max_comp components, NaN-padded."""
    vec = np.full((max_comp * 3,), np.nan, dtype=np.float32)
    for idx, c in enumerate(comps):
        cx = (c['x_min'] + c['x_max']) / 2
        cy = (c['y_min'] + c['y_max']) / 2
        vec[3*idx] = float(cx)
        vec[3*idx+1] = float(cy)
        vec[3*idx+2] = float(c['power'])
    return vec


def generate_random_power_count_sweep(
        output_dir,
        n_counts=(1, 2, 3, 4, 5),
        n_per_count=120,
        power_min=2.0,
        power_max=30.0,
        grid_size=100,
        seed=42,
        save_every=10):
    """Generate count-sweep dataset with fully random positions AND random power split.

    Params format (matching training_data_30W needs):
        shape: (N, 5, 3)  [x_mm, y_mm, power_W] per component, NaN if absent

    Temps format:
        shape: (N, grid_size^2)  flattened temperature field

    Output:
        output_dir/params_count_sweep.npy
        output_dir/temps_count_sweep.npy
        output_dir/samples/  PNG every N samples
        output_dir/samples_summary.csv
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    total = len(n_counts) * n_per_count
    params = np.full((total, 5, 3), np.nan, dtype=np.float32)
    temps  = np.zeros((total, grid_size * grid_size), dtype=np.float32)

    csv_path = os.path.join(output_dir, 'samples_summary.csv')
    csv_f = open(csv_path, 'w')
    csv_f.write('index,n_comp,total_power_W,T_max_C,'
                'x1,y1,p1,x2,y2,p2,x3,y3,p3,x4,y4,p4,x5,y5,p5\n')

    idx = 0
    for n_comp in n_counts:
        print(f'\n=== {n_comp} components x {n_per_count} samples ===')
        for i in range(n_per_count):
            # Random total power (log-uniform for even 2-30W coverage)
            log_p = np.random.uniform(np.log(power_min), np.log(power_max))
            total_power = float(np.exp(log_p))

            # Generate layout
            comps = generate_random_components(n_comp, total_power)

            # Simulate
            T, _, _, _ = pcb_2d_thermal_simulation_sor_optimized(
                grid_size=grid_size,
                ambient_temp=AMBIENT_TEMP,
                max_iterations=100000,
                tolerance=1e-12,
                omega=1.98,
                components_mm=comps,
                pcb_dimensions_mm=(PCB_SIZE_MM, PCB_SIZE_MM)
            )
            T_flat = T.flatten().astype(np.float32)

            # Store params in (N, 5, 3) format
            for j, c in enumerate(comps):
                cx = (c['x_min'] + c['x_max']) / 2
                cy = (c['y_min'] + c['y_max']) / 2
                params[idx, j] = [cx, cy, c['power']]
            temps[idx] = T_flat

            # Print
            powers_str = ', '.join([f"{c['power']:.1f}W" for c in comps])
            print(f"  [{idx:03d}] {n_comp}C total={total_power:.1f}W "
                  f"[{powers_str}]  T_max={T_flat.max():.1f}C")

            # Save PNG every N
            if (i + 1) % save_every == 0 or i == 0 or i == n_per_count - 1:
                pos_tag = build_position_filename(comps)
                png_name = f"sample_{idx:04d}_{n_comp}C_{int(total_power)}W_{pos_tag}.png"
                png_path = os.path.join(samples_dir, png_name)
                fig, ax = plt.subplots(figsize=(5, 4))
                vmin = max(AMBIENT_TEMP, T.min() - 3)
                vmax = T.max() + 3
                ax.imshow(T, cmap='hot', origin='lower',
                          vmin=vmin, vmax=vmax,
                          extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM])
                for c in comps:
                    rect = plt.Rectangle(
                        (c['x_min'], c['y_min']),
                        c['x_max'] - c['x_min'],
                        c['y_max'] - c['y_min'],
                        linewidth=1.2, edgecolor='cyan', facecolor='none')
                    ax.add_patch(rect)
                    cx = (c['x_min'] + c['x_max']) / 2
                    cy = (c['y_min'] + c['y_max']) / 2
                    ax.text(cx, cy, f"{c['power']:.0f}W",
                            ha='center', va='center', color='white',
                            fontsize=7, fontweight='bold')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title(f'{n_comp}C P={total_power:.0f}W Tmax={T.max():.0f}C')
                plt.colorbar(ax.imshow(T, cmap='hot', origin='lower',
                              vmin=vmin, vmax=vmax,
                              extent=[0, PCB_SIZE_MM, 0, PCB_SIZE_MM]),
                             ax=ax, label='Temp (C)')
                plt.tight_layout()
                plt.savefig(png_path, dpi=120)
                plt.close(fig)
                print(f"         PNG -> {png_name}")

            # CSV row
            row = [idx, n_comp, round(total_power, 2), round(float(T_flat.max()), 2)]
            for j in range(5):
                if j < n_comp:
                    row.extend([round(float(params[idx, j, 0]), 1),
                                round(float(params[idx, j, 1]), 1),
                                round(float(params[idx, j, 2]), 1)])
                else:
                    row.extend(['', '', ''])
            csv_f.write(','.join(str(x) for x in row) + '\n')
            csv_f.flush()
            idx += 1

    csv_f.close()

    # Save .npy
    params_path = os.path.join(output_dir, 'params_count_sweep.npy')
    temps_path  = os.path.join(output_dir, 'temps_count_sweep.npy')
    np.save(params_path, params)
    np.save(temps_path, temps)

    print(f'\nSaved: {params_path}  shape={params.shape}')
    print(f'Saved: {temps_path}  shape={temps.shape}')
    print(f'Saved: {csv_path}')

    # Verify
    print('\n=== Verification ===')
    for n_comp in n_counts:
        mask = np.array([np.sum(~np.isnan(p[:, 0])) == n_comp for p in params])
        count = mask.sum()
        sub_p = params[mask]
        powers = [np.nansum(p[:, 2]) for p in sub_p]
        print(f'  {n_comp}C: {count} samples, power range [{min(powers):.1f}, {max(powers):.1f}] W')

    print('\nDone!')
    return idx


def main():
    parser = argparse.ArgumentParser(description='Run PCB thermal analysis')
    parser.add_argument('--components', '-c', help='Path to components JSON file', default=None)
    parser.add_argument('--output', '-o', help='Output folder', default='thermal_analysis_output')
    parser.add_argument('--grid-size', type=int, default=200, help='Grid resolution (e.g. 200)')
    parser.add_argument('--ambient', type=float, default=25.0, help='Ambient temperature (°C)')
    parser.add_argument('--show-plot', action='store_true', help='Show plot interactively')
    parser.add_argument('--variants', type=int, default=15, help='Generate N layouts; the first is the base layout with zero shifts')
    parser.add_argument('--max-shift', type=int, default=50, help='Maximum shift in mm for variants')
    parser.add_argument('--component-sweep', action='store_true',
                        help='Generate dataset by moving one component at a time')
    parser.add_argument('--count-sweep', action='store_true',
                        help='Generate by component count: 1..N, each with fixed number of maps')
    parser.add_argument('--positions-per-component', type=int, default=40,
                        help='Number of positions for each component in sweep mode')
    parser.add_argument('--positions-per-count', type=int, default=40,
                        help='Number of maps for each component count in count-sweep mode')
    parser.add_argument('--max-components', type=int, default=5,
                        help='Maximum component count in count-sweep mode')
    parser.add_argument('--min-components', type=int, default=1,
                        help='Minimum component count in count-sweep mode')
    parser.add_argument('--min-shift-distance', type=float, default=12.0,
                        help='Minimum shift distance (mm) for count-sweep samples')
    parser.add_argument('--sweep-max-iterations', type=int, default=30000,
                        help='Max solver iterations per sample in component sweep mode')
    parser.add_argument('--sweep-tolerance', type=float, default=1e-9,
                        help='Convergence tolerance per sample in component sweep mode')
    args = parser.parse_args()

    if args.components:
        components_mm = load_components_from_json(args.components)
    else:
        # Default layout with 8 components (8 mm x 8 mm each)
        components_mm = [
            {"name": "U1", "x_min": 8.0,  "x_max": 16.0, "y_min": 8.0,  "y_max": 16.0, "power": 2.5},
            {"name": "U2", "x_min": 28.0, "x_max": 36.0, "y_min": 18.0, "y_max": 26.0, "power": 2.2},
            {"name": "U3", "x_min": 50.0, "x_max": 58.0, "y_min": 34.0, "y_max": 42.0, "power": 3.0},
            {"name": "U4", "x_min": 68.0, "x_max": 76.0, "y_min": 56.0, "y_max": 64.0, "power": 2.8},
            {"name": "U5", "x_min": 82.0, "x_max": 90.0, "y_min": 76.0, "y_max": 84.0, "power": 3.2},
            {"name": "U6", "x_min": 44.0, "x_max": 52.0, "y_min": 58.0, "y_max": 66.0, "power": 2.6},
            {"name": "U7", "x_min": 12.0, "x_max": 20.0, "y_min": 42.0, "y_max": 50.0, "power": 2.4},
            {"name": "U8", "x_min": 66.0, "x_max": 74.0, "y_min": 18.0, "y_max": 26.0, "power": 2.9},
        ]

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    remove_legacy_output_files(args.output)

    if args.count_sweep:
        if len(components_mm) < args.max_components:
            raise ValueError(f"count-sweep requires at least {args.max_components} components")
        n_count_groups = args.max_components - args.min_components + 1
        print(
            f"Generating count sweep: {args.min_components}..{args.max_components}, "
            f"positions_per_count={args.positions_per_count}, "
            f"target_samples={n_count_groups * args.positions_per_count}"
        )
        count_info = generate_count_sweep_dataset(
            components_mm=components_mm,
            pcb_dimensions_mm=(100.0, 100.0),
            positions_per_count=args.positions_per_count,
            max_components=args.max_components,
            min_components=args.min_components,
            max_shift=args.max_shift,
            grid_size=args.grid_size,
            output_folder=args.output,
            ambient_temp=args.ambient,
            max_iterations=args.sweep_max_iterations,
            tolerance=args.sweep_tolerance,
            omega=1.98,
            min_shift_distance=args.min_shift_distance,
        )
        print(f"Count-sweep generation completed: {count_info}")
        T = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
        temp_data = []
    elif args.component_sweep:
        if len(components_mm) < 1:
            raise ValueError("component-sweep mode requires at least 1 component")
        print(
            f"Generating component sweep: components={len(components_mm)}, "
            f"positions_per_component={args.positions_per_component}, "
            f"target_samples={len(components_mm) * args.positions_per_component}"
        )
        sweep_info = generate_component_sweep_dataset(
            components_mm=components_mm,
            pcb_dimensions_mm=(100.0, 100.0),
            positions_per_component=args.positions_per_component,
            max_shift=args.max_shift,
            grid_size=args.grid_size,
            output_folder=args.output,
            ambient_temp=args.ambient,
            max_iterations=args.sweep_max_iterations,
            tolerance=args.sweep_tolerance,
            omega=1.98,
        )
        print(f"Sweep generation completed: {sweep_info}")
        # create placeholders for final summary print
        T = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
        temp_data = []
    elif args.variants and args.variants > 0:
        print(f"Generating {args.variants} layouts (first layout uses zero shifts, max shift {args.max_shift}mm) ...")
        saved = generate_and_save_variants(
            components_mm,
            pcb_dimensions_mm=(100.0, 100.0),
            num_variants=args.variants,
            max_shift=args.max_shift,
            grid_size=args.grid_size,
            output_folder=args.output
        )

        if not saved:
            raise RuntimeError('No valid layouts were generated.')

        first_shifts = [tuple(pair) for pair in saved[0]['shifts']]
        first_components = create_shifted_components(components_mm, first_shifts, (100.0, 100.0))
        T, components, component_temps, temp_data = run_pcb_thermal_analysis(
            components_mm=first_components,
            output_folder=args.output,
            filename_prefix=build_shift_filename(first_shifts),
            grid_size=args.grid_size,
            ambient_temp=args.ambient,
            max_iterations=100000,
            tolerance=1e-12,
            omega=1.98,
            pcb_dimensions_mm=(100.0, 100.0),
            show_plot=args.show_plot,
            save_outputs=False
        )

        print('Saved layout files:')
        for item in saved:
            print(f"{item['name']} -> {item['png_path']} | {item['json_path']}")
    else:
        zero_shifts = [(0, 0) for _ in components_mm]
        zero_name = build_shift_filename(zero_shifts)
        T, components, component_temps, temp_data = run_pcb_thermal_analysis(
            components_mm=components_mm,
            output_folder=args.output,
            filename_prefix=zero_name,
            grid_size=args.grid_size,
            ambient_temp=args.ambient,
            max_iterations=100000,
            tolerance=1e-12,
            omega=1.98,
            pcb_dimensions_mm=(100.0, 100.0),
            show_plot=args.show_plot
        )

    print('\nAnalysis completed successfully!')
    print(f'Results saved to folder: {args.output}')
    print(f'Temperature field shape: {T.shape}')
    print(f'Number of temperature data points: {len(temp_data)}')


if __name__ == '__main__':
    main()
