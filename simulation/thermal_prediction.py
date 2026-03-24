import numpy as np
import matplotlib.pyplot as plt
import time
import numba
import json
import os
import argparse
import random

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
                           filename_prefix="pcb_thermal",
                           grid_size=200, 
                           ambient_temp=25.0,
                           max_iterations=100000, 
                           tolerance=1e-12,
                           omega=1.98,
                           pcb_dimensions_mm=(100.0, 100.0),
                           show_plot=False):
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
    image_path = os.path.join(output_folder, f"{filename_prefix}_thermal_map.png")
    fig = plot_temperature(T, components, pcb_dimensions_mm, save_path=image_path, show_plot=show_plot)
    plt.close(fig)
    
    # Generate and save temperature data as JSON
    temp_data = generate_temperature_json_data(T, pcb_dimensions_mm, grid_size)
    json_path = os.path.join(output_folder, f"{filename_prefix}_temperatures.json")
    with open(json_path, 'w') as f:
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


def generate_non_overlapping_shifts(components_mm, pcb_dimensions_mm, num_variants=15, max_shift=50, max_attempts=200):
    """Generate a list of shift-lists. Each shift-list is [(dx,dy), ...] for each component.
    Attempts random integer shifts in range [-max_shift, max_shift] (mm) and enforces no overlap.
    """
    variants = []
    n = len(components_mm)
    length_mm, width_mm = pcb_dimensions_mm

    # original rects
    orig_rects = []
    for c in components_mm:
        orig_rects.append((c['x_min'], c['x_max'], c['y_min'], c['y_max']))

    tries = 0
    while len(variants) < num_variants and tries < max_attempts * num_variants:
        tries += 1
        shifts = []
        for i in range(n):
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            shifts.append((dx, dy))

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

    return variants


def generate_and_save_variants(components_mm, pcb_dimensions_mm=(100.0,100.0), num_variants=15, max_shift=50, grid_size=200, output_folder='thermal_analysis_output'):
    """Generate variants, run simulation for each, and save temperature JSON named by shifts.
    File name format: dx1,dy1,dx2,dy2,... .json
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    variants = generate_non_overlapping_shifts(components_mm, pcb_dimensions_mm, num_variants=num_variants, max_shift=max_shift)
    saved = []
    for shifts in variants:
        # create shifted components
        new_comps = create_shifted_components(components_mm, shifts, pcb_dimensions_mm)

        # run analysis (we only need temp_data)
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
            show_plot=False
        )

        # build filename from shifts
        flat = []
        for dx, dy in shifts:
            flat.append(str(int(dx)))
            flat.append(str(int(dy)))
        name = ','.join(flat)
        json_path = os.path.join(output_folder, f"{name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2)

        # Save a PNG heatmap with the same base name as the JSON
        png_path = os.path.join(output_folder, f"{name}.png")
        fig = plot_temperature(T, comps, pcb_dimensions_mm, save_path=png_path, show_plot=False)
        plt.close(fig)

        saved.append(json_path)

    return saved
def load_components_from_json(path):
    """Load component list from a JSON file.
    The file should contain a list of components with keys:
    name, x_min, x_max, y_min, y_max, power
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Run PCB thermal analysis')
    parser.add_argument('--components', '-c', help='Path to components JSON file', default=None)
    parser.add_argument('--output', '-o', help='Output folder', default='thermal_analysis_output')
    parser.add_argument('--grid-size', type=int, default=200, help='Grid resolution (e.g. 200)')
    parser.add_argument('--ambient', type=float, default=25.0, help='Ambient temperature (°C)')
    parser.add_argument('--show-plot', action='store_true', help='Show plot interactively')
    parser.add_argument('--variants', type=int, default=0, help='Generate N shifted variants and save JSONs')
    parser.add_argument('--max-shift', type=int, default=50, help='Maximum shift in mm for variants')
    args = parser.parse_args()

    if args.components:
        components_mm = load_components_from_json(args.components)
    else:
        # Minimal default component layout if none provided
        components_mm = [
            {"name": "CPU", "x_min": 16.0, "x_max": 24.0, "y_min": 16.0, "y_max": 24.0, "power": 2.0},
            {"name": "GPU", "x_min": 60.0, "x_max": 70.0, "y_min": 50.0, "y_max": 60.0, "power": 4.0}
        ]

    T, components, component_temps, temp_data = run_pcb_thermal_analysis(
        components_mm=components_mm,
        output_folder=args.output,
        filename_prefix='pcb_layout',
        grid_size=args.grid_size,
        ambient_temp=args.ambient,
        max_iterations=100000,
        tolerance=1e-12,
        omega=1.98,
        pcb_dimensions_mm=(100.0, 100.0),
        show_plot=args.show_plot
    )

    # If variants requested, generate and save them (this will run extra simulations)
    if args.variants and args.variants > 0:
        print(f"Generating {args.variants} variants (max shift {args.max_shift}mm) ...")
        saved = generate_and_save_variants(components_mm, pcb_dimensions_mm=(100.0,100.0), num_variants=args.variants, max_shift=args.max_shift, grid_size=args.grid_size, output_folder=args.output)
        print('Saved variant JSONs:')
        for p in saved:
            print(p)

    print('\nAnalysis completed successfully!')
    print(f'Results saved to folder: {args.output}')
    print(f'Temperature field shape: {T.shape}')
    print(f'Number of temperature data points: {len(temp_data)}')


if __name__ == '__main__':
    main()
