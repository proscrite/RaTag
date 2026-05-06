"""
RaTag/simulations/xray_efficiency.py

Monte Carlo simulation to determine the geometric detection efficiency 
of isotropic X-rays compared to on-axis recoils.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

# =========================================================================
# 1. CORE PHYSICS & GEOMETRY HELPERS
# =========================================================================

def solid_angle_factor(r_emit: np.ndarray, z_emit: float, z_pmt: float) -> np.ndarray:
    """
    Calculate relative solid angle factor (1/d^2) for light emitted at (r_emit, z_emit)
    and detected at (0, z_pmt).
    """
    dist_sq = r_emit**2 + (z_pmt - z_emit)**2
    return 1.0 / dist_sq

def generate_disk_source(n_events: int, radius_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random points uniformly distributed on a disk at z=0.
    Returns: (x, y, r)
    """
    u = np.random.uniform(0, 1, n_events)
    r = (radius_mm / 10.0) * np.sqrt(u)  # Convert mm to cm
    phi = np.random.uniform(0, 2 * np.pi, n_events)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, r

def propagate_isotropic_rays(x_start: np.ndarray, y_start: np.ndarray,
                             n_events: int, lambda_cm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate rays isotropically from starting points.
    Returns interaction vertices (r_int, z_int).
    """
    # Path length L ~ Exp(lambda)
    L = -lambda_cm * np.log(np.random.uniform(0, 1, n_events))
    
    # Isotropic Emission Direction (Upper Hemisphere)
    cos_theta = np.random.uniform(0, 1, n_events)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, 2 * np.pi, n_events)
    
    # Interaction Vertex
    dx = L * sin_theta * np.cos(phi)
    dy = L * sin_theta * np.sin(phi)
    dz = L * cos_theta
    
    r_int = np.sqrt((x_start + dx)**2 + (y_start + dy)**2)
    z_int = dz 
    return r_int, z_int

def filter_active_volume(r: np.ndarray, z: np.ndarray, z_min: float, z_max: float,
                         r_max: float) -> np.ndarray:
    """Return boolean mask for events within the cylindrical active volume."""
    return (z > z_min) & (z < z_max) & (r < r_max)

# =========================================================================
# 2. MAIN SIMULATION CONTROLLER
# =========================================================================

def run_geometric_efficiency_mc(n_events: int = 100_000, source_radius_mm: float = 5.0,
                                drift_length_cm: float = 1.4, el_gap_cm: float = 0.8,
                                pmt_distance_cm: float = 3.8, grid_radius_cm: float = 6.0,
                                lambda_xe_cm: float = 2.5) -> Dict:
    """
    Execute the Monte Carlo simulation pipeline.
    
    Returns a dictionary containing:
      - g_ratio: The correction factor
      - geometry: Config used
      - debug_data: Arrays for plotting (vertices, acceptances)
    """
    # 1. Geometry Definitions
    z_gate = drift_length_cm
    z_anode = drift_length_cm + el_gap_cm
    z_pmt = z_anode + pmt_distance_cm
    z_light = (z_gate + z_anode) / 2.0

    # 2. Reference Recoils (Source only)
    _, _, r_source = generate_disk_source(n_events, source_radius_mm)
    acc_recoils = solid_angle_factor(r_source, z_light, z_pmt)
    mean_acc_recoil = float(np.mean(acc_recoils))

    # 3. X-ray Propagation
    x_start, y_start, _ = generate_disk_source(n_events, source_radius_mm)
    r_int, z_int = propagate_isotropic_rays(x_start, y_start, n_events, lambda_xe_cm)

    # 4. Filter for Drift Volume
    mask = filter_active_volume(r_int, z_int, z_min=0, z_max=z_gate, r_max=grid_radius_cm)
    
    # 5. Calculate Efficiency for Accepted X-rays
    r_accepted = r_int[mask]
    z_accepted = z_int[mask]
    acc_xrays = solid_angle_factor(r_accepted, z_light, z_pmt)
    mean_acc_xray = float(np.mean(acc_xrays))

    # 6. Construct Result
    return {
        "g_ratio": mean_acc_xray / mean_acc_recoil,
        "mean_acc_recoil": mean_acc_recoil,
        "mean_acc_xray": mean_acc_xray,
        "efficiency_fraction": float(np.sum(mask) / n_events),
        "geometry": {
            "source_radius_mm": source_radius_mm,
            "drift_length_cm": drift_length_cm,
            "el_gap_cm": el_gap_cm,
            "pmt_distance_cm": pmt_distance_cm,
            "grid_radius_cm": grid_radius_cm,
            "lambda_xe_cm": lambda_xe_cm
        },
        # Data for plotting/debugging (not strictly for JSON export)
        "debug_data": {
            "r_accepted": r_accepted,
            "z_accepted": z_accepted,
            "acc_accepted": acc_xrays,
            "acc_recoils": acc_recoils
        }
    }

# =========================================================================
# 3. I/O AND VISUALIZATION
# =========================================================================

def export_results_to_json(results: Dict, filepath: Union[str, Path]) -> None:
    """Save the simulation summary (excluding large arrays) to JSON."""
    # Filter out numpy arrays to keep JSON clean
    output_data = {k: v for k, v in results.items() if k != "debug_data"}
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Simulation results exported to {filepath}")

def visualize_efficiency(results: Dict, bins: int = 50) -> None:
    """Generate the diagnostic dashboard from simulation results."""
    data = results["debug_data"]
    geo = results["geometry"]
    r = data["r_accepted"]
    z = data["z_accepted"]
    
    # Normalize sensitivity
    mean_recoil = results["mean_acc_recoil"]
    rel_sensitivity = data["acc_accepted"] / mean_recoil
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- 1. Raw Vertices ---
    subset = np.random.choice(len(r), size=min(10000, len(r)), replace=False)
    ax[0, 0].scatter(r[subset], z[subset], s=1, c='tab:blue', alpha=0.3)
    ax[0, 0].axhline(0, color='k', lw=2, label='Cathode')
    ax[0, 0].axhline(geo['drift_length_cm'], color='tab:red', lw=2, linestyle='--', label='Gate')
    ax[0, 0].set_title("Accepted X-ray Vertices")
    ax[0, 0].set_ylabel("Drift Z [cm]")
    ax[0, 0].set_xlim(0, geo['grid_radius_cm'])

    # --- 2. Heatmap ---
    hb = ax[0, 1].hexbin(r, z, C=rel_sensitivity, gridsize=bins, 
                         cmap='viridis', reduce_C_function=np.mean, 
                         mincnt=5, vmax=1.0, vmin=0.6)
    ax[0, 1].set_title("Sensitivity Heatmap")
    fig.colorbar(hb, ax=ax[0, 1], label="Relative Efficiency")
    ax[0, 1].set_facecolor('black')
    ax[0, 1].set_xlim(0, geo['grid_radius_cm'])

    # --- 3. Histograms ---
    h_bins = np.linspace(0.5, 1.1, 60)
    ax[1, 0].hist(rel_sensitivity, bins=h_bins, density=True, color='teal', alpha=0.7, label='X-rays')
    ax[1, 0].hist(data["acc_recoils"] / mean_recoil, bins=h_bins, density=True, color='salmon', alpha=0.6, label='Recoils')
    ax[1, 0].axvline(results["g_ratio"], color='cyan', linestyle='--', label=f'Mean Xray ({results["g_ratio"]:.2f})')
    ax[1, 0].set_xlabel("Relative Efficiency")
    ax[1, 0].legend()

    # --- 4. Radial Profile ---
    r_bins = np.linspace(0, geo['grid_radius_cm'], 30)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    inds = np.digitize(r, r_bins)
    mean_eff = [np.mean(rel_sensitivity[inds == i]) if np.sum(inds == i) > 10 else np.nan for i in range(1, len(r_bins))]
    
    ax[1, 1].plot(r_centers, mean_eff, 'o-', color='tab:purple')
    ax[1, 1].axhline(1.0, color='red', linestyle='--', label='On-Axis Recoil')
    ax[1, 1].set_xlabel("Radius [cm]")
    ax[1, 1].set_ylabel("Mean Relative Efficiency")
    ax[1, 1].set_ylim(0.5, 1.05)
    ax[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    res = run_geometric_efficiency_mc()
    print(f"G_ratio: {res['g_ratio']:.4f}")
    visualize_efficiency(res)
    export_results_to_json(res, "xray_sim_results.json")