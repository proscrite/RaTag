import build
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional
from dataclasses import replace
from scipy.interpolate import RegularGridInterpolator

from .datatypes import TransmissionRun, TransmissionPoint
from .etl_keithly import process_single_measurement, calculate_transmission_efficiency, calculate_electric_fields
from .plotting import plot_summary_curve, plot_multiple_transmission_curves, plot_sigmoid_fit
from RaTag.core.fitting import fit_sigmoid 

# ---------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------

def process_single_run(run_id: str, run_data: dict, base_dir: Path, el_gap_cm=0.8, drift_gap_cm=1.4) -> TransmissionRun:
    points = []
    fixed_params = run_data.get('fixed_params', {})
    
    # 1. First Pass: Load data and compute fields immediately
    for point_config in run_data['measurements']:
        v_cat = point_config.get('cathode_voltage', fixed_params.get('cathode_voltage'))
        v_gate = point_config.get('gate_voltage', fixed_params.get('gate_voltage'))
        v_anode = point_config.get('anode_voltage', fixed_params.get('anode_voltage'))
        
        prefix = point_config['file_prefix']
        matching_files = list(base_dir.glob(f"{prefix}_CurrentLog_*.txt"))
        
        if not matching_files:
            continue
            
        mean_pA, std_pA = process_single_measurement(matching_files[0])
        e_drift, e_el, r_factor = calculate_electric_fields(v_cat, v_gate, v_anode, el_gap_cm, drift_gap_cm)
        
        points.append(TransmissionPoint(
            v_cathode=v_cat, v_gate=v_gate, v_anode=v_anode,
            i_mean_pA=mean_pA, i_std_pA=std_pA,
            e_drift_V_cm=e_drift, e_el_V_cm=e_el, r_factor=r_factor
        ))
        
    if not points:
        return TransmissionRun(run_id=run_id, description=run_data.get('description', ''), points=[])

    # 2. Add Transmission if I0 exists
    zero_pts = [pt for pt in points if pt.v_anode == 0]
    if zero_pts:
        p0 = zero_pts[0]
        updated_points = []
        for pt in points:
            if pt.v_anode == 0:
                updated_points.append(pt) # Keep T=0 for normalization point
            else:
                t, t_err = calculate_transmission_efficiency(pt.i_mean_pA, pt.i_std_pA, p0.i_mean_pA, p0.i_std_pA)
                # 'replace' cleanly creates a new frozen dataclass with the updated values
                updated_points.append(replace(pt, transmission=t, transmission_err=t_err))
        
        points = updated_points

    # Sort by R-factor for clean plotting
    points.sort(key=lambda x: x.r_factor)
    drift_field = points[0].e_drift_V_cm if points else 0.0

    return TransmissionRun(run_id=run_id, 
                           description=run_data.get('description', ''), 
                           points=points,
                           drift_field_V_cm=drift_field) 


# ---------------------------------------------------------
# Task 1: Data Assembly
# ---------------------------------------------------------
def build_run_catalog(runs_config: dict, base_dir: Path) -> dict:
    """Wraps the workflow layer to generate dataclasses for all runs."""
    return {
        run_id: process_single_run(run_id, data, base_dir) 
        for run_id, data in runs_config.items()
    }

# ---------------------------------------------------------
# Task 2: Modeling and Export
# ---------------------------------------------------------
def export_transparency_model(runs_catalog: dict, out_file: Path):
    """Fits sigmoids to each run and exports the parameter anchors."""
    anchors = {"e_drift_V_cm": [], "k": [], "x0": []}
    updated_catalog = {}

    for run_id, run_data in runs_catalog.items():
        # Only fit runs that actually sweep the R factor (skip t=0 normalization runs)
        if any(pt.transmission > 0 for pt in run_data.points):
            # Extract arrays for fitting
            r_factors = [pt.r_factor for pt in run_data.points if pt.v_anode > 0]
            transmissions = [pt.transmission for pt in run_data.points if pt.v_anode > 0]
            
            try:
                popt, R_fit, T_fit = fit_sigmoid(r_factors, transmissions)
            except Exception as e:
                print(f"  Fit failed for {run_id}: {e}")
                try:
                    popt, R_fit, T_fit = fit_sigmoid(r_factors, transmissions, p0=[0.3, 3.0])  # Try a different initial guess
                    print(f"  Fit succeeded for {run_id} with alternative initial guess.")
                except Exception as e2:
                    print(f"  Fit failed even with alternative initial guess for {run_id}: {e2}")
                    continue
            
            if popt is not None:
                # We assume E_drift is constant per run
                e_drift = run_data.points[0].e_drift_V_cm
                anchors["e_drift_V_cm"].append(e_drift)
                anchors["k"].append(float(popt[0]))
                anchors["x0"].append(float(popt[1]))
                updated_catalog[run_id] = replace(run_data, fit_params={"k": float(popt[0]), "x0": float(popt[1])})
            else:
                print(f"  Fit did not converge for {run_id}. Skipping anchor extraction.")
                updated_catalog[run_id] = run_data
        else:
            print(f"  No valid transmission points for {run_id}. Skipping fit.")
            updated_catalog[run_id] = run_data
    # Sort by drift field to ensure np.interp works correctly downstream
    sorted_indices = sorted(range(len(anchors["e_drift_V_cm"])), key=lambda i: anchors["e_drift_V_cm"][i])
    sorted_anchors = {key: [val[i] for i in sorted_indices] for key, val in anchors.items()}
    
    with open(out_file, "w") as f:
        json.dump({"model": "dynamic_sigmoid", "anchors": sorted_anchors}, f, indent=4)
    print("Exported physics-informed parameter anchors.")

    return updated_catalog

# ---------------------------------------------------------
# Task 3: Visualizations
# ---------------------------------------------------------
def generate_diagnostic_plots(runs_catalog: dict, plots_config: dict, out_path: Path):
    """Handles all matplotlib generation, driven by the config."""
    for plot_id, plot_config in plots_config.items():
        if plot_config.get('type') != 'derived':
            continue
        print(f"Generating Plot: {plot_config['title']}...")
        sweep_var = plot_config.get('sweep_variable', 'r_factor')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_id in plot_config['runs']:
            if run_id in runs_catalog and any(pt.transmission > 0 for pt in runs_catalog[run_id].points):
                run_data = runs_catalog[run_id]
                
                # Base scatter/line plot
                plot_summary_curve(run_data, sweep_var, ax)
                
                # Overlay fit if it's a derived plot
                if plot_id == "gte_vs_r_factor":
                    plot_sigmoid_fit(run_data, sweep_var, ax)
                    
        ax.set_title(plot_config['title'], fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        fig.savefig(str(out_path / f"{plot_id}.png"), dpi=300)
        plt.close(fig)


def generate_physics_surface(runs_catalog: dict, output_dir: Path):
    """Reconstructs the 2D surface from catalog fits and plots it."""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ed_anchors, k_anchors, x0_anchors = [], [], []

    for run in runs_catalog.values():
        if getattr(run, 'fit_params', None) is not None:
            ed_anchors.append(run.points[0].e_drift_V_cm)
            k_anchors.append(run.fit_params['k'])
            x0_anchors.append(run.fit_params['x0'])
            
    if not ed_anchors:
        print("No fits available to generate surface.")
        return
        
    sorted_idx = np.argsort(ed_anchors)
    ed_anchors, k_anchors, x0_anchors = np.array(ed_anchors)[sorted_idx], np.array(k_anchors)[sorted_idx], np.array(x0_anchors)[sorted_idx]
    
    ed_range = np.linspace(max(10, min(ed_anchors) - 100), max(ed_anchors) + 200, 50)
    r_range = np.linspace(0, 35, 50)
    ED, R = np.meshgrid(ed_range, r_range)
    
    
    k_dynamic = np.interp(ED, ed_anchors, k_anchors)
    x0_dynamic = np.interp(ED, ed_anchors, x0_anchors)
    T_sigmoid = np.clip(1.0 / (1.0 + np.exp(-k_dynamic * (R - x0_dynamic))), 0.0, 1.0)
    
    ax.plot_surface(ED, R, T_sigmoid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Drift Field (V/cm)', labelpad=10)
    ax.set_ylabel('R (E_EL/E_d)', labelpad=10)
    ax.set_zlabel('Transmission T', labelpad=10)
    save_file = output_dir / "transparency_surface_sigmoid.png"
    fig.savefig(str(save_file), dpi=300, bbox_inches='tight')

    # 3. Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    evaluator = build_transparency_evaluator(output_dir / "transmission_model.json")
    T_hybrid = evaluator(ED, ED * R)
    
    ax.plot_surface(ED, R, T_hybrid, cmap='plasma', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Drift Field (V/cm)', labelpad=10)
    ax.set_ylabel('R (E_EL/E_d)', labelpad=10)
    ax.set_zlabel('Transmission T', labelpad=10)
    ax.set_title("Empirical Transparency Surface", fontsize=14, fontweight='bold')
    
    save_file = output_dir / "transparency_surface_linear.png"
    fig.savefig(str(save_file), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig)


def build_transparency_evaluator(json_path: Path):
    """
    Builds a pure functional evaluator using RegularGridInterpolator 
    over the 1D physics-informed sigmoid fits stored in a JSON artifact.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    anchors = data.get("anchors", {})
    
    # Extract arrays (handling potential key differences based on your export logic)
    ed_anchors = np.array(anchors.get("e_drift_V_cm", []))
    k_anchors = np.array(anchors.get("k", anchors.get("sigmoid_k", [])))
    x0_anchors = np.array(anchors.get("x0", anchors.get("sigmoid_x0", [])))
    
    if len(ed_anchors) == 0:
        raise ValueError(f"No valid anchors found in {json_path}")
        
    # 1. Sort them to ensure strictly increasing axes
    idx = np.argsort(ed_anchors)
    ed_axis = ed_anchors[idx]
    k_axis = k_anchors[idx]
    x0_axis = x0_anchors[idx]
    
    # 2. Define the R axis (extended to R=40 to guarantee the flat saturation plateau)
    r_axis = np.linspace(0, 40, 200) 
    
    # 3. Populate the Grid Values (shape: len(ed_axis) x len(r_axis))
    grid_values = np.zeros((len(ed_axis), len(r_axis)))
    for i, (k, x0) in enumerate(zip(k_axis, x0_axis)):
        grid_values[i, :] = 1.0 / (1.0 + np.exp(-k * (r_axis - x0)))
        
    # 4. Create the Interpolator (Extrapolates flatly at the boundaries)
    interpolator = RegularGridInterpolator((ed_axis, r_axis), grid_values, 
                                           bounds_error=False, fill_value=None)
    
    # 5. The Pure Function Closure
    def evaluate_transparency(e_drift, e_el):
        e_drift = np.asarray(e_drift)
        r_ratio = e_el / e_drift
        
        # Flatten and stack to support both 1D arrays and 2D meshgrids
        pts = np.column_stack((e_drift.flatten(), r_ratio.flatten()))
        t_val = interpolator(pts)
        
        return np.clip(t_val.reshape(e_drift.shape), 0.0, 1.0)
        
    return evaluate_transparency
# ------------------------------------------------------------------------------
# Additional Plotting (not strictly required for the model) for completeness
# ------------------------------------------------------------------------------


def generate_primary_plots(all_runs: dict, plots_config: dict, output_dir: Path):
    """Orchestrates data extraction, plotting, and artifact saving."""
    
    
    for plot_id, plot_config in plots_config.items():
        if plot_config.get('type', 'primary') != 'primary':
            continue
        print(f"Generating Plot: {plot_config['title']}...")
        
        # Gather the runs needed for this specific plot
        plot_runs = [all_runs[run_id] for run_id in plot_config['runs'] 
                     if run_id in all_runs and len(all_runs[run_id].points) > 0]
        
        if not plot_runs:
            print(f"  Warning: No valid runs found for {plot_id}. Skipping.")
            continue
            
        # Create the figure
        fig = plot_multiple_transmission_curves(runs=plot_runs, 
                                                sweep_variable=plot_config['sweep_variable'],
                                                title=plot_config['title'] )
        
        # Save the artifact
        save_path = output_dir / f"{plot_id}.png"
        fig.savefig(save_path, dpi=300)
        print(f"  Saved -> {save_path}")
    
    return all_runs