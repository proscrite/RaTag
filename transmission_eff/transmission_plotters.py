import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np 
from pathlib import Path 
from typing import Optional

from .datatypes import TransmissionRun

def plot_raw_current_timeseries(df: pd.DataFrame, use_pA: bool = True) -> plt.Figure:
    """
    Plots current vs time/index to validate raw electrometer data.
    Takes a dataframe straight from load_keithly_data.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Check if a time column exists, otherwise use index
    x_data = df['Time'] if 'Time' in df.columns else df.index
    
    if use_pA:
        y_data = df['Current_A'] * 1e12
        ylabel = "Current (pA)"
    else:
        y_data = df['Current_A']
        ylabel = "Current (A)"
        
    ax.plot(x_data, y_data, marker='.', linestyle='-', alpha=0.7, color='steelblue')
    ax.set(xlabel='Time' if 'Time' in df.columns else 'Measurement Index', ylabel=ylabel, title='Raw Electrometer Timeseries')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_current_histogram(df: pd.DataFrame, 
                           mean_val: Optional[float] = None, 
                           std_val: Optional[float] = None) -> plt.Figure:
    """
    Plots a histogram of the raw current in pA to visually identify outliers.
    Optionally overlays the calculated mean and outlier thresholds.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Always plot histogram in pA for readability
    currents_pA = df['Current_A'] * 1e12
    
    ax.hist(currents_pA, bins=50, color='lightslategray', edgecolor='black', alpha=0.7)
    
    # Overlay the calculated bounds if provided
    if mean_val is not None and std_val is not None:
        ax.axvline(mean_val, color='crimson', linestyle='-', label=f'Mean: {mean_val:.2f} pA')
        ax.axvline(mean_val + 3*std_val, color='orange', linestyle='--', label='±3 Sigma Cut')
        ax.axvline(mean_val - 3*std_val, color='orange', linestyle='--')
        ax.legend()
        
    ax.set(xlabel='Current (pA)', ylabel='Frequency', title='Current Distribution with Outlier Thresholds')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_transmission_curve(run_data: TransmissionRun, sweep_variable: str = 'v_gate', ax: Optional[plt.Axes] = None) -> plt.Figure:
    """  Plots the final transmission curve for a single run. """
    if not run_data.points:
        print(f"  -> Skipping drawing {run_data.run_id}: list of points is empty.")
        # Return whatever figure exists if we are in a multi-plot, or a blank one
        return ax.figure if ax else plt.figure()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    if sweep_variable == 'v_gate':
        x_vals = [pt.v_gate for pt in run_data.points]
        xlabel = "Gate Voltage (V)"
        ylabel = "Anode Current (pA)"
    elif sweep_variable == 'v_cathode':
        x_vals = [pt.v_cathode for pt in run_data.points]
        xlabel = "Cathode Voltage (V)"
        ylabel = "Gate Current (pA)"
    else:
        x_vals = [pt.v_anode for pt in run_data.points]
        xlabel = "Anode Voltage (V)"
        ylabel = "Gate Current (pA)"
        
    y_vals = [pt.i_mean_pA for pt in run_data.points]
    y_errs = [pt.i_std_pA for pt in run_data.points]
    
    sorted_indices = sorted(range(len(x_vals)), key=lambda k: x_vals[k])
    x_vals = [x_vals[i] for i in sorted_indices]
    y_vals = [y_vals[i] for i in sorted_indices]
    y_errs = [y_errs[i] for i in sorted_indices]
    
    # Add label here so the multi-plot legend picks it up!
    run_label = f"{run_data.run_id}: {run_data.description}"
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='-o', 
                capsize=4, capthick=1.5, markersize=6, label=run_label)
    
    # 2. Add metadata
    fixed_text = []
    if sweep_variable != 'v_gate': fixed_text.append(f"Vg: {run_data.points[0].v_gate}V")
    if sweep_variable != 'v_cathode': fixed_text.append(f"Vc: {run_data.points[0].v_cathode}V")
    if sweep_variable != 'v_anode': fixed_text.append(f"Va: {run_data.points[0].v_anode}V")
    
    # Note: If it's a multi-plot, the orchestrator might overwrite this title later, which is fine!
    ax.set(xlabel=xlabel, ylabel=ylabel, 
           title=f"{run_label}\nFixed: {', '.join(fixed_text)}")
    
    fig.tight_layout()
    return fig


def plot_multiple_transmission_curves(runs: list[TransmissionRun], sweep_variable: str, title: str) -> plt.Figure:
    """
    Plots multiple transmission curves on the same figure by injecting a shared 'ax'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_data in runs:
        plot_transmission_curve(run_data, sweep_variable=sweep_variable, ax=ax)
    
    # Override the title with the Master Title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_summary_curve(run_data: TransmissionRun, sweep_variable: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    # We filter out the V_anode=0 point so it doesn't anchor the plot at T=0
    plot_points = [pt for pt in run_data.points if pt.v_anode != 0]

    if sweep_variable == 'v_anode':
        x_vals = [pt.v_anode for pt in plot_points]
        ax.set_xlabel("Anode Voltage (V)")
    elif sweep_variable == 'e_el_V_cm':
        x_vals = [pt.e_el_V_cm for pt in plot_points]
        ax.set_xlabel("EL Field (V/cm)")
    elif sweep_variable == 'r_factor':
        x_vals = [pt.r_factor for pt in plot_points]
        ax.set_xlabel("R factor (E_drift / E_EL)")
        
    y_vals = [pt.transmission for pt in plot_points]
    y_errs = [pt.transmission_err for pt in plot_points]
    
    label = f"{run_data.run_id} (E_drift: {run_data.drift_field_V_cm:.0f} V/cm)"
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='-o', capsize=4, markersize=6, label=label)
    ax.set_ylabel("Gate Transmission Efficiency (GTE)")
    
    return ax

def plot_sigmoid_fit(run_data: TransmissionRun, sweep_variable: str, ax=None):
    from RaTag.core.fitting import sigmoid
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if not run_data.fit_params:
        print(f"  -> No fit parameters for {run_data.run_id}, cannot plot sigmoid fit.")
        return ax
    
    k = run_data.fit_params['k']
    x0 = run_data.fit_params['x0']
    
    # Generate a smooth curve for the fit
    if sweep_variable == 'r_factor':
        x_fit = np.linspace(0, 35, 100)
    elif sweep_variable == 'e_el_V_cm':
        x_fit = np.linspace(0, 10_000, 100)
    else:
        x_fit = np.linspace(0, max(pt.v_anode for pt in run_data.points), 100)
    
    y_fit = sigmoid(x_fit, k, x0)
    
    label = f"{run_data.run_id} Fit (k={k:.2f}, x0={x0:.2f})"
    ax.plot(x_fit, y_fit, linestyle='--', color='gray', label="__nolabel__")
    
    return ax