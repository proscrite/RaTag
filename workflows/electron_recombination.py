"""
Calibration module for X-ray based energy calibration and recombination analysis.

This module provides functions to:
1. Fit X-ray S2 area distributions to extract calibration constants
2. Compute gain factors (g_S2) from known X-ray energies
3. Calculate electron recombination fractions from ion S2 areas
"""

from os import path
from typing import Dict, Tuple, Optional
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import replace

from RaTag.core.datatypes import Run, S2Areas


def compute_calibration_constants(A_x_mean: float, E_gamma: float, W_value: float) -> Tuple[float, float]:
    """
    Compute calibration constants from X-ray measurements.

    Args:
        A_x_mean: Mean X-ray S2 area (mV·µs)
        E_gamma: X-ray photon energy (eV)
        W_value: Mean energy per electron-ion pair (eV)

    Returns:
        Tuple of (N_e_exp, g_S2) where:
            - N_e_exp: Expected number of electrons
            - g_S2: Gain factor (mV·µs per electron)
    """
    N_e_exp = E_gamma / W_value
    g_S2 = A_x_mean / N_e_exp
    return N_e_exp, g_S2


def _plot_calibration_results(run: Run, df_recomb: pd.DataFrame) -> plt.Figure:
    """Generate diagnostic plots for calibration results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Normalized ion S2 areas
    ax1.errorbar(df_recomb['drift_field'], df_recomb['N_e_meas'], 
                 yerr=df_recomb['dN_e_meas'], fmt='o',
                 markersize=8, capsize=5, label='$A_{ion} / A_x$')
    ax1.set(xlabel='Drift Field (V/cm)', ylabel='Measured electrons', title=f'{run.run_id}: Ion recoil electrons vs Drift Field')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Recombination fraction
    ax2.errorbar(df_recomb['drift_field'], df_recomb['recomb_factor'],
                 yerr=df_recomb['recomb_error'], fmt='s',
                 markersize=8, capsize=5, color='crimson', label='Recombination fraction $r$')
    ax2.set(xlabel='Drift Field (V/cm)', ylabel='Recombination Fraction $r$', title=f'{run.run_id}: Electron recombination factor vs Drift Field')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    return fig


def _load_gs2_data(path_gs2_results: str) -> Tuple[float, float, float]:
    """
    Load g_S2 calibration data from JSON file.

    Args:
        path_gs2_results: Path to the JSON file with g_S2 results
    Returns:
        Tuple of (g_S2, uncertainty_g_S2)
    """
    
    with open(path_gs2_results, 'r') as f:
        gs2_data = json.load(f)
    g_S2 = gs2_data['gs2']
    error_g_S2 = gs2_data['d_gs2']
    W_factor = gs2_data.get('Wi', 22)
    return g_S2, error_g_S2, W_factor


def _load_ion_fitted_areas(run: Run) -> pd.DataFrame:
    """
    Load fitted ion S2 areas from disk.

    Args:
        run: Run object containing sets
        suffix: Suffix for the stored ion area files

    Returns:
        DataFrame with fitted ion S2 areas per set
    """
    path_csv = run.root_directory / 'processed_data' / f'{run.run_id}_s2_vs_drift.csv'
    return pd.read_csv(path_csv)

def _get_expected_electrons(run: Run, W_factor: float) -> float:
    """
    Compute expected number of electrons from recoil energy.

    Args:
        run: Run object with recoil energy information
        W_factor: Mean energy per electron-ion pair (eV)
    Returns:
        Expected number of electrons
    """
    E_recoil = run.recoil_energy * 1e3 # keV to eV
    N_e_exp = E_recoil / W_factor
    return N_e_exp

def compute_recombination(df_recoil: pd.DataFrame,
                          g_S2: float, N_e_exp: float) -> pd.DataFrame:
    """
    Compute recombination fractions from ion S2 areas.

    Args:
        df_recoil: DataFrame with fitted ion S2 areas and uncertainties
        g_S2: Gain factor (mV·µs per electron)
        N_e_exp: Expected number of electrons from recoil energy
    Returns:
        DataFrame with added columns for measured electrons and recombination fractions
    """
    ion_areas = df_recoil['s2_mean'].values
    ion_uncerts = df_recoil['s2_ci95'].values

    print(f"  → Loaded fitted ion S2 areas for {len(ion_areas)} drift field points")

    # Measured electron numbers
    N_e_meas = ion_areas / g_S2 # type: ignore
    dN_e_meas = ion_uncerts / g_S2 # type: ignore

    # Recombination fraction: r = 1 - N_e_meas / N_e_exp
    r = 1 - N_e_meas / N_e_exp
    dr = dN_e_meas / N_e_exp

    df_recoil['N_e_meas'] = N_e_meas
    df_recoil['dN_e_meas'] = dN_e_meas
    df_recoil['recomb_factor'] = r
    df_recoil['recomb_error'] = dr

    return df_recoil


def recombination_workflow(run: Run, path_gs2: str):
    """
    Complete calibration and recombination analysis pipeline.

    Args:
        run: Run object with X-ray and ion data
        ion_fitted_areas: Dictionary of fitted ion S2 areas per set.
                         If None, will attempt to load from disk.
        xray_bin_cuts: Range for X-ray histogram fitting
        xray_nbins: Number of bins for X-ray histogram
        flag_plot: If True, generate diagnostic plots

    Returns:
        DataFrame with recombination analysis results
    """
    print("=" * 60)
    print("RECOMBINATION ANALYSIS")
    print("=" * 60)
    
    # 1. Load X-ray results and ion fitted areas
    print("\n[1/4] Loading X-ray calibration results...")
    g_S2, error_g_S2, W_factor = _load_gs2_data(path_gs2)
    df_recoil = _load_ion_fitted_areas(run)
    
    print(f"  → g_S2: {g_S2:.4f} ± {error_g_S2:.4f} mV·µs/electron")
    print(f"  → W factor: {W_factor} eV")
    
    # 2. Compute Expected number of electrons
    print("\n[2/4] Computing expected number of electrons from recoil energy...")
    N_e_exp = _get_expected_electrons(run, W_factor)

    print(f"  → Expected number of electrons: {N_e_exp:.1f} e-")

    # 3. Extract ion S2 areas and compute recombination
    print("\n[3/4] Computing recombination fractions...")
    df_recombination = compute_recombination(df_recoil, g_S2, N_e_exp)
    
    print(f"  → Recombination fractions computed for {len(df_recombination)} field points")
    print(f"  → r range: [{df_recombination['recomb_factor'].min():.3f}, {df_recombination['recomb_factor'].max():.3f}]")
    
    # 5. Generate plots and persist results
    print("\n[4/4] Plotting and storing recombination results...")
    fig = _plot_calibration_results(run, df_recombination)
    pathout_fig = run.root_directory / 'processed_data' / f'{run.run_id}_recombination_plots.png'
    fig.savefig(pathout_fig, dpi=300)

    pathout_recomb = run.root_directory / 'processed_data' / f'{run.run_id}_recomb_factors.csv'
    df_recombination.to_csv(pathout_recomb, index=False)
    print(f"  → Recombination results stored at: {pathout_recomb}")
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    
    return df_recombination

