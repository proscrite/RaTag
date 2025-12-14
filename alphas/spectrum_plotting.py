"""
Alpha spectrum visualization functions.

This module provides plotting utilities for:
1. Individual peak fits (Crystal Ball model)
2. Multi-peak preliminary fits
3. Energy calibration curves and residuals
4. Calibrated spectra with literature markers
5. Isotope energy ranges
6. Hierarchical fits with deconvolved components

All functions return matplotlib figure/axis objects for further customization.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from lmfit.model import ModelResult

from RaTag.alphas.spectrum_fitting import SpectrumData, EnergyCalibration, IsotopeRange


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def mark_peak_position(ax, x0, name, color='red', y_fraction=0.95):
    """
    Helper to mark peak position with vertical line and label.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    x0 : float
        Peak position (fitted x0 value)
    name : str
        Peak name for label
    color : str
        Line and text color
    y_fraction : float
        Fraction of y-axis limit for text placement (0-1)
    """
    ax.axvline(x0, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(x0, ax.get_ylim()[1] * y_fraction, name, 
            rotation=90, ha='left', va='top', fontsize=10, 
            fontweight='bold', color=color)


def plot_fit_peak(ax, fit_result, name, color, x_range=None, n_points=300):
    """
    Helper function to plot a fitted peak with vertical line marker.
    
    Args:
        ax: Matplotlib axis
        fit_result: lmfit ModelResult
        name: Peak name for legend
        color: Plot color
        x_range: (x_min, x_max) for evaluation grid (if None, use ±0.5 around peak)
        n_points: Number of points for smooth curve
    """
    # Extract peak position (works for both regular CB and Po212 composite fits)
    x0 = fit_result.params['cb_x0'].value
    
    # Define evaluation grid
    if x_range is None:
        E_fine = np.linspace(x0 - 0.5, x0 + 0.5, n_points)
    else:
        E_fine = np.linspace(x_range[0], x_range[1], n_points)
    
    # Evaluate fit and plot
    fit_curve = fit_result.eval(x=E_fine)
    ax.plot(E_fine, fit_curve, color=color, linewidth=2, 
            label=f'{name} (x₀={x0:.3f} MeV)', alpha=0.9)
    ax.axvline(x0, color=color, linestyle='--', linewidth=1, alpha=0.8)


def plot_residuals(ax, x, data, model, show_poisson=True):
    """Plot residuals with optional Poisson error bands.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    x : array
        X-axis values (energy)
    data : array
        Observed data counts
    model : array
        Model prediction
    show_poisson : bool
        If True, show ±√N Poisson error bands
    """
    residuals = data - model
    ax.step(x, residuals, where='mid', color='black', linewidth=1.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    
    if show_poisson:
        ax.fill_between(x, -np.sqrt(data), np.sqrt(data), 
                        alpha=0.3, color='gray', label='±√N')
    
    ax.set_ylabel('Residuals')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
 

def plot_preliminary_fits(spectrum, fit_results, peak_names, calibration=None, 
                         ax=None, figsize=(8, 4)):
    """
    Plot preliminary multi-peak fits in SCA scale.
    
    Parameters
    ----------
    spectrum : SpectrumData
        Raw spectrum data (in SCA scale)
    fit_results : Dict[str, ModelResult]
        Fitted peak results from fit_multi_crystalball_progressive
    peak_names : List[str]
        Names of peaks to plot (in order)
    calibration : EnergyCalibration, optional
        If provided, adds top x-axis with calibrated energy scale and marks fitted positions
    ax : matplotlib axis, optional
        If provided, plot on this axis. Otherwise create new figure.
    figsize : Tuple[float, float]
        Figure size (only used if ax is None)
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot data histogram
    ax.hist(spectrum.energies, bins=120, range=(4, 8.2), 
            histtype='step', color='black', linewidth=1.5, label='Data')
    
    # Plot all peaks
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(peak_names)))
    for i, name in enumerate(peak_names):
        if name not in fit_results:
            continue
        plot_fit_peak(ax, fit_results[name], name, colors[i])
    
    # Mark fitted peak positions from fit_results
    for name in peak_names:
        if name in fit_results:
            x0_SCA = fit_results[name].params['cb_x0'].value
            mark_peak_position(ax, x0_SCA, name, color='red', y_fraction=0.95)
    
    ax.set(xlabel='Energy (mV) - SCA Scale (uncalibrated)', ylabel='Counts', 
           title=f'Preliminary Fit: {len(fit_results)} Peaks')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    # Add top x-axis with calibrated energy if calibration provided
    if calibration is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(calibration.apply(np.array(ax.get_xlim())))
        ax2.set_xlabel('Energy (MeV) - Calibrated', fontsize=10)
    
    if ax is None:
        plt.tight_layout()
    
    return fig, ax


def plot_energy_calibration(calibration_linear, calibration_quad, ax1=None, ax2=None, figsize=(14, 5)):
    """
    Plot energy calibration curves and residuals comparison.
    
    Parameters
    ----------
    calibration_linear : EnergyCalibration
        Linear calibration (order=1)
    calibration_quad : EnergyCalibration
        Quadratic calibration (order=2)
    ax1 : matplotlib axis, optional
        Axis for calibration curves (if None, create new figure)
    ax2 : matplotlib axis, optional
        Axis for residuals (if None, create new figure)
    figsize : Tuple[float, float]
        Figure size (only used if axes are None)
        
    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes
    """
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax1.figure
    
    # Left: Calibration curves comparison
    E_SCA_range = np.linspace(4, 9, 200)
    ax1.plot(E_SCA_range, calibration_linear.apply(E_SCA_range), 'b--', 
             linewidth=2, label='Linear', alpha=0.7)
    ax1.plot(E_SCA_range, calibration_quad.apply(E_SCA_range), 'r-', 
             linewidth=2.5, label='Quadratic')
    
    # Plot anchor points
    for name, (E_SCA, E_true) in calibration_quad.anchors.items():
        ax1.plot(E_SCA, E_true, 'ko', markersize=10)
        ax1.annotate(name, (E_SCA, E_true), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax1.set(xlabel='E_SCA (mV) - Instrumental Scale', ylabel='E_true (MeV)', 
            title='Energy Calibration Curves')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Right: Residuals comparison
    names = list(calibration_quad.anchors.keys())
    x = np.arange(len(names))
    
    ax2.bar(x - 0.175, calibration_linear.residuals, 0.35, label='Linear', 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax2.bar(x + 0.175, calibration_quad.residuals, 0.35, label='Quadratic', 
            color='tomato', alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set(xlabel='Isotope', ylabel='Residual (MeV)', title='Calibration Residuals', 
            xticks=x, xticklabels=names)
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(fontsize=11)
    
    if ax1 is None or ax2 is None:
        plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_isotope_ranges(spectrum_calibrated, isotope_ranges, ax=None, figsize=(12, 6)):
    """
    Plot derived isotope energy ranges with purity windows.
    
    Parameters
    ----------
    spectrum_calibrated : np.ndarray
        Calibrated energy values (in MeV)
    isotope_ranges : Dict[str, IsotopeRange]
        Derived isotope ranges (contains E_peak from fits)
    ax : matplotlib axis, optional
        If provided, plot on this axis. Otherwise create new figure.
    figsize : Tuple[float, float]
        Figure size (only used if ax is None)
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot histogram
    ax.hist(spectrum_calibrated, bins=120, range=(4.5, 9.2), 
            histtype='step', color='black', linewidth=1.5, label='Data (Calibrated)')
    
    # Shade derived ranges
    colors = plt.cm.Set3(np.linspace(0, 1, len(isotope_ranges)))
    
    for i, (name, range_obj) in enumerate(isotope_ranges.items()):
        ax.axvspan(range_obj.E_min, range_obj.E_max, alpha=0.3, 
                   color=colors[i], label=f'{name} range')
        
        # Mark fitted peak position using helper
        mark_peak_position(ax, range_obj.E_peak, name, color='red', y_fraction=0.95)
    
    ax.set(xlabel='Energy (MeV) - Calibrated', ylabel='Counts', 
           title='Derived Isotope Energy Ranges (Calibrated)')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
    
    return fig, ax

# ----------------------------------------------------------------------------
# COMPREHENSIVE CALIBRATION FIGURE
# ----------------------------------------------------------------------------

def plot_calibration_summary(spectrum, fit_results, peak_names, 
                               calibration_linear, calibration_quad,
                               spectrum_calibrated, isotope_ranges,
                               figsize=(16, 12)):
    """
    Create comprehensive 4-panel figure with calibration results.
    
    Layout:
    - Top row: Energy calibration curves (left) + residuals (right)
    - Bottom row: Preliminary fits in SCA with calibrated scale (left) + isotope ranges (right)
    
    Parameters
    ----------
    spectrum : SpectrumData
        Raw spectrum data (in SCA scale)
    fit_results : Dict[str, ModelResult]
        Fitted peak results
    peak_names : List[str]
        Names of peaks to plot
    calibration_linear : EnergyCalibration
        Linear calibration
    calibration_quad : EnergyCalibration
        Quadratic calibration
    spectrum_calibrated : np.ndarray
        Calibrated energy values (in MeV)
    isotope_ranges : Dict[str, IsotopeRange]
        Derived isotope ranges
    figsize : Tuple[float, float]
        Figure size
        
    Returns
    -------
    fig, axes : matplotlib figure and 2x2 array of axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Top row: Energy calibration
    plot_energy_calibration(calibration_linear, calibration_quad, 
                           ax1=axes[0, 0], ax2=axes[0, 1])
    
    # Bottom left: Preliminary fits with calibrated scale on top
    plot_preliminary_fits(spectrum, fit_results, peak_names,
                         calibration=calibration_linear,
                         ax=axes[1, 0])
    
    # Bottom right: Isotope ranges
    plot_isotope_ranges(spectrum_calibrated, isotope_ranges,
                       ax=axes[1, 1])
    
    plt.tight_layout()
    return fig, axes

# ----------------------------------------------------------------------------
# HIERARCHICAL FIT PLOTTING
# ----------------------------------------------------------------------------

def plot_hierarchical_fit(energies_calibrated, counts_calibrated, result_hierarchical, 
                         figsize=(14, 10), height_ratios=(2, 1)):
    """
    Plot hierarchical fit with deconvolved components and residuals.
    
    Parameters
    ----------
    energies_calibrated : np.ndarray
        Calibrated energy bin centers (MeV)
    counts_calibrated : np.ndarray
        Histogram counts
    result_hierarchical : ModelResult
        Hierarchical fit result from fit_full_spectrum_hierarchical
    figsize : Tuple[float, float]
        Figure size
    height_ratios : Tuple[int, int]
        Height ratios for (spectrum, residuals) subplots
        
    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                    gridspec_kw={'height_ratios': height_ratios})
    
    # Top panel: Full spectrum with components
    ax1.step(energies_calibrated, counts_calibrated, where='mid', 
             color='black', alpha=0.6, linewidth=1.5, label='Data')
    ax1.plot(energies_calibrated, result_hierarchical.best_fit * counts_calibrated.max(), 
             'r-', linewidth=2, label='Hierarchical Fit')
    
    # Plot all components
    components = result_hierarchical.eval_components(x=energies_calibrated)
    colors = plt.cm.tab10(np.arange(len(components)))
    
    for i, (comp_name, comp_values) in enumerate(components.items()):
        label = comp_name.rstrip('_')
        comp_scaled = comp_values * counts_calibrated.max()
        linestyle = ':' if '_sat_' in comp_name else '--'
        ax1.plot(energies_calibrated, comp_scaled, linestyle=linestyle, 
                 color=colors[i], alpha=0.7, linewidth=1.5, label=label)
    
    ax1.set(ylabel='Counts', title='Hierarchical Fit: 9 Peaks (Complete Th-232 chain)')
    ax1.legend(loc='upper right', fontsize=12, ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals
    plot_residuals(ax2, energies_calibrated, counts_calibrated, 
                   result_hierarchical.best_fit * counts_calibrated.max())
    ax2.set_xlabel('Energy [MeV]')
    
    plt.tight_layout()
    return fig, (ax1, ax2)
