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

from RaTag.alphas.spectrum_fitting import (
    SpectrumData, EnergyCalibration, IsotopeRange, v_crystalball, 
    _extract_fit_params_calibrated
)


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


# ============================================================================
# OVERLAP RESOLUTION VISUALIZATION
# ============================================================================

def _sample_overlap_events(spectrum_calibrated: np.ndarray,
                           range1_orig: 'IsotopeRange',
                           range2_orig: 'IsotopeRange',
                           n_samples: int = 15) -> np.ndarray:
    """Sample events from overlap region (or near boundary if no overlap)."""
    if range1_orig.E_max > range2_orig.E_min:
        # True overlap: sample from overlapping interval
        sample_min = range2_orig.E_min
        sample_max = range1_orig.E_max
    else:
        # Gap: sample near boundary (±0.15 MeV)
        gap_center = (range1_orig.E_max + range2_orig.E_min) / 2
        sample_min = gap_center - 0.15
        sample_max = gap_center + 0.15
    
    overlap_events = spectrum_calibrated[
        (spectrum_calibrated >= sample_min) & (spectrum_calibrated <= sample_max)
    ]
    
    if len(overlap_events) > n_samples:
        return np.random.choice(overlap_events, n_samples, replace=False)
    return overlap_events


def _plot_probability_landscape(ax: plt.Axes, E_grid: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                                E_boundary: float, iso1: str, iso2: str):
    """Plot Panel A: Probability landscape with color gradient."""
    # Likelihood ratio
    likelihood_ratio = P1 / (P1 + P2 + 1e-12)
    
    # Color gradient background
    for i in range(len(E_grid)-1):
        color = plt.cm.RdYlBu_r(likelihood_ratio[i])
        ax.axvspan(E_grid[i], E_grid[i+1], color=color, alpha=0.3, linewidth=0)
    
    # Plot PDFs
    ax.plot(E_grid, P1, 'r-', linewidth=2, label=f'{iso1} PDF', alpha=0.8)
    ax.plot(E_grid, P2, 'b-', linewidth=2, label=f'{iso2} PDF', alpha=0.8)
    ax.axvline(E_boundary, color='black', linestyle='--', linewidth=2.5,
              label=f'Boundary ({E_boundary:.3f} MeV)')
    
    # Twin axis for likelihood ratio
    ax_twin = ax.twinx()
    ax_twin.plot(E_grid, likelihood_ratio, 'k:', linewidth=1.5, alpha=0.5, label='Likelihood Ratio')
    ax_twin.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax_twin.set(ylabel='P(iso1) / [P(iso1) + P(iso2)]', ylim=(-0.05, 1.05))
    
    ax.set(xlabel='Energy (MeV)', ylabel='Probability Density',
           title=f'Panel A: Probability Landscape\n{iso1} vs {iso2}')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_event_distribution(ax: plt.Axes, spectrum_calibrated: np.ndarray,
                             E_min: float, E_max: float, E_boundary: float,
                             isotope_ranges_windowed: Dict[str, 'IsotopeRange'],
                             isotope_ranges_resolved: Dict[str, 'IsotopeRange'],
                             iso1: str, iso2: str):
    """Plot Panel B: Event distribution with original vs resolved ranges."""
    # Histogram
    hist, bins = np.histogram(spectrum_calibrated, bins=150, range=(E_min, E_max))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.step(bin_centers, hist, where='mid', color='black', alpha=0.7, linewidth=1.5, label='Events')
    
    # Original ranges (with overlap)
    range1_orig = isotope_ranges_windowed[iso1]
    range2_orig = isotope_ranges_windowed[iso2]
    ax.axvspan(range1_orig.E_min, range1_orig.E_max, alpha=0.15, color='red',
              label=f'{iso1} (original)')
    ax.axvspan(range2_orig.E_min, range2_orig.E_max, alpha=0.15, color='blue',
              label=f'{iso2} (original)')
    
    # Resolved boundaries
    range1_res = isotope_ranges_resolved[iso1]
    range2_res = isotope_ranges_resolved[iso2]
    ax.axvline(range1_res.E_max, color='red', linestyle='--', linewidth=2, label=f'{iso1} boundary')
    ax.axvline(range2_res.E_min, color='blue', linestyle='--', linewidth=2, label=f'{iso2} boundary')
    ax.axvline(E_boundary, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
    
    ax.set(xlabel='Energy (MeV)', ylabel='Counts',
           title='Panel B: Event Distribution\nOriginal (overlap) vs Resolved')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_sample_diagnostics(ax: plt.Axes, sample_events: np.ndarray, E_boundary: float,
                             x0_1_cal: float, sigma_1_cal: float, beta_1: float, m_1: float,
                             x0_2_cal: float, sigma_2_cal: float, beta_2: float, m_2: float,
                             iso1: str, iso2: str):
    """Plot Panel C: Sample event diagnostics with likelihood ratios."""
    if len(sample_events) == 0:
        ax.text(0.5, 0.5, 'No events in overlap region\n(expand range or reduce n_sigma)',
               ha='center', va='center', fontsize=12, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Compute likelihood ratios
        sample_P1 = v_crystalball(sample_events, N=1.0, beta=beta_1, m=m_1,
                                 x0=x0_1_cal, sigma=sigma_1_cal)
        sample_P2 = v_crystalball(sample_events, N=1.0, beta=beta_2, m=m_2,
                                 x0=x0_2_cal, sigma=sigma_2_cal)
        sample_ratios = sample_P1 / (sample_P1 + sample_P2 + 1e-12)
        
        # Assign and plot
        assignments = np.where(sample_events < E_boundary, iso1, iso2)
        colors = ['red' if a == iso1 else 'blue' for a in assignments]
        
        for i, (E, ratio, color) in enumerate(zip(sample_events, sample_ratios, colors)):
            ax.scatter(E, ratio, s=100, c=color, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.text(E, ratio + 0.03, str(i+1), fontsize=8, ha='center')
    
    # Decision boundary
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.axvline(E_boundary, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set(xlabel='Energy (MeV)', ylabel='Likelihood Ratio P(iso1)',
           title=f'Panel C: Sample Events (n={len(sample_events)})\nColor = Assignment',
           ylim=(-0.05, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)


def plot_overlap_resolution(spectrum_calibrated: np.ndarray,
                            fit_results: Dict[str, ModelResult],
                            calibration: 'EnergyCalibration',
                            isotope_ranges_windowed: Dict[str, IsotopeRange],
                            isotope_ranges_resolved: Dict[str, IsotopeRange],
                            overlap_pairs: List[Tuple[str, str]],
                            figsize: Tuple[float, float] = (16, 5)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize overlap resolution using likelihood crossover method (3-panel figure).
    
    Returns:
        fig, axes : matplotlib figure and list of 3 axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for iso1, iso2 in overlap_pairs:
        if iso1 not in fit_results or iso2 not in fit_results:
            continue
        
        # Extract calibrated parameters for both isotopes
        x0_1_cal, sigma_1_cal, beta_1, m_1 = _extract_fit_params_calibrated(fit_results[iso1], calibration)
        x0_2_cal, sigma_2_cal, beta_2, m_2 = _extract_fit_params_calibrated(fit_results[iso2], calibration)
        
        # Energy grid for plotting (±3σ around peaks)
        E_min = x0_1_cal - 3*sigma_1_cal
        E_max = x0_2_cal + 3*sigma_2_cal
        E_grid = np.linspace(E_min, E_max, 1000)
        
        # Evaluate PDFs
        P1 = v_crystalball(E_grid, N=1.0, beta=beta_1, m=m_1, x0=x0_1_cal, sigma=sigma_1_cal)
        P2 = v_crystalball(E_grid, N=1.0, beta=beta_2, m=m_2, x0=x0_2_cal, sigma=sigma_2_cal)
        
        # Get boundary from resolved ranges
        E_boundary = isotope_ranges_resolved[iso1].E_max
        
        # Plot three panels
        _plot_probability_landscape(axes[0], E_grid, P1, P2, E_boundary, iso1, iso2)
        
        _plot_event_distribution(axes[1], spectrum_calibrated, E_min, E_max, E_boundary,
                                isotope_ranges_windowed, isotope_ranges_resolved, iso1, iso2)
        
        # Sample events and plot diagnostics
        range1_orig = isotope_ranges_windowed[iso1]
        range2_orig = isotope_ranges_windowed[iso2]
        sample_events = _sample_overlap_events(spectrum_calibrated, range1_orig, range2_orig)
        
        _plot_sample_diagnostics(axes[2], sample_events, E_boundary,
                                x0_1_cal, sigma_1_cal, beta_1, m_1,
                                x0_2_cal, sigma_2_cal, beta_2, m_2,
                                iso1, iso2)
    
    plt.tight_layout()
    return fig, axes
