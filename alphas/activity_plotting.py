"""
Visualization functions for activity analysis.

This module provides plotting utilities for:
1. Activity vs time (with error bars)
2. Decay curve fits (exponential model)
3. Count rate evolution
4. Multi-isotope activity comparisons
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta

from RaTag.alphas.activity_analysis import (
    ActivityMeasurement,
    DecayFitResult
)


# ============================================================================
# ACTIVITY TIME SERIES PLOTTING
# ============================================================================

def plot_activity_timeseries(measurements: List[ActivityMeasurement],
                             decay_fit: Optional[DecayFitResult] = None,
                             ax: Optional[Axes] = None,
                             figsize: Tuple[float, float] = (12, 6),
                             use_relative_time: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot activity vs time with error bars and optional decay fit.
    
    Args:
        measurements: List of ActivityMeasurement
        decay_fit: Optional DecayFitResult to overlay fitted curve
        ax: Matplotlib axis (if None, create new figure)
        figsize: Figure size (only used if ax is None)
        use_relative_time: If True, plot hours since first measurement.
                          If False, use absolute datetime.
        
    Returns:
        (fig, ax) tuple
        
    Example:
        >>> fig, ax = plot_activity_timeseries(ra224_measurements, decay_fit)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract data
    if use_relative_time:
        t0 = measurements[0].timestamp
        t = np.array([(m.timestamp - t0) / 3600 for m in measurements])  # Hours since start
        xlabel = 'Time since first measurement [hours]'
    else:
        t = np.array([m.acquisition_time for m in measurements])
        xlabel = 'Date'
    
    A = np.array([m.activity for m in measurements])
    A_err = np.array([m.activity_err for m in measurements])
    
    # Plot measurements with error bars
    isotope = measurements[0].isotope
    ax.errorbar(t, A, yerr=A_err, fmt='o', markersize=8, capsize=5,
                color='steelblue', ecolor='steelblue', label=f'{isotope} data',
                linewidth=2, markeredgewidth=1.5, markeredgecolor='navy')
    
    # Overlay decay fit if provided
    if decay_fit is not None:
        if use_relative_time:
            t_fit = np.linspace(t.min(), t.max(), 200)
            A_fit = decay_fit.A0 * np.exp(-decay_fit.lambda_decay * t_fit)
        else:
            # Convert datetime to hours for fit evaluation
            t_hours = np.array([(m.timestamp - measurements[0].timestamp) / 3600 for m in measurements])
            t_fit_hours = np.linspace(t_hours.min(), t_hours.max(), 200)
            t_fit = [measurements[0].acquisition_time + 
                    timedelta(hours=h) for h in t_fit_hours]
            A_fit = decay_fit.A0 * np.exp(-decay_fit.lambda_decay * t_fit_hours)
        
        ax.plot(t_fit, A_fit, 'r-', linewidth=2.5, alpha=0.8,
               label=f'Fit: T½ = {decay_fit.half_life:.1f} ± {decay_fit.half_life_err:.1f} h\n' +
                     f'Literature: T½ = {decay_fit.half_life_literature:.1f} h\n' +
                     f'χ²ᵣ = {decay_fit.chi2_reduced:.2f}')
    
    ax.set(xlabel=xlabel, ylabel='Activity [Bq]',
           title=f'{isotope} Activity vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Format x-axis for datetime
    if not use_relative_time:
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
        fig.autofmt_xdate()
    
    return fig, ax


def plot_count_rate_timeseries(measurements: List[ActivityMeasurement],
                               ax: Optional[Axes] = None,
                               figsize: Tuple[float, float] = (12, 6)) -> Tuple[Figure, Axes]:
    """
    Plot count rate vs time (alternative to activity, no efficiency correction).
    
    Useful for relative activity comparisons without knowing absolute efficiency.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract data
    t0 = measurements[0].timestamp
    t = np.array([(m.timestamp - t0) / 3600 for m in measurements])  # Hours
    
    rate = np.array([m.count_rate for m in measurements])
    rate_err = np.array([m.count_rate_err for m in measurements])
    
    isotope = measurements[0].isotope
    ax.errorbar(t, rate, yerr=rate_err, fmt='s', markersize=7, capsize=4,
                color='forestgreen', ecolor='forestgreen', label=f'{isotope}',
                linewidth=1.5, markeredgewidth=1.5, markeredgecolor='darkgreen')
    
    ax.set(xlabel='Time since first measurement [hours]',
           ylabel='Count rate [counts/hour]',
           title=f'{isotope} Count Rate vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return fig, ax


def plot_multi_isotope_activity(measurements_dict: dict,
                                figsize: Tuple[float, float] = (14, 7)) -> Tuple[Figure, Axes]:
    """
    Plot activity for multiple isotopes on same axes (normalized to initial).
    
    Useful for comparing decay rates and checking secular equilibrium.
    
    Args:
        measurements_dict: Dict of {isotope_name: List[ActivityMeasurement]}
        
    Example:
        >>> measurements = {
        ...     'Ra224': ra224_measurements,
        ...     'Rn220': rn220_measurements,
        ...     'Po216': po216_measurements,
        ... }
        >>> fig, ax = plot_multi_isotope_activity(measurements)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = cm.get_cmap('tab10')(np.linspace(0, 0.8, len(measurements_dict)))
    
    for i, (isotope, measurements) in enumerate(measurements_dict.items()):
        t0 = measurements[0].timestamp
        t = np.array([(m.timestamp - t0) / 3600 for m in measurements])
        
        A = np.array([m.activity for m in measurements])
        A_err = np.array([m.activity_err for m in measurements])
        
        # Normalize to initial activity
        A_norm = A / A[0]
        A_norm_err = A_err / A[0]
        
        ax.errorbar(t, A_norm, yerr=A_norm_err, fmt='o-', markersize=6,
                   capsize=4, color=colors[i], label=isotope, linewidth=2)
    
    ax.set(xlabel='Time since first measurement [hours]',
           ylabel='Normalized activity (A/A₀)',
           title='Multi-Isotope Activity Evolution (Normalized)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    return fig, ax


def plot_activity_diagnostic(measurements: List[ActivityMeasurement],
                             decay_fit: DecayFitResult,
                             figsize: Tuple[float, float] = (16, 10)) -> Tuple[Figure, Tuple[Axes, Axes, Axes, Axes]]:
    """
    Comprehensive 4-panel diagnostic figure for activity analysis.
    
    Layout:
    - Top left: Activity vs time with fit
    - Top right: Residuals (data - fit)
    - Bottom left: Count rate vs time
    - Bottom right: Fit statistics and parameters
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Activity with fit
    ax1 = fig.add_subplot(gs[0, 0])
    plot_activity_timeseries(measurements, decay_fit, ax=ax1)
    
    # Top right: Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    t0 = measurements[0].timestamp
    t = np.array([(m.timestamp - t0) / 3600 for m in measurements])
    A = np.array([m.activity for m in measurements])
    A_err = np.array([m.activity_err for m in measurements])
    A_fit = decay_fit.A0 * np.exp(-decay_fit.lambda_decay * t)
    residuals = A - A_fit
    
    ax2.errorbar(t, residuals, yerr=A_err, fmt='o', markersize=6, capsize=4,
                color='black', ecolor='gray')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set(xlabel='Time [hours]', ylabel='Residuals [Bq]',
           title='Fit Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Count rate
    ax3 = fig.add_subplot(gs[1, 0])
    plot_count_rate_timeseries(measurements, ax=ax3)
    
    # Bottom right: Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Format statistics
    stats_text = f"""
    Decay Fit Statistics
    ═══════════════════════════════════
    
    Isotope: {decay_fit.isotope}
    
    Initial Activity:
      A₀ = {decay_fit.A0:.2f} ± {decay_fit.A0_err:.2f} Bq
    
    Decay Constant:
      λ = {decay_fit.lambda_decay:.6f} ± {decay_fit.lambda_err:.6f} h⁻¹
    
    Half-Life:
      T½ (fitted)     = {decay_fit.half_life:.2f} ± {decay_fit.half_life_err:.2f} h
      T½ (literature) = {decay_fit.half_life_literature:.2f} h
      Difference      = {decay_fit.half_life - decay_fit.half_life_literature:.2f} h
      
    Fit Quality:
      χ²ᵣ = {decay_fit.chi2_reduced:.3f}
      N_points = {len(measurements)}
    
    Time Range:
      Start: {measurements[0].acquisition_time.strftime('%Y-%m-%d %H:%M')}
      End:   {measurements[-1].acquisition_time.strftime('%Y-%m-%d %H:%M')}
      Duration: {(measurements[-1].timestamp - measurements[0].timestamp)/3600:.1f} hours
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    return fig, (ax1, ax2, ax3, ax4)
