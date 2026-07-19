import matplotlib.pyplot as plt # type: ignore
import time
from matplotlib.colors import LogNorm
from matplotlib.colors import LogNorm
import numpy as np 
import math
from dataclasses import replace
from contextlib import contextmanager
from typing import Optional, Tuple, Callable, Any
from pathlib import Path
import pandas as pd

from RaTag.io.file_ops import load_npz_arrays
from RaTag.core.datatypes import PMTWaveform, SetPmt, S2Areas, Run
from RaTag.core.dataIO import load_wfm, iter_waveforms
from RaTag.core.units import s_to_us, V_to_mV
from RaTag.core.paths import get_output_root
# --------------------------------
# Basic waveform plotter
# --------------------------------

def plot_waveform(wf: PMTWaveform, frame: Optional[int] = None, ax=None, title: str = "Waveform", color: str = "b"):
    """Plot a single waveform."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if wf.ff:
        if frame is not None:
            V = wf.v[frame, :]
        else:
            frame = np.random.randint(0, wf.nframes)
        V = wf.v[frame, :]
        t = wf.t
    else:
        t, V = wf.t, wf.v
    
    if wf.t[1] - wf.t[0] < 1e-7:  # Hardcoded threshold to distinguish s vs µs
        # print("Converting time to µs for better readability")
        t = s_to_us(t)  # convert to µs
        V = V_to_mV(V)  # convert to mV
    
    wf_index = Path(wf.source).name.replace(".wfm", "").replace("Wfm", "").split("_")[-1] # type: ignore
    title = f"{title}, File {wf_index}"
    if wf.ff:
        title += f", frame {frame} of {wf.nframes}"
    
    ax.set(title=title, xlabel="Time (µs)", ylabel="Signal (mV)")
    ax.plot(t, V, color=color, alpha=1)
    ax.set_xticks(np.arange(min(t), max(t), step=(max(t)-min(t))/10)) # type: ignore
    ax.grid(True) 
    return ax, V.max()

def plot_frames(set_pmt: SetPmt, n_frames: int = 8, start_frame: int = 0, **kwargs):
    """
    Plots a sequence of individual frames as subplots.
    Usage in Jupyter: plot_frames(set, n_frames=4, start_frame=12)
    """
    from RaTag.io.file_ops import iter_frames
    
    fig, axes = plt.subplots(n_frames, 1, figsize=(10, 4*n_frames), layout="constrained")
    if n_frames == 1: axes = [axes]
    
    # We leverage our new efficient generator!
    frame_gen = iter_frames(set_pmt, max_frames=n_frames, start_frame=start_frame)
    
    for i in range(n_frames):
        try:
            wf_frame = next(frame_gen)
            plot_waveform(wf_frame, ax=axes[i], **kwargs)
        except StopIteration:
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, "No more frames available", ha='center', fontsize=12)
            
    return fig, axes


def plot_random_frames(set_pmt: SetPmt, n_frames: int = 8, **kwargs):
    """
    Plots a random selection of frames from across the entire set.
    Ensures true randomness by picking random files and random frames within them.
    """
    from RaTag.io.file_ops import load_random_waveform
    
    fig, axes = plt.subplots(n_frames, 1, figsize=(10, 4*n_frames), layout="constrained")
    if n_frames == 1: axes = [axes]
    
    for i in range(n_frames):
        wf, frame_idx = load_random_waveform(set_pmt)
        plot_waveform(wf, frame=frame_idx, ax=axes[i], **kwargs)
            
    return fig, axes

# ------------------------------------------------
# Advanced waveform plotters (with S1/S2 window)
# ------------------------------------------------

def _get_metadata_kwargs(kwargs: dict, set_pmt: SetPmt):
    """Helper to get timing parameters from kwargs or set attributes."""
    time_keys = ["t_s1", "t_s1_std", "t_s2_start", "t_s2_start_std", "t_s2_end", "t_s2_end_std"]
    for key in time_keys:
        if key not in kwargs:
            kwargs[key] = getattr(set_pmt, key, None)

    for key in kwargs.keys():    
        if key not in time_keys:
            raise ValueError(f"Unknown parameter: {key}")

    return kwargs

def _plot_window_shading(ax: plt.Axes, kwargs: dict, key: str, y_max: float, color: str = 'blue'):
    """Helper to plot vertical lines and shaded std regions."""

    t_mean = kwargs.get(f"{key}")
    t_std = kwargs.get(f"{key}_std", 0)

    if t_mean is not None:
        ax.axvline(t_mean, color=color, linestyle='--',
                            lw=1.5, label='{} ± σ'.format(key.replace('t_', ' ')))


    if t_std is not None and t_std > 0:
        # print(f'y_max: {y_max:.2f} V') 
        ax.fill_betweenx([0, y_max], 
                         t_mean - t_std,
                         t_mean + t_std,
                         color=color, alpha=0.1)

def plot_set_windows(set_pmt: SetPmt, 
                     file_index: int = None, frame: int = None, # type: ignore
                     ax = None, color: str = "b", **kwargs) -> tuple:
    """
    Plot multiple waveforms with S1 and S2 timing markers.
    
    Args:
        set_pmt: SetPmt object
        file_index: index of file in the set to plot (if None, assigned randomly)
        frame: index of the frame in the FF file to plot (if None, assigned randomly)
        ax: Optional axes to plot on
        color: Waveform color
        **kwargs: Optional timing parameters:
            t_s1: Mean S1 time (µs)
            t_s1_std: Std dev of S1 time (µs)
            t_s2_start: Mean S2 start time (µs)
            t_s2_start_std: Std dev of S2 start time (µs)
            t_s2_end: Mean S2 end time (µs)
            t_s2_end_std: Std dev of S2 end time (µs)
        
    Returns:
        (fig, axes)
    """

    
    kwargs = _get_metadata_kwargs(kwargs, set_pmt) # get timing params
    
    if file_index is None:
        file_index = np.random.randint(0, len(set_pmt.filenames))
    
    fn = set_pmt.filenames[file_index]

    if ax == None:
        ax = plt.gca()

    wf = load_wfm(set_pmt.source_dir / fn)
    _, v_max = plot_waveform(wf, frame=frame, ax=ax, title=f"Gate {set_pmt.gate} V", color=color)

    _plot_window_shading(ax, kwargs, "t_s1", v_max, "green")
    _plot_window_shading(ax, kwargs, "t_s2_start", v_max, "red")
    _plot_window_shading(ax, kwargs, "t_s2_end", v_max, "purple")
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return ax

def plot_n_waveforms(set_pmt: SetPmt, n_waveforms: int, **kwargs) -> tuple:
    """
    Plot multiple waveforms with S1 and S2 timing markers.
    
    Args:
        set_pmt: SetPmt object
        n_waveforms: Number of waveforms to plot
        **kwargs: Optional timing parameters:
            t_s1_mean: Mean S1 time (µs)
            t_s1_std: Std dev of S1 time (µs)
            t_s2_start_mean: Mean S2 start time (µs)
            t_s2_start_std: Std dev of S2 start time (µs)
            t_s2_end_mean: Mean S2 end time (µs)
            t_s2_end_std: Std dev of S2 end time (µs)
        
    Returns:
        (fig, axes)
    """

    fig, axes = plt.subplots(n_waveforms, 1, figsize=(10, 4*n_waveforms))
    if n_waveforms == 1:
        axes = [axes]

    selected_files = np.random.choice(set_pmt.filenames, size=n_waveforms, replace=False)
    # print(selected_files)

    for ax, fn in zip(axes, selected_files):
        plot_set_windows(set_pmt, file_index=set_pmt.filenames.index(fn), ax=ax, **kwargs)
    
    return fig, axes


def plot_window_validation(ax: plt.Axes, 
                           wf: PMTWaveform, 
                           frame: Optional[int],
                           set_pmt: SetPmt,
                           color: str = "b") -> None:
    """Pure plotting function for timing validation. Zero I/O."""
    
    # We pass the title directly to plot_waveform, which smartly appends the File Index to it!
    title = f"{set_pmt.source_dir.name} | Gate {set_pmt.gate} V"
    _, v_max = plot_waveform(wf, frame=frame, ax=ax, title=title, color=color)

    _# Local helper for clean shading logic directly from SetPmt attributes
    def shade_window(key: str, fill_color: str):
        t_mean = getattr(set_pmt, key, None)
        t_std = getattr(set_pmt, f"{key}_std", 0)
        
        if t_mean is not None:
            label = f"{key.replace('t_', ' ')} ± σ"
            ax.axvline(t_mean, color=fill_color, linestyle='--', lw=1.5, label=label)
            
        if t_std and t_std > 0:
            ax.fill_betweenx([0, v_max], t_mean - t_std, t_mean + t_std, color=fill_color, alpha=0.1)

    shade_window("t_s1", "green")
    shade_window("t_s2_start", "red")
    shade_window("t_s2_end", "purple")
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

def plot_timing_errorbar(drift_fields: np.ndarray,
                         means: np.ndarray,
                         stds: np.ndarray,
                         label: str,
                         color: str,
                         marker: str,
                         ax: plt.Axes) -> None:
    """
    Plot single timing parameter vs drift field with error bars.
    
    Pure plotting function - minimal responsibility.
    
    Args:
        drift_fields: Drift field values (V/cm)
        means: Mean timing values (µs)
        stds: Standard deviations (µs)
        label: Legend label
        color: Line/marker color
        marker: Marker style ('o', 's', '^', etc.)
        ax: Matplotlib axes to plot on
    """
    ax.errorbar(drift_fields, means, yerr=stds,
                fmt=f'{marker}-', label=label, color=color, 
                capsize=5, markersize=8, linewidth=2)


def plot_timing_vs_drift_field(drift_fields: np.ndarray,
                                timing_data: dict[str, dict],
                                title: str = "Timing vs Drift Field") -> tuple:
    """
    Plot timing estimates as a function of drift field.
    
    Pure plotting function - iterates over timing parameters.
    
    Args:
        drift_fields: Array of drift field values (V/cm)
        timing_data: Dict mapping param names to {'mean': array, 'std': array}
                    Keys: 't_s1', 't_s2_start', 't_s2_end'
        title: Plot title
        
    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configuration for each timing parameter
    plot_config = [
        ('t_s1', 'S1 (prompt)', 'blue', 'o'),
        ('t_s2_start', 'S2 start (drift)', 'green', 's'),
        ('t_s2_end', 'S2 end', 'red', '^')
    ]
    
    # Plot each parameter (if data exists)
    for param_name, label, color, marker in plot_config:
        if param_name in timing_data:
            t_data = timing_data[param_name]
            if len(t_data['mean']) > 0:  # Check for non-empty t_data
                plot_timing_errorbar(drift_fields=drift_fields,
                                     means=t_data['mean'], stds=t_data['std'],
                                     label=label, color=color, marker=marker, ax=ax )
    
    # Formatting
    ax.set(xlabel='Drift Field (V/cm)', ylabel='Time (µs)', title=title)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, ax


# --------------------------------
# Histogram + Gaussian fit
# --------------------------------

def plot_hist_fit(s2: S2Areas, nbins=100, bin_cuts=(0, 5), ax=None):
    """
    Plot S2 area histogram with fit.
    
    Handles both old Gaussian fits and new Crystal Ball fits.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        ax.text(0.5, 0.5, "No data in range", ha='center', va='center')
        return fig, ax

    n, bins, patches = ax.hist(area_vec, bins=nbins, alpha=0.6, color='g', label="Data")
    ax.set_xlabel("S2 Area (mV·µs)")
    ax.set_ylabel("Counts")
    ax.set_title(f"S2 Area Histogram for Set {s2.source_dir.name}")
    ax.grid(True)

    if s2.fit_success and s2.fit_result:
        # Check if it's new format (dict) or old format (lmfit result)
        if isinstance(s2.fit_result, dict):
            # New format - use plot_s2_fit_result
            plt.close(fig)  # Close the simple plot
            fig, axes = plot_s2_fit_result(s2.fit_result, s2.areas, 
                                           set_name=s2.source_dir.name)
            return fig, axes
        else:
            # Old format - existing Gaussian/lmfit result plot
            x = np.linspace(bin_cuts[0], bin_cuts[1], 1000)
            y = s2.fit_result.eval(x=x)  # Use stored fit result to evaluate
            ax.plot(x, y, 'r-', label="Gaussian Fit")
            ax.axvline(s2.mean, color='b', ls='--', 
                      label=f"Mean: {s2.mean:.2f} ± {s2.ci95:.2f}")
            ax.legend()
    else:
        ax.text(0.5, 0.9, "Fit failed or not performed", 
                ha='center', va='center', transform=ax.transAxes)

    return fig, ax


def plot_s2_fit_result(result: dict, data: np.ndarray, set_name: str = '', 
                       figsize: tuple = (16, 5)):
    """
    Plot S2 area fit results with appropriate visualization based on method.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fit_s2_area_auto, fit_s2_simple_cb, or fit_s2_two_stage
    data : array-like
        Original S2 area data
    set_name : str, optional
        Name/identifier for the dataset (for plot title)
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    axes : array of matplotlib.axes.Axes
        The axes objects
        
    Notes
    -----
    For 'simple' method: creates single plot with data and fit
    For 'two_stage' method: creates two subplots showing background subtraction and signal fit
    """
    hist_data = result['histogram']
    
    if result['method'] == 'simple':
        # Single plot for simple method
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        
        ax.hist(data, bins=hist_data['bins'], alpha=0.5, color='blue', label='Data')
        
        x_smooth = np.linspace(hist_data['bins'][0], hist_data['bins'][-1], 500)
        fit_curve = result['result'].eval(x=x_smooth)
        ax.plot(x_smooth, fit_curve, 'r-', linewidth=2, 
                label=f"CB Fit (x₀={result['peak_position']:.2f})")
        
        ax.axvline(result['peak_position'], color='red', linestyle=':', alpha=0.7, 
                   label=f"Peak: {result['peak_position']:.2f} mV·µs")
        ax.set_xlabel('S2 Area (mV·µs)', fontsize=11)
        ax.set_ylabel('Counts', fontsize=11)
        ax.set_title(f"{set_name}\n{result['method']} method | χ²/dof = {result['redchi']:.2f}", 
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        axes = np.array([ax])
        
    else:  # two_stage
        # Two subplots for two-stage method
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Background subtraction
        ax1 = axes[0]
        ax1.hist(data, bins=hist_data['bins'], 
                 alpha=0.4, color='green', label='Original Data')
        bg_curve = result['result_bg'].eval(x=hist_data['bin_centers'])
        ax1.plot(hist_data['bin_centers'], bg_curve, 'b--', linewidth=2, 
                 label=f"Background (μ={result['bg_center']:.2f})")
        ax1.bar(hist_data['bin_centers'], hist_data['subtracted'], 
                width=np.diff(hist_data['bins'])[0], alpha=0.6, color='orange', 
                label='Subtracted')
        ax1.axvline(result['lower_bound'], color='gray', linestyle='--', 
                    alpha=0.7, label=f"Lower bound: {result['lower_bound']:.2f}")
        ax1.set_xlabel('S2 Area (mV·µs)', fontsize=11)
        ax1.set_ylabel('Counts', fontsize=11)
        ax1.set_title('Stage 1: Background Subtraction', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Right: Signal fit
        ax2 = axes[1]
        ax2.bar(hist_data['bin_centers'], hist_data['subtracted'], 
                width=np.diff(hist_data['bins'])[0], alpha=0.5, color='orange', 
                label='Subtracted Data')
        x_smooth = np.linspace(hist_data['bins'][0], hist_data['bins'][-1], 500)
        sig_curve = result['result_sig'].eval(x=x_smooth)
        ax2.plot(x_smooth, sig_curve, 'g-', linewidth=2, 
                 label=f"CB Fit (x₀={result['peak_position']:.2f})")
        ax2.axvline(result['peak_position'], color='green', linestyle=':', 
                    alpha=0.7)
        ax2.axvline(result['lower_bound'], color='gray', linestyle='--', 
                    alpha=0.7, label=f"Lower bound: {result['lower_bound']:.2f}")
        ax2.set_xlabel('S2 Area (mV·µs)', fontsize=11)
        ax2.set_ylabel('Counts', fontsize=11)
        ax2.set_title(f"Stage 2: Signal Fit\nχ²/dof = {result['redchi']:.2f}", 
                      fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(set_name, fontsize=13, y=1.02)
    
    plt.tight_layout()
    return fig, axes

def plot_s2_vs_drift(df: pd.DataFrame, 
                     run_id: str,
                     ylabel: str = "Mean S2 Area (mV·µs)",
                     title_suffix: str = "",
                     hue: str = None) -> tuple:
    """
    Plot S2 area vs drift field from DataFrame.
    
    Pure plotting function - no computation or normalization.
    
    Args:
        df: DataFrame with columns: drift_field, s2_mean, s2_ci95
            If hue is specified, also needs column matching hue name
        run_id: Run identifier for title
        ylabel: Y-axis label
        title_suffix: Optional suffix for title
        hue: Optional column name for grouping (e.g., 'isotope')
    
    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hue is None:
        # Single series plot
        ax.errorbar(df['drift_field'], df['s2_mean'], yerr=df['s2_ci95'],
                    fmt='o', capsize=5, markersize=8, linewidth=2, color='blue')
    else:
        # Multi-series plot (one per hue value)
        colors = {'Ra224': 'red', 'Rn220': 'blue', 'Po216': 'green', 
                  'Po212': 'orange', 'Th228': 'purple'}
        
        for group_value in df[hue].unique():
            df_group = df[df[hue] == group_value]
            color = colors.get(group_value, None)
            ax.errorbar(df_group['drift_field'], df_group['s2_mean'], 
                       yerr=df_group['s2_ci95'],
                       fmt='o', label=group_value, color=color,
                       capsize=3, markersize=6, alpha=0.8)
        
        ax.legend(loc='best', fontsize=10)
    
    ax.set(xlabel="Drift field (V/cm)", ylabel=ylabel,
           title=f"Run {run_id} — Mean S2 Area vs Drift Field{title_suffix}")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, ax

def plot_xray_histogram(areas: np.ndarray, run_id: str, nbins: int = 100, 
                        bin_cuts: tuple = (0.6, 20), fit_result=None, 
                        mean: float = None, ci95: float = None):
    """
    Plot combined X-ray area histogram with optional fit.
    
    Args:
        areas: X-ray S2 areas
        run_id: Run identifier
        nbins: Number of histogram bins
        bin_cuts: (min, max) range for histogram
        fit_result: Optional lmfit ModelResult
        mean: Optional fitted mean
        ci95: Optional 95% CI
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter areas
    area_vec = areas[(areas > bin_cuts[0]) & (areas < bin_cuts[1])]
    
    if len(area_vec) == 0:
        ax.text(0.5, 0.5, "No data in range", ha='center', va='center')
        return fig, ax
    
    # Plot histogram
    n, bins, patches = ax.hist(area_vec, bins=nbins, alpha=0.6, color='blue', label="X-ray Data")
    ax.set_xlabel("S2 Area (mV·µs)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Combined X-ray S2 Areas — Run {run_id}")
    ax.grid(True)
    
    # Plot fit if provided
    if fit_result is not None and mean is not None:
        x = np.linspace(bin_cuts[0], bin_cuts[1], 1000)
        y = fit_result.eval(x=x)
        ax.plot(x, y, 'r-', linewidth=2, label="Gaussian Fit")
        ax.axvline(mean, color='darkred', ls='--', linewidth=2,
                  label=f"Mean: {mean:.2f} ± {ci95:.2f}")
        ax.legend()
    
    return fig, ax


def plot_s2_diffusion_analysis(drift_times: np.ndarray,
                               sigma_obs_squared: np.ndarray,
                               speeds_drift: np.ndarray,
                               drift_fields: np.ndarray,
                               pressure: float,
                               figsize: tuple = (10, 10)) -> tuple:
    """
    Plot S2 duration variance vs drift parameters for diffusion analysis.
    
    Args:
        drift_times: Drift times (µs)
        sigma_obs_squared: Observed variance (µs²)
        speeds_drift: Drift speeds (mm/µs)
        drift_fields: Drift fields (V/cm)
        pressure: Gas pressure (bar)
        figsize: Figure size
        
    Returns:
        (fig, axes)
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: σ² vs t_drift
    axes[0].scatter(drift_times, sigma_obs_squared, s=50, alpha=0.7)
    axes[0].set(xlabel="Drift Time $t_d$ (µs)",
               ylabel="$\\sigma_{obs}^2$ (µs²)",
               title="S2 Duration Variance vs Drift Time")
    axes[0].grid(alpha=0.3)
    
    # Fit and overlay
    if len(drift_times) > 2:

        fit = np.polyfit(drift_times, sigma_obs_squared, 1)
        x_fit = np.linspace(drift_times.min(), drift_times.max(), 100)
        axes[0].plot(x_fit, fit[0] * x_fit + fit[1], 'r--', lw=2,
                    label=f'Linear fit: σ² = {fit[0]:.3f}·t + {fit[1]:.3f}')
        axes[0].legend()
    
    # Plot 2: σ² vs t_d/v_d²
    speeds_squared = speeds_drift ** 2
    axes[1].scatter(drift_times / speeds_squared, sigma_obs_squared, 
                   s=50, alpha=0.7, color='orange')
    axes[1].set(xlabel="$t_d / v_d^2$ (µs·mm⁻²)",
               ylabel="$\\sigma_{obs}^2$ (µs²)",
               title="Normalized by Drift Speed²")
    axes[1].grid(alpha=0.3)
    
    # Plot 3: σ² vs reduced drift field
    reduced_field = drift_fields / pressure
    axes[2].scatter(reduced_field, sigma_obs_squared, 
                   s=50, alpha=0.7, color='green')
    axes[2].set(xlabel="Reduced Drift Field (V·cm⁻¹·bar⁻¹)",
               ylabel="$\\sigma_{obs}^2$ (µs²)",
               title="S2 Variance vs Reduced Field")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_alpha_energy_spectrum(energies: np.ndarray,
                               title: str = 'Alpha Energy Spectrum',
                               nbins: int = 120,
                               energy_range: tuple = (4, 8),
                               ax: Optional[plt.Axes] = None,
                               normalize: bool = False) -> tuple:
    """
    Plot alpha energy spectrum histogram.
    
    Args:
        energies: Array of alpha energies [MeV]
        title: Plot title
        nbins: Number of histogram bins
        energy_range: (min, max) energy range [MeV]
        ax: Optional axes to plot on
        normalize: If True, normalize histogram to max bin = 1
    Returns:
        (fig, ax) tuple
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Normalization mode: scale max bin to 1
    if normalize:
        # compute histogram and plot normalized step (peak -> 1)
        n, bins = np.histogram(energies, bins=nbins, range=energy_range)
        maxc = n.max() if n.max() > 0 else 1
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.step(bin_centers, n.astype(float) / float(maxc), where='mid')
    
    else:
        n, bins, patches = ax.hist(energies, bins=nbins, range=energy_range, alpha=0.7, edgecolor='black')
    
    ax.set(xlabel='Energy [MeV]', ylabel='Counts' if not normalize else 'Normalized counts', title=title)
    ax.grid(True, alpha=0.3)

    fig = plt.gcf()
    fig.tight_layout()

    return fig, ax


def plot_time_histograms(times: np.ndarray,
                        title: str = "Time Distribution",
                        mean: Optional[float] = None,
                        std: Optional[float] = None,
                        xlabel: str = "Time (µs)",
                        color: str = 'blue',
                        ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot timing histogram with mean and std markers.
    
    Simple histogram plotter that can be used standalone or as subplot.
    
    Args:
        times: Timing array to plot
        title: Plot title
        mean: Mean/mode value (for vertical line)
        std: Standard deviation (for shaded region)
        xlabel: X-axis label
        color: Fill color for std region
        ax: Optional axes to plot on (for subplots)
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Plot histogram
    n, bins, _ = ax.hist(times, bins=50, alpha=0.7, color=color)
    
    # Add mean line and std shading
    if mean is not None:
        ax.axvline(mean, color='red', linestyle='--', label=f'Mode: {mean:.2f} µs')
        
        if std is not None:
            ax.fill_between((mean - std, mean + std), 0, max(n),
                           color=color, alpha=0.2,
                           label=f'± σ: {std:.2f} µs')
    
    ax.set(xlabel=xlabel, ylabel='Counts', title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig

# --------------------------------------------
# -- Grouped histograms for isotope results
# --------------------------------------------

def _compute_histogram_range(data: pd.Series, percentile: float = 95.0) -> tuple:
    """
    Compute auto-range for histogram based on percentile.
    
    Parameters
    ----------
    data : pd.Series
        Data to compute range for
    percentile : float, optional
        Percentile to use as upper limit (default: 95.0)
        
    Returns
    -------
    tuple of (lower, upper) or None if no data
    """
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return None
    
    upper_limit = np.percentile(clean_data, percentile)
    mean_val = np.mean(clean_data)
    median_val = np.median(clean_data)
    
    # print(f"Histogram range: (0, {upper_limit:.2f}) [{percentile}th percentile] "
    #       f"(mean={mean_val:.2f}, median={median_val:.2f})")
    
    return (0, upper_limit)


def _get_fit_curve(fit_result: dict) -> tuple:
    """
    Extract fit curve from result dict (handles both simple and two_stage methods).
    
    Parameters
    ----------
    fit_result : dict
        Fit result from fit_multiiso_s2
        
    Returns
    -------
    tuple of (x_smooth, fit_curve) or (None, None) if no histogram data
    """
    hist_data = fit_result.get('histogram', {})
    bin_centers = hist_data.get('bin_centers', np.array([]))
    
    if len(bin_centers) == 0:
        return None, None
    
    # Create smooth x-axis
    x_smooth = np.linspace(bin_centers[0], bin_centers[-1], 500)
    
    # Get appropriate result based on method
    if fit_result.get('method') == 'two_stage':
        fit_curve = fit_result['result_sig'].eval(x=x_smooth)
    else:
        fit_curve = fit_result['result'].eval(x=x_smooth)
    
    return x_smooth, fit_curve


def _plot_isotope_histogram(ax: plt.Axes,
                            data: np.ndarray,
                            bins: int,
                            hist_range: tuple,
                            isotope: str,
                            column: str,
                            fit_result: dict = None) -> None:
    """
    Plot histogram for a single isotope with optional fit overlay.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    data : np.ndarray
        Data values to histogram
    bins : int
        Number of bins
    hist_range : tuple
        (min, max) range for histogram
    isotope : str
        Isotope name for title
    column : str
        Column name for title
    fit_result : dict, optional
        Fit result from fit_multiiso_s2 (if None, plots histogram only)
    """
    # Plot histogram
    ax.hist(data, bins=bins, range=hist_range, alpha=0.6, color='blue', label='Data')
    ax.set_title(f"{isotope} – {column}", fontsize=10)
    
    # Early return if no fit
    if fit_result is None:
        return
    
    # Get and plot fit curve
    x_smooth, fit_curve = _get_fit_curve(fit_result)
    if x_smooth is None:
        return
    
    ax.plot(x_smooth, fit_curve, 'r-', linewidth=2,
           label=f"Fit: μ={fit_result['peak_position']:.2f}")
    ax.axvline(fit_result['peak_position'], color='red', 
              linestyle=':', alpha=0.7)
    ax.legend(fontsize=8)


def plot_grouped_histograms(df: pd.DataFrame,
                            value_columns: list[str],
                            bins: int = 100, 
                            figsize=(10, 4),
                            fit_results: dict = None):
    """
    Plot grouped histograms for each isotope and each value column.

    Parameters
    ----------
    df : DataFrame
        Must contain 'isotope' and columns in value_columns.
    value_columns : list[str]
        Columns to plot (one subplot per column).
    bins : int
        Histogram bins.
    figsize : tuple
        Figure size.
    fit_results : dict, optional
        Dictionary of {isotope: fit_result_dict} from fit_multiiso_s2.
        If provided, will overlay fit curves on histograms.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with subplots for all value columns
    """
    isotopes = sorted(df["isotope"].unique())
    n_isotopes = len(isotopes)
    n_cols = len(value_columns)
    
    # Create figure with grid: rows = isotopes, columns = value_columns
    fig, axes = plt.subplots(n_isotopes, n_cols, 
                             figsize=(figsize[0] * n_cols, figsize[1] * n_isotopes),
                             sharex='col', squeeze=False)

    for j, col in enumerate(value_columns):
        # Determine auto-range for this column
        print(f"Column '{col}':")
        hist_range = _compute_histogram_range(df[col])
        
        for i, iso in enumerate(isotopes):
            # Get data and fit result for this isotope
            vals = df[df["isotope"] == iso][col].dropna().values
            fit_result = fit_results.get(iso) if fit_results else None
            
            # Plot histogram with optional fit
            _plot_isotope_histogram(ax=axes[i, j], data=vals,
                                    bins=bins, hist_range=hist_range, 
                                    isotope=iso, column=col,
                                    fit_result=fit_result)
        
        # Add x-label to bottom row (after loop)
        axes[n_isotopes - 1, j].set_xlabel(col)

    fig.tight_layout()
    return fig

# --------------------------------
# -- Combined timing histograms
# --------------------------------

def plot_timing_histograms(ax: plt.Axes, set_name: str, arrays: dict, bins: int = 100) -> None:
    """Pure plotter. Accepts the arrays dict directly. Zero I/O."""
    if not arrays:
        ax.text(0.5, 0.5, "No Timing Data", ha='center', va='center')
        ax.set_title(set_name)
        return
    
    signals_config = [
        ("t_s1", "tab:blue", "S1 Times"),
        ("t_s2_start", "tab:orange", "S2 Start Times"),
        ("t_s2_end", "tab:green", "S2 End Times")
    ]

    for key, color, label in signals_config:
        arr = arrays.get(key, np.array([]))
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) > 0:
            ax.hist(arr_clean, bins=bins, alpha=0.7, color=color, label=label)

    ax.set(title=f"Timing Distributions - {set_name}", xlabel="Time (µs)", ylabel="Counts")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# --------------------------------
# -- Run-level dashboards
# --------------------------------
def _get_grid_dims(n_items: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_items))
    rows = math.ceil(n_items / cols) if cols > 0 else 1
    return rows, cols


@contextmanager
def catch_plot_errors(ax: plt.Axes, title: str):
    """Context manager to gracefully catch and display plotting errors on the axis."""
    try:
        yield
    except Exception as e:
        ax.clear()
        ax.text(0.5, 0.5, f"Plot Failed\n{e}", ha='center', va='center', wrap=True)
        ax.set_title(title)

def build_fig_grid(run: Run, title: str = None) -> tuple[plt.Figure, list[tuple[SetPmt, plt.Axes]]]:
    """Creates a Matplotlib grid and returns the Figure + paired iterable cells."""
    rows, cols = _get_grid_dims(len(run.sets))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), layout="constrained")
    axes = np.atleast_2d(axes)
    
    if title: fig.suptitle(title, fontsize=16)

    grid_cells = []
    for idx, set_pmt in enumerate(run.sets):
        r, c = divmod(idx, cols)
        grid_cells.append((set_pmt, axes[r, c]))

    for idx in range(len(run.sets), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    return fig, grid_cells

def plot_run_timing_vs_field(run: Run) -> plt.Figure:
    """
    Extracts timing attributes from the run and plots them vs drift field.
    Returns None if no valid data is found.
    """
    param_names = ['t_s1', 't_s2_start', 't_s2_end']
    
    # 1. Filter to sets that actually have the calculated physics
    valid_sets = [s for s in run.sets if all(getattr(s, p, None) is not None for p in param_names)]
    
    if not valid_sets:
        return None
        
    # 2. Extract Data cleanly using list comprehensions
    drift_fields = np.array([s.drift_field for s in valid_sets])
    
    timing_data = {}
    for p in param_names:
        timing_data[p] = {
            'mean': np.array([getattr(s, p) for s in valid_sets]),
            'std': np.array([getattr(s, f"{p}_std", 0.0) for s in valid_sets])
        }
        
    fig, _ = plot_timing_vs_drift_field(
        drift_fields=drift_fields,
        timing_data=timing_data,
        title=f"Timing vs Drift Field - {run.run_id}"
    )
    
    return fig



def plot_s2areas_summary(ax: plt.Axes, 
                    set_name: str, 
                    s2_areas: S2Areas,
                    bin_cuts: tuple = (0, 15),
                    fit_model: Optional[Any] = None,
                    lower_bound: Optional[float] = None,
                    color: str = 'orange') -> None:
    """Plot S2 area histogram with optional fit overlay. Handles missing data gracefully."""
    
    # 1. Plot raw data
    ax.hist(s2_areas.areas, bins=100, range=bin_cuts, alpha=0.5, color=color, label='Data')
    
    # 2. Overlay fit
    if fit_model is not None:
        x_smooth = np.linspace(bin_cuts[0], bin_cuts[1], 500)
        y_fit = fit_model.eval(x=x_smooth)
        fit_mean = fit_model.params['sig_x0'].value
        ax.plot(x_smooth, y_fit, 'g-', lw=2, label=f"Fit (μ={fit_mean:.2f})")
        ax.axvline(fit_mean, color='green', linestyle=':', alpha=0.7)
        ax.set(ylim=(0, np.max(fit_model.best_fit)*1.75))
    if lower_bound is not None:
        ax.axvline(lower_bound, color='red', linestyle='--', alpha=0.7, label=f"Lower Fit Bound: {lower_bound:.2f}")
    # 3. Formatting
    ax.set(title=set_name, xlabel='S2 Area (mV·µs)', ylabel='Counts')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_run_s2_vs_field(run: Run) -> plt.Figure:
    """Extracts S2 metadata from the run and plots it vs drift field."""
    valid_sets = [s for s in run.sets if s.area_s2_fit_success]
    if not valid_sets:
        return None
        
    import pandas as pd
    df = pd.DataFrame([{
        'drift_field': s.drift_field,
        's2_mean': s.area_s2_mean,
        's2_ci95': s.area_s2_ci95
    } for s in valid_sets])
    
    fig, _ = plot_s2_vs_drift(df, run.run_id)
    return fig

def plot_xray_candidate(ax: plt.Axes, 
                        wf: PMTWaveform, 
                        frame: Optional[int], 
                        t_s1: float,
                        s2_start: float,
                        is_accepted: bool) -> None:
    """100% Pure plotter for an X-ray candidate waveform."""
    title_prefix = "✓ Accepted" if is_accepted else "✗ Rejected"
    color = "green" if is_accepted else "red"
    
    # Use standard plotter
    _, v_max = plot_waveform(wf, frame=frame, ax=ax, title=title_prefix, color=color)
    
    ax.axvline(t_s1, color='blue', linestyle='--', label='S1')
    ax.axvline(s2_start, color='purple', linestyle='--', label='S2 Start')
    ax.fill_betweenx([0, v_max], t_s1, s2_start, color='orange', alpha=0.1, label='Search Perimeter')
    
    ax.legend(fontsize=8, loc='upper right')


def plot_xray_validation(accepted_wfs: list,
                         rejected_wfs: list,
                         t_s1: float,
                         s2_start: float,
                         title: str = "X-ray Classification Validation"):
    """
    Pure plotting function for the 4x2 validation dashboard.
    Expects lists of (wf, frame) tuples and explicit set-level timing constants.
    """
    n_frames = max(len(accepted_wfs), len(rejected_wfs))
    
    if n_frames == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        return fig
    
    fig, axes = plt.subplots(n_frames, 2, figsize=(14, 3.5 * n_frames))
    if n_frames == 1:
        axes = axes.reshape(1, -1) 
    
    fig.suptitle(title, fontsize=14, y=0.995)
    
    for i in range(n_frames):
        # Left column (Accepted)
        ax_left = axes[i, 0]
        if i < len(accepted_wfs):
            wf, frame = accepted_wfs[i]
            plot_xray_candidate(ax_left, wf, frame, t_s1, s2_start, is_accepted=True)
        else:
            ax_left.axis('off')
            
        # Right column (Rejected)
        ax_right = axes[i, 1]
        if i < len(rejected_wfs):
            wf, frame = rejected_wfs[i]
            plot_xray_candidate(ax_right, wf, frame, t_s1, s2_start, is_accepted=False)
        else:
            ax_right.axis('off')
    
    plt.tight_layout()
    return fig

# --------------------------------
# -- Deprecated functions
# --------------------------------


def plot_waveform_with_cuts(wf: PMTWaveform, set_pmt: SetPmt,
                            width_s2: float):
    t, V = wf.t, wf.v
    t_s1 = set_pmt.t_s1
    t_drift = set_pmt.time_drift / 1e6 # convert us to s
    t_end = wf.t[-1]

    drift_window = (t_s1, t_s1 + t_drift)
    s2_window = (drift_window[1], drift_window[1] + width_s2)
    post_s2_window = (s2_window[1], t_end)

    plt.plot(t, V)
    # wf.plot()
    plt.axvline(drift_window[0], color="k", label="S1")
    plt.axvline(drift_window[1], color="m", label="S2 start")
    plt.axvline(s2_window[1], color="r", label="S2 end")
    plt.legend()


def plot_winS2_wf(wf: PMTWaveform, t_s1: float, time_drift: float, width_s2: float, ts2_tol: float = 0, ax=None):
    """Plot waveform with S1 and S2 window markers.
    For FastFrame waveforms, plots the average of all frames.
    
    Args:
        wf: PMTWaveform to plot.
        t_s1: S1 time in µs.
        time_drift: Drift time in µs.
        width_s2: Width of S2 window in µs.
        ts2_tol: Optional tolerance to add to S2 start time in µs.
        ax: Optional matplotlib Axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if wf.ff:
        # Average all frames for FastFrame
        V = wf.v.mean(axis=0)
        t = wf.t
    else:
        t, V = wf.t, wf.v
    t = s_to_us(t)  # convert to µs
    V = V_to_mV(V)  # convert to mV
    
    wf_index = Path(wf.source).name.replace(".wfm", "").replace("Wfm", "")
    title = f"Waveform {wf_index}"
    if wf.ff:
        title += f" (Average of {wf.nframes} frames)"
    
    ax.set(title=title, xlabel="Time (µs)", ylabel="Signal (mV)")
    ax.plot(t, V)

    s2_start = t_s1 + time_drift + ts2_tol
    s2_end = s2_start + width_s2
    ax.axvline(t_s1, color="k", ls="--", label="S1", lw=0.5, zorder=-1)
    ax.axvline(s2_start, color="m", ls="--", lw=0.5, zorder=-1)
    ax.axvline(s2_end, color="r", ls="--", lw=0.5, zorder=-1)
    ax.fill_betweenx(ax.get_ylim(), s2_start, s2_end, color='m', alpha=0.3, label="S2 window")
    ax.legend()


# --------------------------------------------------------------------
# -- Other diagnostic plots: S1 histograms, 2D histograms...
# --------------------------------------------------------------------
def plot_s1_vs_s2_2d(path, ax=None):
    """Plots a 2D histogram of S1 vs S2 areas to visualize the geometric correlation."""
    arrays = np.load(path)
    s1_areas = arrays['s1_areas']
    s2_areas = arrays['s2_areas']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # We use LogNorm() so the rare tail events aren't visually crushed by the main peak
    h = ax.hist2d(s1_areas, s2_areas, 
                  bins=[100, 100], 
                  range=[[0, 0.15], [0, 100]], # Adjust S2 upper limit if needed
                  cmap='viridis', 
                  norm=LogNorm())
    
    # Add the colorbar attached to this specific axis
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Counts (Log Scale)', rotation=270, labelpad=15)
    
    set_name = path.stem.replace('_s2_areas_noS1_cuts', '')
    ax.set(xlabel='S1 Area [mV·μs]', 
           ylabel='S2 Area [mV·μs]', 
           title=f'S1 vs S2 - {set_name}')
    ax.grid(alpha=0.2)
    

def plot_s1_area_distribution(path, ax=None):
    s1_areas = np.load(path)['s1_areas']
    
    # Dynamically find the cut
    from RaTag.el_tpc.fit_s2_area import optimize_s1_cut
    optimal_cut, model_data = optimize_s1_cut(s1_areas)
    
    if ax is None:
        ax = plt.gca()
        
    # CRITICAL: density=True so the GMM curves scale correctly to the histogram
    ax.hist(s1_areas, bins=100, range=(0, 0.15), density=True, alpha=0.5, color='tab:blue')
    
    # Overlay the GMM curves if the fit was successful
    if model_data is not None:
        gmm = model_data['gmm']
        weights = gmm.weights_
        covs = gmm.covariances_.flatten()
        means = gmm.means_.flatten()
        x = model_data['x_smooth']
        
        def gaussian(x_val, mu, sig2, w):
            return w * (1.0 / np.sqrt(2 * np.pi * sig2)) * np.exp(-0.5 * ((x_val - mu)**2 / sig2))
            
        curve_web = gaussian(x, means[model_data['web_idx']], covs[model_data['web_idx']], weights[model_data['web_idx']])
        curve_hole = gaussian(x, means[model_data['hole_idx']], covs[model_data['hole_idx']], weights[model_data['hole_idx']])
        
        ax.plot(x, curve_web, 'b-', lw=1.5, label='Modeled "Web"')
        ax.plot(x, curve_hole, 'orange', lw=1.5, label='Modeled "Hole"')
        ax.plot(x, curve_web + curve_hole, 'k--', lw=1)

    ax.axvline(x=optimal_cut, color='r', linestyle='--', label=f'Bayesian Cut = {optimal_cut:.3f}')
    
    # Extract set name from filename for the title
    set_name = path.stem.replace('_s2_areas', '')
    ax.set(xlabel='S1 Area [mV·μs]', ylabel='Density', title=f'S1 areas - {set_name}')
    ax.legend(fontsize=8)

def plot_s2_area_distribution(path, ax=None):
    arrays = np.load(path)
    s1_areas = arrays['s1_areas']
    s2_areas = arrays['s2_areas']
    
    # Dynamically find the cut to slice the S2 data
    from RaTag.el_tpc.fit_s2_area import optimize_s1_cut
    optimal_cut, _ = optimize_s1_cut(s1_areas)
    
    mask = s1_areas <= optimal_cut
    s2_areas_web = s2_areas[mask]
    s2_areas_tail = s2_areas[~mask]
    uids_web = arrays['uids'][mask]
    uids_tail = arrays['uids'][~mask]
    


    if ax is None:
        ax = plt.gca()
        
    ax.hist(s2_areas_web, bins=100, range=(0, 52), alpha=0.5, density=True, label=f'S2 (Web) N={len(s2_areas_web)}')
    ax.hist(s2_areas_tail, bins=100, range=(0, 52), alpha=0.5, density=True, label=f'S2 (Hole) N={len(s2_areas_tail)}')
    
    set_name = path.stem.replace('_s2_areas', '')
    ax.set(xlabel='Peak Area [mV·μs]', ylabel='Normalized Density', title=f'S2 areas - {set_name}')
    ax.legend(fontsize=8)
    s2_areas_web = S2Areas(areas=s2_areas_web, uids=uids_web)
    s2_areas_tail = S2Areas(areas=s2_areas_tail, uids=uids_tail)
    return s2_areas_web, s2_areas_tail

def plot_set_s1_vs_drift_time(set_pmt, ax=None):
    """Loads and merges data for a single set, and plots on the given axis."""
    # 1. Load both sets of arrays
    timing_arrays = load_npz_arrays(set_pmt, 'timing')
    area_arrays = load_npz_arrays(set_pmt, 's2_areas')
    
    if not timing_arrays or not area_arrays:
        return False
        
    uids_t = timing_arrays.get('uids', np.array([]))
    t_s1 = timing_arrays.get('t_s1', np.array([]))
    t_s2_start = timing_arrays.get('t_s2_start', np.array([]))
    
    uids_a = area_arrays.get('uids', np.array([]))
    s1_areas = area_arrays.get('s1_areas', np.array([]))
    
    if len(uids_t) == 0 or len(uids_a) == 0 or len(t_s1) == 0:
        return False

    # 2. Safely merge by UID to guarantee perfect alignment
    df_time = pd.DataFrame({'uid': uids_t, 't_s1': t_s1, 't_s2_start': t_s2_start}).drop_duplicates(subset=['uid'])
    df_area = pd.DataFrame({'uid': uids_a, 's1_area': s1_areas}).drop_duplicates(subset=['uid'])
    
    df_merged = pd.merge(df_time, df_area, on='uid', how='inner')
    if df_merged.empty:
        return False
        
    # 3. Calculate physical drift time
    drift_times = df_merged['t_s2_start'] - df_merged['t_s1']
    
    # Filter out unphysical negative drift times
    valid_mask = drift_times > 0.0
    dt_flat = drift_times[valid_mask].values
    s1_flat = df_merged['s1_area'][valid_mask].values
    
    if len(dt_flat) == 0:
        return False

    # 4. Plotting
    if ax is None:
        ax = plt.gca()

    # Dynamically frame the X-axis for this specific set
    max_drift = max(dt_flat.max() * 1.05, 1.0) 
    
    h = ax.hist2d(
        x=dt_flat, 
        y=s1_flat, 
        bins=[100, 100],
        range=[[0, max_drift], [0, 0.15]],
        cmap='viridis', 
        norm=LogNorm()
    )
    
    # Add a mini colorbar to each subplot
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_title(set_pmt.source_dir.name, fontsize=10)
    ax.set_xlabel("Drift Time [µs]", fontsize=8)
    ax.set_ylabel("S1 Area [mV·μs]", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(alpha=0.2)
    
    return True
# --------------------------------------------
# -- S1/S2 searching diagnostic plots
# --------------------------------------------

def plot_full_coincidence_diagnostic(wf, config, frame_idx=0, ax=None):
    """
    Step-by-step diagnostic of the exact coincidence math.
    Updated for the 'First-Peak' (Chronological) S1 finding logic and Late S2 cuts.
    """
    from RaTag.waveform.preprocessing import subtract_pedestal, moving_average, threshold_clip
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        
    # --- 1. Preprocessing ---
    wf_single = replace(wf, v=wf.v[[frame_idx]] if wf.ff else wf.v, nframes=1)
    wf_sub = subtract_pedestal(wf_single, config.n_pedestal)
    
    t = wf_single.t
    v_raw = wf_single.v[0] if wf_single.ff else wf_single.v
    v_sub = wf_sub.v[0] if wf_single.ff else wf_sub.v
    dt = t[1] - t[0]
    
    # --- 2. EXACT find_s1 Logic (First-Peak) ---
    s1_mask = (t >= config.s1_t_min) & (t <= config.s1_t_max)
    t_s1_sliced = t[s1_mask]
    v_s1_sliced = v_sub[s1_mask]
    
    # Chronological First-Peak Logic
    over_thresh = v_s1_sliced > config.s1_v_min
    if np.any(over_thresh):
        peak_idx = np.argmax(over_thresh)
        anchor_type = "First Crossing"
    else:
        # If nothing crosses the threshold, argmax defaults to 0.
        # We plot the global max instead to visually show the tallest (failing) peak.
        peak_idx = np.argmax(v_s1_sliced)
        anchor_type = "Global Max (Failing)"
        
    s1_time = t_s1_sliced[peak_idx]
    s1_height = v_s1_sliced[peak_idx]
    
    s1_int_mask = (t >= s1_time - 0.05) & (t <= s1_time + 0.05)
    s1_area = np.sum(v_sub * s1_int_mask) * dt
    
    pass_s1_height = (s1_height >= config.s1_v_min) and (s1_height <= config.s1_v_max)
    pass_s1_area = (s1_area >= 0.0) and (s1_area <= config.s1_max_area)
    
    # If we didn't cross the threshold, force a fail even if the area is technically valid
    s1_valid = pass_s1_height and pass_s1_area and np.any(over_thresh)
    
    # --- 3. EXACT find_s2 Logic ---
    t_min_s2 = -2.5
    wf_smooth = moving_average(wf_single, window=config.s2_window_ma)
    wf_clip = threshold_clip(wf_smooth, threshold=config.bs_threshold)
    v_clip = wf_clip.v[0] if wf_clip.ff else wf_clip.v
    
    s2_mask = t > t_min_s2
    t_s2_sliced = t[s2_mask]
    v_s2_sliced = v_clip[s2_mask]
    
    s2_peak_idx = np.argmax(v_s2_sliced)
    s2_time = t_s2_sliced[s2_peak_idx]
    s2_height = v_s2_sliced[s2_peak_idx]
    
    pass_s2_height = s2_height > config.s2_threshold
    
    # Find S2 Boundaries
    dyn_thresh = max(s2_height * config.s2_fraction, config.s2_threshold)
    below_thresh = v_s2_sliced <= dyn_thresh
    
    left_mask = np.arange(len(v_s2_sliced)) <= s2_peak_idx
    valid_below_left = below_thresh & left_mask
    start_idx = len(v_s2_sliced) - 1 - np.argmax(np.fliplr([valid_below_left])[0]) if np.any(valid_below_left) else 0
    s2_start = t_s2_sliced[min(start_idx + 1, len(v_s2_sliced) - 1)]
    
    right_mask = np.arange(len(v_s2_sliced)) >= s2_peak_idx
    valid_below_right = below_thresh & right_mask
    end_idx = np.argmax(valid_below_right) if np.any(valid_below_right) else len(v_s2_sliced) - 1
    s2_end = t_s2_sliced[max(end_idx - 1, 0)]
    
    s2_width = s2_end - s2_start
    s2_int_mask = (t >= s2_time - 1.5) & (t <= s2_time + 1.5)
    s2_area = np.sum(v_sub * s2_int_mask) * dt
    
    # Check for your new s2_start_max cut (Defaults to 0.5 if not in config)
    s2_start_max = getattr(config, 's2_start_max', 0.5) 
    pass_s2_start_late = s2_start <= s2_start_max
    
    pass_s2_width = s2_width >= config.s2_min_width
    pass_s2_area_min = s2_area >= config.s2_min_area
    pass_s2_area_max = s2_area <= config.s2_max_area
    s2_valid = pass_s2_height and pass_s2_width and pass_s2_area_min and pass_s2_area_max and (s2_peak_idx > 0) and pass_s2_start_late
    
    # --- 4. Plotting ---
    ax.plot(t, v_raw, color='gray', alpha=0.3, label='Raw Waveform')
    ax.plot(t, v_sub, color='blue', lw=1.2, label='Pedestal Subtracted')
    ax.plot(t, v_clip, color='purple', lw=1.0, alpha=0.5, label='S2 Smoothed/Clipped')
    ax.axhline(0, color='black', lw=0.8, linestyle='-')
    
    # Visual Threshold line for First-Peak finding
    ax.axhline(config.s1_v_min, color='orange', lw=1.5, linestyle=':', label=f'S1 Threshold ({config.s1_v_min} mV)')
    
    # S1 Visuals
    ax.axvspan(config.s1_t_min, config.s1_t_max, color='yellow', alpha=0.1, label='S1 Search Window')
    ax.axvspan(s1_time - 0.05, s1_time + 0.05, color='green', alpha=0.2, label='S1 Integration')
    marker_color = 'r*' if s1_valid else 'kx'
    ax.plot(s1_time, s1_height, marker_color, markersize=10, label=f'S1 Anchor ({anchor_type})')
    
    # S2 Visuals
    ax.axvspan(s2_start, s2_end, color='red', alpha=0.1, label='Found S2 Boundaries')
    if not pass_s2_start_late:
        ax.axvline(s2_start_max, color='red', linestyle='--', label=f's2_start_max Cut ({s2_start_max} µs)')
    
    # --- 5. Diagnostic Text Box ---
    s1_status = "PASS" if s1_valid else "FAIL"
    s2_status = "PASS" if s2_valid else "FAIL"
    coinc_status = "ACCEPTED" if (s1_valid and s2_valid) else "REJECTED"
    
    text_str = (
        f"Frame {frame_idx} Diagnostics | Final Status: {coinc_status}\n"
        f"----------------------------------------------------\n"
        f"S1 CUTS (Status: {s1_status})\n"
        f" Anchor: {anchor_type} @ {s1_time:.3f} µs\n"
        f" Height: {s1_height:.2f} mV\t[{config.s1_v_min} - {config.s1_v_max}]\t-> {'PASS' if pass_s1_height else 'FAIL'}\n"
        f" Area:   {s1_area:.4f} mV*µs\t[0.0 - {config.s1_max_area}]\t-> {'PASS' if pass_s1_area else 'FAIL'}\n"
        f"----------------------------------------------------\n"
        f"S2 CUTS (Status: {s2_status})\n"
        f" Height: {s2_height:.2f} mV\t[> {config.s2_threshold}]\t\t-> {'PASS' if pass_s2_height else 'FAIL'}\n"
        f" Width:  {s2_width:.2f} µs\t[> {config.s2_min_width}]\t\t-> {'PASS' if pass_s2_width else 'FAIL'}\n"
        f" Area:   {s2_area:.1f} mV*µs\t[{config.s2_min_area} - {config.s2_max_area}]\t-> {'PASS' if (pass_s2_area_min and pass_s2_area_max) else 'FAIL'}\n"
        f" Start:  {s2_start:.2f} µs\t[<= {s2_start_max}]\t\t-> {'PASS' if pass_s2_start_late else 'FAIL (Late S2)'}\n"
        f" Index0 Trap Avoided: \t\t\t-> {'PASS' if s2_peak_idx > 0 else 'FAIL (Tail Slice)'}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
            
    ax.set_ylim(-5.0, min(max(s1_height, s2_height) * 1.2, 50.0))
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Signal (mV)')
    ax.legend(loc='upper right', fontsize=8)
    
    if ax is None:
        plt.show()