import matplotlib.pyplot as plt # type: ignore
import time
import numpy as np 
# import ipywidgets as widgets # type: ignore
# from IPython.display import display
from typing import Optional
from pathlib import Path
import pandas as pd

from RaTag.core.datatypes import PMTWaveform, SetPmt, RejectionLog, S2Areas, Run
from RaTag.core.dataIO import load_wfm, iter_waveforms
from RaTag.core.units import s_to_us, V_to_mV

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

# ------------------------------------------------
# Advanced waveform plotters (with S1/S2 window)
# ------------------------------------------------

def _get_metadata_kwargs(kwargs: dict, metadata: dict):
    """Helper to get timing parameters from kwargs or metadata."""
    time_keys = ["t_s1", "t_s1_std", "t_s2_start", "t_s2_start_std", "t_s2_end", "t_s2_end_std"]
    for key in time_keys:
        if key not in kwargs:
            kwargs[key] = metadata.get(key)

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

    
    kwargs = _get_metadata_kwargs(kwargs, set_pmt.metadata) # get timing params
    
    if file_index is None:
        file_index = np.random.randint(0, len(set_pmt.filenames))
    
    fn = set_pmt.filenames[file_index]

    if ax == None:
        ax = plt.gca()

    wf = load_wfm(set_pmt.source_dir / fn)
    _, v_max = plot_waveform(wf, frame=frame, ax=ax, title=f"Gate {set_pmt.metadata['gate']} V", color=color)

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
    print(selected_files)

    for ax, fn in zip(axes, selected_files):
        plot_set_windows(set_pmt, file_index=set_pmt.filenames.index(fn), ax=ax, **kwargs)
    
    return fig, axes

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
# Interactive plotters
# --------------------------------


def make_interactive(plot_fn):
    """Decorator that adds interactive scrolling to a waveform plotting function.
    
    The wrapped function must take a waveform as its first argument and return
    a matplotlib axes object.
    """
    def wrapper(set_pmt: SetPmt, *args, fix_axes=True, **kwargs):
        files = list(set_pmt.filenames)
        xlim = None

        def _plot(idx):
            fig, ax = plt.subplots()
            wf = load_wfm(set_pmt.source_dir / files[idx])
            plot_fn(set_pmt, wf, *args, **kwargs, ax=ax)

            if fix_axes:
                nonlocal xlim
                if xlim is None:
                    xlim = ax.get_xlim()
                else:
                    ax.set_xlim(xlim)
            ax.relim()
            ax.autoscale(axis="y")
            plt.gcf().canvas.draw()

        slider = widgets.IntSlider(
            min=0, 
            max=len(files)-1,
            step=1,
            value=0,
            description='Waveform:'
        )
        out = widgets.interactive_output(_plot, {"idx": slider})
        display(slider, out)
        
    return wrapper

# Now we can decorate plot_winS2_wf to make it interactive
@make_interactive
def scroll_winS2(set_pmt: SetPmt, wf: PMTWaveform, width_s2: float, ts2_tol: float = 0, ax=None):
    """Interactive version of plot_winS2_wf."""

    t_s1 = set_pmt.metadata.get("t_s1")
    time_drift = set_pmt.time_drift

    if t_s1 is None:
        raise ValueError("t_s1 must be provided either as argument or in set metadata")
    if time_drift is None:
        raise ValueError("time_drift must be provided either as argument or in set")
    
    return plot_winS2_wf(wf, t_s1, time_drift, width_s2, ts2_tol, ax)


def plot_run_winS2(run: Run, ts2_tol: float = 0, scroll: bool = False):
    """Plot S1/S2 windows for one waveform from each set in a run."""
    if not scroll:
        n_sets = len(run.sets)
        fig, axes = plt.subplots(n_sets, 1, figsize=(10, 4*n_sets))
        if n_sets == 1:
            axes = [axes]
    else:
        fig, axes = None, []

    def _plot_adapter(set_pmt: SetPmt, wf: PMTWaveform, width_s2: float, ts2_tol: float, ax=None):
        t_s1 = set_pmt.metadata.get("t_s1")
        time_drift = set_pmt.time_drift
        return plot_winS2_wf(wf, t_s1, time_drift, width_s2, ts2_tol, ax)

    decorated_fn = make_interactive(_plot_adapter) if scroll else _plot_adapter

    for idx, set_pmt in enumerate(run.sets):
        try:
            if scroll:
                # do NOT pass wf — the decorator will supply it
                decorated_fn(set_pmt, width_s2=run.width_s2, ts2_tol=ts2_tol)
            else:
                wf = load_wfm(set_pmt.source_dir / set_pmt.filenames[0])
                decorated_fn(set_pmt, wf, run.width_s2, ts2_tol, ax=axes[idx])

                axes[idx].set_title(f"Set {set_pmt.source_dir.name}")
        except Exception as e:
            if not scroll:
                axes[idx].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')

    if not scroll:
        plt.tight_layout()
    return fig, axes


#### Manual iteration version
def iter_plot_waveforms(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float):
    for idx, wf in enumerate(iter_waveforms(set_pmt)):
        plot_cut_results(wf, set_pmt, logs, width_s2)
        yield

# Use: next(gen)  # plot first
#      next(gen)  # plot next
#      ...

### Interactive version with slider

def scroll_waveforms(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float):
    files = list(set_pmt.filenames)
    def _plot(idx):
        fig, ax = plt.subplots()
        wf = load_wfm(set_pmt.source_dir / files[idx])
        plot_cut_results(wf, set_pmt, logs, ax=ax,
                         width_s2 = width_s2);
        # ax = plt.gca()
        ax.set_xlim(-1.7e-5, 5e-5)
        ax.relim()
        ax.autoscale(axis="y")     # auto y-scale
        plt.gcf().canvas.draw()

    slider =  widgets.IntSlider(min=0, max=len(set_pmt.filenames)-1, step=1, value=0)
    out = widgets.interactive_output(_plot, {"idx": slider})
    display(slider, out)

### Auto slide version

def slideshow(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float, delay=2.0, ax:plt.Axes = None):
    """Auto-advance through waveforms with a fixed delay (in seconds)."""
    if ax is None:
        ax = plt.gca()
    for idx, wf in enumerate(set_pmt.iter_waveforms()):
        ax.clear()
        print(f"Waveform {idx+1}/{len(set_pmt.filenames)}")
        plot_cut_results(wf, set_pmt, logs=logs, 
                         width_s2=width_s2, ax=ax)
        plt.draw()
        time.sleep(delay)
        plt.pause(0.01)  # allow GUI to update


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
                               energy_range: tuple = (4, 8)) -> tuple:
    """
    Plot alpha energy spectrum histogram.
    
    Args:
        energies: Array of alpha energies [MeV]
        title: Plot title
        nbins: Number of histogram bins
        energy_range: (min, max) energy range [MeV]
        
    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(energies, bins=nbins, range=energy_range, alpha=0.7, edgecolor='black')
    ax.set(xlabel='Energy [MeV]', ylabel='Counts', title=title)
    ax.grid(True, alpha=0.3)
    
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
# Deprecated functions
# --------------------------------


def plot_waveform_with_cuts(wf: PMTWaveform, set_pmt: SetPmt,
                            width_s2: float):
    t, V = wf.t, wf.v
    t_s1 = set_pmt.metadata["t_s1"]
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

def plot_cut_results(wf: PMTWaveform, set_pmt: SetPmt, logs: list[RejectionLog],
                     width_s2: float, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # t, V = wf.t, wf.v
    wf_index = getattr(wf, "index", None)
    ax.set(title=f"Waveform {wf_index}",
           xlabel="Time (s)", ylabel="Signal (V)")
    # ax.plot(t, V, color="0.6")
    wf.plot()
    for log in logs:
        ok, tsel, Vsel = log.cut_fn(wf)
        color = "g" if ok else "r"
        ax.plot(tsel, Vsel, color, label=f"{log.cut_name} {'PASS' if ok else 'FAIL'}")

    # markers for S1 / S2
    t_s1 = set_pmt.metadata.get("t_s1")
    t_drift = set_pmt.time_drift
    if t_s1 and t_drift:
        s2_start = t_s1 + t_drift
        s2_end = s2_start + width_s2
        ax.axvline(t_s1, color="k", ls="--", label="S1")
        ax.axvline(s2_start, color="m", ls="--", label="S2 start")
        ax.axvline(s2_end, color="r", ls="--", label="S2 end")

    ax.legend()
    return ax

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


def plot_xray_validation(set_pmt: SetPmt,
                        accepted_sample: list,
                        rejected_sample: list,
                        title: str = "X-ray Classification Validation"):
    """
    Plot validation figure showing accepted vs rejected X-ray candidates.
    
    Creates 2-column layout: left column shows accepted examples,
    right column shows rejected examples.
    
    Args:
        set_pmt: SetPmt to load waveforms from
        accepted_sample: List of (file_seq, frame_idx) tuples for accepted frames
        rejected_sample: List of (file_seq, frame_idx) tuples for rejected frames
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    from RaTag.core.uid_utils import parse_file_seq_from_name
    
    # Build mapping from file_seq to file_index
    file_seq_to_index = {parse_file_seq_from_name(fn): idx 
                         for idx, fn in enumerate(set_pmt.filenames)}
    
    n_frames = len(accepted_sample)  # Both samples have same length
    
    if n_frames == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No frames to plot", ha='center', va='center')
        ax.axis('off')
        return fig
    
    # Create 2-column layout
    fig, axes = plt.subplots(n_frames, 2, figsize=(14, 3.5 * n_frames))
    if n_frames == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    
    fig.suptitle(title, fontsize=14, y=0.995)
    
    # Plot both columns in single loop
    for i in range(n_frames):
        # Left column: accepted (green)
        file_seq, frame_idx = accepted_sample[i]
        file_index = file_seq_to_index[file_seq]  # Convert file_seq to file_index
        ax_left = axes[i, 0]
        plot_set_windows(set_pmt, file_index=file_index, frame=frame_idx, 
                        ax=ax_left, color='green')
        ax_left.set_title(f"✓ Accepted (File {file_seq}, Frame {frame_idx})", 
                         fontsize=10, color='darkgreen')
        
        # Right column: rejected (red)
        file_seq, frame_idx = rejected_sample[i]
        file_index = file_seq_to_index[file_seq]  # Convert file_seq to file_index
        ax_right = axes[i, 1]
        plot_set_windows(set_pmt, file_index=file_index, frame=frame_idx, 
                        ax=ax_right, color='red')
        ax_right.set_title(f"✗ Rejected (File {file_seq}, Frame {frame_idx})", 
                          fontsize=10, color='darkred')
    
    plt.tight_layout()
    return fig
