import matplotlib.pyplot as plt # type: ignore
import time
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from typing import List
from pathlib import Path

from .datatypes import PMTWaveform, SetPmt, RejectionLog, S2Areas, Run
from .dataIO import load_wfm, iter_waveforms
from .units import s_to_us, V_to_mV

def plot_waveform(wf: PMTWaveform, ax=None, title: str = "Waveform", color: str = "g"):
    """Plot a single waveform."""
    if ax is None:
        fig, ax = plt.subplots()

    if wf.ff:
        print("Averaging FastFrame waveform for plotting")
        # Average all frames for FastFrame
        V = wf.v[0, :]
        t = wf.t
    else:
        t, V = wf.t, wf.v
    
    if wf.t[1] - wf.t[0] < 1e-7:  # Hardcoded threshold to distinguish s vs µs
        # print("Converting time to µs for better readability")
        t = s_to_us(t)  # convert to µs
        V = V_to_mV(V)  # convert to mV
    
    wf_index = Path(wf.source).name.replace(".wfm", "").replace("Wfm", "")
    title = f"{title} {wf_index}"
    if wf.ff:
        title += f" (Average of {wf.nframes} frames)"
    
    ax.set(title=title, xlabel="Time (µs)", ylabel="Signal (mV)")
    ax.plot(t, V, color=color, alpha=0.6)
    return ax

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
    Plot S2 area histogram with Gaussian fit.
    
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

def plot_s2_vs_drift(run: Run, fitted: dict[str, S2Areas], normalized: bool = False):
    """
    Plot S2 area vs drift field.
    
    Args:
        run: Run object
        fitted: Dictionary of fitted S2Areas
        normalized: If True, normalize by X-ray reference (requires run.A_x_mean and run.g_S2)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    drift_fields = [s.drift_field for s in run.sets]
    means = [fitted[s.source_dir.name].mean for s in run.sets]
    ci95s = [fitted[s.source_dir.name].ci95 for s in run.sets]
    
    if normalized and run.A_x_mean is not None and run.g_S2 is not None:
        # Normalize by X-ray reference
        means = [m / run.A_x_mean for m in means]
        ci95s = [c / run.A_x_mean for c in ci95s]
        ylabel = "Normalized S2 Area (A_ion / A_xray)"
        title_suffix = " (Normalized)"
    else:
        ylabel = "Mean S2 Area (mV·µs)"
        title_suffix = ""

    ax.errorbar(drift_fields, means, yerr=ci95s, fmt='o', capsize=5, markersize=8)
    ax.set(xlabel="Drift field (V/cm)", 
           ylabel=ylabel, 
           title=f"Run {run.run_id} — Mean S2 Area vs Drift Field{title_suffix}")
    ax.grid(True)
    
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

def plot_s1_time_distribution(s1_times: List[float], 
                            title: str = "S1 Peak Times Distribution",
                            method: str = "mad"):
    """Plot histogram of S1 peak times with mean and std."""
    from RaTag.constructors import compute_s2_variance
    plt.figure(figsize=(10, 6))

    n, bins, _ = plt.hist(s1_times, bins=50);
    cbins = 0.5 * (bins[1:] + bins[:-1])
    t_mode, dt_mean = compute_s2_variance(s1_times, method=method)

    # t_mode = cbins[np.argmax(n)]
    # dt_mean = np.std(s1_times)
    plt.axvline(t_mode, color='red', linestyle='--', label='Mode S1 time')

    plt.fill_between((t_mode - dt_mean, t_mode + dt_mean), 0, max(n),
                     color='g', alpha=0.2)

    plt.gca().set(xlabel='S1 Peak Time (μs)', ylabel='Counts', title=title)
    plt.legend()

def plot_waveforms_with_s1_s2(set_pmt: SetPmt,
                             n_waveforms: int = 10,
                             t_s1_mean: float = None,
                             t_s1_std: float = None,
                             t_s2_start_mean: float = None,
                             t_s2_start_std: float = None,
                             t_s2_end_mean: float = None,
                             t_s2_end_std: float = None) -> tuple:
    """
    Plot multiple waveforms with S1 and S2 timing markers.
    
    Args:
        set_pmt: SetPmt object
        n_waveforms: Number of waveforms to plot
        t_s1_mean: Mean S1 time (µs), from metadata if None
        t_s1_std: S1 time spread (µs)
        t_s2_start_mean: Mean S2 start time (µs)
        t_s2_start_std: S2 start spread (µs)
        t_s2_end_mean: Mean S2 end time (µs)
        t_s2_end_std: S2 end spread (µs)
        figsize: Figure size
        
    Returns:
        (fig, axes)
    """
    from .dataIO import load_wfm
    
    # Get S1 from metadata if not provided
    if t_s1_mean is None:
        t_s1_mean = set_pmt.metadata.get("t_s1")
    if t_s1_std is None:
        t_s1_std = set_pmt.metadata.get("t_s1_std", 0)
    if t_s2_start_mean is None:
        t_s2_start_mean = set_pmt.metadata.get("t_s2_start_mean")
    if t_s2_start_std is None:
        t_s2_start_std = set_pmt.metadata.get("t_s2_start_std", 0)
    if t_s2_end_mean is None:
        t_s2_end_mean = set_pmt.metadata.get("t_s2_end_mean")
    if t_s2_end_std is None:
        t_s2_end_std = set_pmt.metadata.get("t_s2_end_std", 0)
    
    # Select waveforms
    filenames = set_pmt.filenames[:n_waveforms]

    fig, axes = plt.subplots(len(filenames), 1, figsize=(6, 4*n_waveforms), sharex=False)
    if n_waveforms == 1:
        axes = [axes]
    
    for i, fn in enumerate(filenames):
        wf = load_wfm(set_pmt.source_dir / fn)
        plot_waveform(wf, ax=axes[i])
        wf.v = V_to_mV(wf.v[0, :]) if wf.ff else V_to_mV(wf.v)
        axes[i].set_title(f"Waveform {i+1}", fontsize=10)
        
        # S1 markers
        if t_s1_mean is not None:
            axes[i].axvline(t_s1_mean, color='red', linestyle='--', 
                          lw=1.5, label='S1 Mean')
            if t_s1_std > 0:
                y_max = wf.v.max()
                print(f'y_max: {y_max:.2f} V') 
                axes[i].fill_betweenx([0, y_max], 
                                     t_s1_mean - t_s1_std,
                                     t_s1_mean + t_s1_std,
                                     color='red', alpha=0.1, label='S1 ±σ')
        
        # S2 start markers
        if t_s2_start_mean is not None:
            axes[i].axvline(t_s2_start_mean, color='blue', linestyle='--',
                          lw=1.5, label='S2 Start Mean')
            if t_s2_start_std is not None and t_s2_start_std > 0:
                y_max = wf.v.max()
                axes[i].fill_betweenx([0, y_max],
                                     t_s2_start_mean - t_s2_start_std,
                                     t_s2_start_mean + t_s2_start_std,
                                     color='blue', alpha=0.2, label='S2 Start ±σ')
        
        # S2 end markers
        if t_s2_end_mean is not None:
            axes[i].axvline(t_s2_end_mean, color='purple', linestyle='--',
                          lw=1.5, label='S2 End Mean')
            if t_s2_end_std is not None and t_s2_end_std > 0:
                y_max = wf.v.max()
                axes[i].fill_betweenx([0, y_max],
                                     t_s2_end_mean - t_s2_end_std,
                                     t_s2_end_mean + t_s2_end_std,
                                     color='purple', alpha=0.2, label='S2 End ±σ')
        
        axes[i].legend(fontsize=8, loc='upper left')
        axes[i].grid(alpha=0.3)
    
    axes[-1].set_xlabel("Time (µs)")
    fig.suptitle(f"S1/S2 Timing - {set_pmt.source_dir.name}", fontsize=12)
    plt.tight_layout()
    
    return fig, axes


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