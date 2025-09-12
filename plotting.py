import matplotlib.pyplot as plt # type: ignore
import time
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from .datatypes import PMTWaveform, SetPmt, RejectionLog, S2Areas, Run
from .dataIO import load_wfm

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


#### Manual iteration version
def iter_plot_waveforms(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float):
    for idx, wf in enumerate(set_pmt.iter_waveforms()):
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
    if ax is None:
        fig, ax = plt.subplots()

    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        ax.text(0.5, 0.5, "No data in range", ha='center', va='center')
        return ax

    n, bins, patches = ax.hist(area_vec, bins=nbins, alpha=0.6, color='g', label="Data")
    ax.set_xlabel("S2 Area (mV·µs)")
    ax.set_ylabel("Counts")
    ax.set_title(f"S2 Area Histogram for Set {s2.set_id}")
    ax.grid(True)

    if s2.fit_success and s2.fit_result:
        x = np.linspace(bin_cuts[0], bin_cuts[1], 1000)
        y = s2.fit_result.eval(x=x)  # Use stored fit result to evaluate
        ax.plot(x, y, 'r-', label="Gaussian Fit")
        ax.axvline(s2.mean, color='b', ls='--', 
                  label=f"Mean: {s2.mean:.1f} ± {s2.ci95:.1f}")
        ax.legend()
    else:
        ax.text(0.5, 0.9, "Fit failed or not performed", 
                ha='center', va='center', transform=ax.transAxes)

    return ax

def plot_s2_vs_drift(run: Run, fitted: dict[str, S2Areas]):
    drift_fields = [s.drift_field for s in run.sets]
    means = [fitted[s.source_dir.name].mean for s in run.sets]
    ci95s = [fitted[s.source_dir.name].ci95 for s in run.sets]

    plt.errorbar(drift_fields, means, yerr=ci95s,
                 fmt='o', capsize=5)
    plt.gca().set(xlabel="Drift field (V/cm)", ylabel ="Mean S2 area (mV·µs)", 
                  title =f"Run {run.run_id} — Mean S2 Area vs Drift Field")
    plt.grid(True)
