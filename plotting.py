import matplotlib.pyplot as plt # type: ignore
import time
import ipywidgets as widgets
from IPython.display import display

from .waveforms import PMTWaveform
from .measurement import SetPmt, RejectionLog
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
    plt.axvline(drift_window[0], color="k", label="S1")
    plt.axvline(drift_window[1], color="m", label="S2 start")
    plt.axvline(s2_window[1], color="r", label="S2 end")
    plt.legend()

def plot_cut_results(wf: PMTWaveform, set_pmt: SetPmt, logs: list[RejectionLog],
                     wf_index: int, width_s2: float, ax = None):
    if ax is None:
        ax = plt.gca()
    t, V = wf.t, wf.v
    t_s1 = set_pmt.metadata["t_s1"]
    t_drift = set_pmt.time_drift / 1e6  # µs→s
    t_end = wf.t[-1]

    drift_window = (t_s1, t_s1 + t_drift)
    s2_window = (drift_window[1], drift_window[1] + width_s2)
    post_s2_window = (s2_window[1], t_end)

    # Map logs into a dict: cut_name -> passed?
    cut_results = {log.cut_name: wf_index in log.passed for log in logs}

    # base trace
    ax.plot(t, V, label=f"Waveform {wf_index}")

    # drift region
    drift_seg = (t > drift_window[0]) & (t < drift_window[1])
    ax.plot(t[drift_seg], V[drift_seg],
             "g" if cut_results["drift_region"] else "r", lw=2)

    # post-S2 region
    post_seg = t > s2_window[1]
    ax.plot(t[post_seg], V[post_seg],
             "g" if cut_results["post_s2"] else "r", lw=2, )

    # # baseline (before S1)
    # base_seg = (t > set_pmt.metadata["baseline_window"][0]) & (t < set_pmt.metadata["baseline_window"][1])
    # ax.plot(t[base_seg], V[base_seg],
    #          "g" if cut_results["baseline"] else "r", lw=2)

    # vertical markers
    ax.axvline(drift_window[0], color="k", label="S1")
    ax.axvline(drift_window[1], color="m", label="S2 start")
    ax.axvline(s2_window[1], color="r", label="S2 end")

    ax.legend()


#### Manual iteration version
def iter_plot_waveforms(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float):
    for idx, wf in enumerate(set_pmt.iter_waveforms()):
        plot_cut_results(wf, set_pmt, logs, width_s2, wf_index=idx)
        yield

# Use: next(gen)  # plot first
#      next(gen)  # plot next
#      ...

### Interactive version with slider

def scroll_waveforms(set_pmt: SetPmt, logs: list[RejectionLog], width_s2: float):
    files = list(set_pmt.filenames)
    def _plot(idx):
        wf = load_wfm(set_pmt.source_dir / files[idx])
        plot_cut_results(wf, set_pmt, logs, 
                         width_s2 = width_s2, wf_index=idx)
        ax = plt.gca()
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
        plot_cut_results(wf, set_pmt, logs=logs, 
                         width_s2=width_s2, wf_index=idx, ax=ax)
        plt.pause(delay)