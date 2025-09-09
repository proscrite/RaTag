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