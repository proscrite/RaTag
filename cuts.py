# RaTag/cuts.py
from .waveforms import Waveform, PMTWaveform, SiliconWaveform
import numpy as np

def make_baseline_cut(bs_window: tuple[float, float], threshold: float = 0.01, nmax_grass: int = 5):
    """
    Create a baseline noise cut function.
    The cut function returns True if waveform passes, False otherwise.
    Args:
        bs_window: time window (t_min, t_max) to evaluate baseline noise
        threshold: voltage threshold to count "grass" points
        nmax_grass: maximum allowed number of points above threshold in bs_window
    Returns: cut function"""

    def cut(wf):
        V_segment = wf.slice(bs_window[0], bs_window[1]).v
        return (V_segment > threshold).sum() <= nmax_grass
    return cut


def make_time_amplitude_cut(t_start: float, t_end: float = None,
                             threshold: float = 0.02, nmax_grass: int = 5):
    """
    Create an amplitude noise cut function in a given time window.
    The cut function returns True if waveform passes, False otherwise.
    Args:
        t_start: start time of the window
        t_end: end time of the window (None means up to waveform end)
        threshold: voltage threshold to count "grass" points
        nmax_grass: maximum allowed number of points above threshold in t_window
    Returns: cut function"""

    def cut(wf):
        t1 = wf.t[-1] if t_end is None else t_end
        V_segment = wf.slice(t_start, t1).v
        return (V_segment > threshold).sum() <= nmax_grass
    return cut


def amplitude_cut(wf: Waveform, min_amp: float, max_amp: float) -> bool:
    """
    Pass if max amplitude is within [min_amp, max_amp].
    """
    peak = np.max(wf.v) - np.min(wf.v)
    return (min_amp <= peak <= max_amp)


def time_window_cut(wf: Waveform, t_min: float, t_max: float) -> bool:
    """
    Pass if signal peak lies within [t_min, t_max].
    """
    t_peak = wf.t[np.argmax(wf.v)]
    return (t_min <= t_peak <= t_max)
