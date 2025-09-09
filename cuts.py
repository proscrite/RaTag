# RaTag/cuts.py
import numpy as np
from typing import Callable, Any
from dataclasses import dataclass
from .waveforms import Waveform, PMTWaveform, SiliconWaveform
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

@dataclass(frozen=True)
class RejectionLog:
    cut_name: str
    cut_fn: Callable[[Any], tuple[bool, np.ndarray, np.ndarray]]  # wf -> (bool, t_pass, V_pass)
    passed: list[int]
    rejected: list[int]
    reason: str = ""

# -------------------------------
# Cut factories
# -------------------------------
def make_time_amplitude_cut(t_start: float, t_end: float = None,
                            threshold: float = 0.02, nmax_grass: int = 5):
    def cut_fn(wf):
        t1 = wf.t[-1] if t_end is None else t_end
        mask = (wf.t >= t_start) & (wf.t <= t1)
        tsel, Vsel = wf.t[mask], wf.v[mask]
        ok = int((Vsel > threshold).sum()) <= nmax_grass
        return ok, tsel, Vsel
    return cut_fn


def make_vertical_cut(vmin: float):
    """Reject if max(V) < vmin."""
    def cut_fn(wf):
        ok = wf.v.max() >= vmin
        return ok, wf.t[ok], wf.v[ok]
    return cut_fn


def moving_average(x, window_size=9):
    """Simple moving average filter"""
    return np.convolve(x, np.ones(window_size)/window_size, mode="same")

def make_smoothed_vertical_cut(vmin: float, t_start: float, t_end: float, smooth_window: int = 9):
    """Accept only values of smoothed wf over vmin"""
    def cut_fn(wf):
        s2_mask = (wf.t >= t_start) & (wf.t <= t_end)
        V_s2 = wf.v[s2_mask]
        V_smooth = moving_average(V_s2, smooth_window)
        ok = V_smooth[V_smooth > vmin]
        return ok, wf.t[ok], wf.v[ok]
    return cut_fn