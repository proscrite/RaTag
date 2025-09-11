# transformations.py
# Transformations on waveforms only (not on sets)
import numpy as np
from .waveforms import PMTWaveform

def t_in_us(wf: PMTWaveform, ) -> PMTWaveform:
    """Scale waveform values by a constant factor."""
    return PMTWaveform(t=wf.t * 1e6, v=wf.v, source=wf.source)

def v_in_mV(wf: PMTWaveform) -> PMTWaveform:
    """Scale waveform values by a constant factor."""
    return PMTWaveform(t=wf.t, v=wf.v * 1e3, source=wf.source)

def subtract_pedestal(wf: PMTWaveform, n_points: int = 200) -> PMTWaveform:
    pedestal = wf.v[:n_points].mean()
    return PMTWaveform(t=wf.t, v=wf.v - pedestal, source=wf.source)

def extract_window(wf: PMTWaveform, t_start: float, t_end: float) -> PMTWaveform:
    mask = (wf.t >= t_start) & (wf.t <= t_end)
    return PMTWaveform(t=wf.t[mask], v=wf.v[mask], source=wf.source)

def moving_average(wf: PMTWaveform, window: int = 9) -> PMTWaveform:
    v = np.convolve(wf.v, np.ones(window)/window, mode="same")
    return PMTWaveform(t=wf.t, v=v, source=wf.source)

def threshold_clip(wf: PMTWaveform, threshold: float = 0.02) -> PMTWaveform:
    v = wf.v.copy()
    v[v < threshold] = 0.0
    return PMTWaveform(t=wf.t, v=v, source=wf.source)

def integrate(wf: PMTWaveform, dt: float = 2e-4) -> float:
    return np.sum(wf.v) * dt

# Convenience pipeline (compose functions)
def s2_area_pipeline(
    wf: PMTWaveform,
    t_window: tuple[float, float],
    n_pedestal: int = 200,
    ma_window: int = 10,
    threshold: float = 0.02,
    dt: float = 2e-4,
) -> float:
    wf = t_in_us(wf)
    wf = v_in_mV(wf)
    wf = subtract_pedestal(wf, n_points=n_pedestal)
    wf = extract_window(wf, *t_window)
    wf = moving_average(wf, window=ma_window)
    wf = threshold_clip(wf, threshold=threshold)
    return integrate(wf, dt=dt)
 