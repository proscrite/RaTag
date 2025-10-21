# transformations.py
# Transformations on waveforms only (not on sets)
import numpy as np
import itertools
from dataclasses import replace
from typing import Callable
from scipy.signal import find_peaks
from pathlib import Path
from .dataIO import load_wfm
from .datatypes import PMTWaveform

def t_in_us(wf: PMTWaveform, ) -> PMTWaveform:
    """Scale waveform values by a constant factor."""
    return replace(wf, t=wf.t * 1e6)

def v_in_mV(wf: PMTWaveform) -> PMTWaveform:
    """Scale waveform values by a constant factor."""
    return replace(wf, v=wf.v * 1e3)

def subtract_pedestal(wf: PMTWaveform, n_points: int = 200) -> PMTWaveform:
    pedestal = wf.v[:n_points].mean()
    return replace(wf, v=wf.v - pedestal)

def extract_window(wf: PMTWaveform, t_start: float, t_end: float) -> PMTWaveform:
    """Extract time window from waveform, handling both FastFrame and single frame formats."""
    mask = (wf.t >= t_start) & (wf.t <= t_end)
    if wf.ff:
        v = wf.v[:, mask]  # Apply mask to all frames along time axis
    else:
        v = wf.v[mask]
    return replace(wf, t=wf.t[mask], v=v)

def moving_average(wf: PMTWaveform, window: int = 9) -> PMTWaveform:
    """Apply moving average, handling both FastFrame and single frame formats."""
    kernel = np.ones(window)/window
    if wf.ff:
        # Apply convolution to each frame
        v = np.array([np.convolve(frame, kernel, mode="same") for frame in wf.v])
    else:
        v = np.convolve(wf.v, kernel, mode="same")
    return replace(wf, v=v)

def threshold_clip(wf: PMTWaveform, threshold: float = 0.02) -> PMTWaveform:
    v = wf.v.copy()
    v[v < threshold] = 0.0
    return replace(wf, v=v)

def integrate_riemann(wf: PMTWaveform, dt: float = 2e-4) -> np.ndarray:
    """
    Integrate waveform, returning array of integration values.
    For FastFrame: returns array of length nframes
    For single frame: returns array of length 1
    """
    if wf.ff:
        return np.sum(wf.v, axis=1) * dt  # Integrate each frame separately
    return np.array([np.sum(wf.v) * dt])  # Return single-element array

def integrate_trapz(wf: PMTWaveform, dt: float = 2e-4) -> np.ndarray:
    """
    Integrate waveform, returning array of integration values.
    For FastFrame: returns array of length nframes
    For single frame: returns array of length 1
    """
    if wf.ff:
        return np.trapz(wf.v, dx=dt, axis=1)  # Integrate each frame separately
    return np.array([np.trapz(wf.v, dx=dt)])  # Return single-element array

# Convenience pipeline (compose functions)
def s2_area_pipeline(wf: PMTWaveform,
                     t_window: tuple[float, float],
                     n_pedestal: int = 200,
                     ma_window: int = 10,
                     bs_threshold: float = 0.8,
                     dt: float = 2e-4,
                     integrator: Callable[[PMTWaveform, float], np.ndarray] = integrate_trapz,
                     ) -> float:
    
    wf = t_in_us(wf)
    wf = v_in_mV(wf)
    wf = subtract_pedestal(wf, n_points=n_pedestal)
    wf = extract_window(wf, *t_window)
    wf = moving_average(wf, window=ma_window)
    wf = threshold_clip(wf, threshold=bs_threshold)
    return integrator(wf, dt=dt)
 

# -------------------------------
# Batch processing helpers
# -------------------------------

def batch_filenames(filenames: list[Path], batch_size: int = 20):
    """Yield batches of filenames (lists) of size batch_size."""
    it = iter(sorted(filenames))
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch

def average_waveform(batch_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute average waveform for a batch of files, handling both FastFrame and single frame formats.
    For FastFrame files, averages all frames in the file.
    For single frame files, averages across files in batch.
    
    Args:
        batch_files: List of paths to waveform files
        
    Returns:
        tuple (t, V_avg) where:
            t: Time array
            V_avg: Average voltage array
    """
    if isinstance(batch_files, PMTWaveform):  # Single FastFrame file passed directly
        wf = batch_files
        if not wf.ff:
            raise ValueError("Expected FastFrame waveform")
        return wf.t, wf.v.mean(axis=0) if wf.ff else wf.v
    
    waveforms = [load_wfm(fn) for fn in batch_files]
    t = waveforms[0].t
    
    V_list = []
    for wf in waveforms:
        if wf.ff:
            # For FastFrame, use all frames from this single file
            V_list.extend(wf.v)  # wf.v is already a matrix of shape (nframes, samples)
            break  # Only use first FastFrame file
        else:
            # For single frame, add the single waveform
            V_list.append(wf.v)
            
    V_stack = np.stack(V_list)
    V_avg = V_stack.mean(axis=0)
    
    return t, V_avg

def find_s1_in_avg(t, V, height_S1=0.001, min_distance=200):
    """Find S1 peak index in averaged waveform.
     Returns None if no peak found.
     Note: this assumes S1 appears before t=0.
    """
    mask = t < 0
    inds = find_peaks(V[mask], height=height_S1, distance=min_distance)[0]
    if len(inds) == 0:
        return None
    return inds[np.argmax(V[inds])] if len(inds) > 1 else inds[0]
