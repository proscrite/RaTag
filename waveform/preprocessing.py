"""Pure waveform preprocessing operations."""

import numpy as np
from dataclasses import replace
import itertools
from pathlib import Path

from core.dataIO import load_wfm
from core.datatypes import PMTWaveform


def subtract_pedestal(wf: PMTWaveform, n_points: int = 200) -> PMTWaveform:
    pedestal = wf.v[:n_points].mean()
    return replace(wf, v=wf.v - pedestal)

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

def standard_preprocessing(wf: PMTWaveform,
                           n_pedestal: int = 200,
                           ma_window: int = 9,
                           threshold: float = 0.02) -> PMTWaveform:
    """Apply standard preprocessing: pedestal subtraction, moving average, threshold clipping."""
    wf = subtract_pedestal(wf, n_points=n_pedestal)
    wf = moving_average(wf, window=ma_window)
    wf = threshold_clip(wf, threshold=threshold)
    return wf
# --------------------------------------
# Batch processing helpers and averages
# --------------------------------------

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
