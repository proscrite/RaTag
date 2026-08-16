"""Pure waveform preprocessing operations."""

import numpy as np
from dataclasses import replace
import itertools
from pathlib import Path
import warnings

from RaTag.core.dataIO import load_wfm
from RaTag.core.datatypes import PMTWaveform, Waveform
from scipy.ndimage import uniform_filter1d


def subtract_pedestal(wf: Waveform, n_points: int = 200) -> Waveform:
    if wf.ff:  # 2D array: (nframes, nsamples)
        # keepdims=True ensures shape is (nframes, 1) so it broadcasts correctly during subtraction
        pedestals = wf.v[:, :n_points].mean(axis=1, keepdims=True)
        return replace(wf, v=wf.v - pedestals)
    
    else:  # 1D array: (nsamples,)
        pedestal = wf.v[:n_points].mean()
        return replace(wf, v=wf.v - pedestal)

def subtract_thresholded_pedestal(wf: Waveform, threshold_bs: float = 0.3) -> Waveform:
    """
    Vectorized baseline subtraction using a thresholded masked mean.
    Replicates outlier-rejection boolean masking without Python loops.
    """
    if wf.ff:  # 2D array: (nframes, nsamples)
        
        # 2. Mask positive outliers above the threshold with NaN
        masked_slice = np.where(wf.v < threshold_bs, wf.v, np.nan)
        
        # 3. Count valid points to replicate the `len(v_bs) >= 10` condition
        valid_counts = np.sum(~np.isnan(masked_slice), axis=1, keepdims=True)
        
        # 4. Compute the mean of valid points (ignoring NaNs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pedestals = np.nanmean(masked_slice, axis=1, keepdims=True)
        
        # 5. Apply the fallback (median of the slice) where valid points < 10
        fallback_mask = valid_counts < 10
        if np.any(fallback_mask):
            fallback_pedestals = np.median(v_slice, axis=1, keepdims=True)
            pedestals = np.where(fallback_mask, fallback_pedestals, pedestals)
            
        return replace(wf, v=wf.v - pedestals)
        
    else:  # 1D array: (nsamples,)
        valid_points = wf.v[wf.v < threshold_bs]
        
        if len(valid_points) >= 10:
            pedestal = np.mean(valid_points)
        else:
            pedestal = np.median(wf.v)
            
        return replace(wf, v=wf.v - pedestal)


def subtract_min_baseline(wf_smooth: Waveform) -> Waveform:
    """
    Subtract the minimum value of the waveform as a baseline.
    This is a simple alternative to pedestal subtraction, but currently not in use.
    For this function to work correctly, the input waveform should already be smoothed (e.g., via moving average).
    """
    if wf_smooth.ff:  # 2D array: (nframes, nsamples)
        min_baselines = np.min(wf_smooth.v, axis=1, keepdims=True)
        return replace(wf_smooth, v=wf_smooth.v - min_baselines)
    
    else:  # 1D array: (nsamples,)
        min_baseline = np.min(wf_smooth.v)
        return replace(wf_smooth, v=wf_smooth.v - min_baseline)

def moving_average(wf: PMTWaveform, window: int = 9) -> PMTWaveform:
    """Apply moving average, handling both FastFrame and single frame formats."""
    kernel = np.ones(window)/window
    if wf.ff:
        # Apply convolution to each frame
        # v = np.array([np.convolve(frame, kernel, mode="same") for frame in wf.v])
        v = uniform_filter1d(wf.v, size=window, axis=1, )  # Vectorized moving average for FastFrame
    else:
        v = uniform_filter1d(wf.v, size=window) # Vectorized moving average for single frame
        # v = np.convolve(wf.v, kernel, mode="same")
    return replace(wf, v=v)

def threshold_clip(wf: PMTWaveform, threshold: float = 0.02) -> PMTWaveform:
    v = wf.v.copy()
    v[v < threshold] = 0.0
    return replace(wf, v=v)

def dither_waveform(wf: Waveform, dither_amplitude: float = 0.02) -> Waveform:
    """Add uniform random noise to the waveform to linearize ADC quantization."""
    if dither_amplitude <= 0:
        return wf
        
    dither = np.random.uniform(-dither_amplitude, dither_amplitude, size=wf.v.shape)

    return replace(wf, v=wf.v + dither)

def standard_preprocessing(wf: PMTWaveform,
                           n_pedestal: int = 200,
                           ma_window: int = 9,
                           threshold: float = 0.02) -> PMTWaveform:
    """Apply standard preprocessing: pedestal subtraction, moving average, threshold clipping."""
    wf = subtract_pedestal(wf, n_points=int(n_pedestal))
    wf = moving_average(wf, window=ma_window)
    wf = threshold_clip(wf, threshold=threshold)
    return wf

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
