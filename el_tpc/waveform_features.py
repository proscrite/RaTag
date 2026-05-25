from typing import Optional, Dict, Union, List
import numpy as np

from RaTag.waveform.preprocessing import moving_average, subtract_pedestal, threshold_clip
from RaTag.core.datatypes import Waveform

def compute_waveform_baseline(wf: Waveform, n_points: int = 200) -> tuple[float, float]:
    """
    Calculates the baseline median and noise floor (std) of a single waveform 
    using the pre-trigger region.
    """
    safe_points = min(n_points, wf.nframes)
    
    # Extract the pre-trigger slice safely for both 1D and 2D flat formats
    pre_trigger_slice = wf.v[:, :safe_points] if wf.ff else wf.v[:safe_points]
    
    baseline_median = float(np.median(pre_trigger_slice))
    baseline_std = float(np.std(pre_trigger_slice))
    
    return baseline_median, baseline_std


def _compute_left_half_std(times_clean: np.ndarray, mode: float) -> float:
    """
    Computes a robust standard deviation for heavily right-skewed distributions 
    by mirroring the variance of the left half of the peak. 
    Used to prevent trailing tails from artificially widening the S2 End window.
    """
    left_vals = times_clean[times_clean <= mode]
    if len(left_vals) > 1:
        return float(np.sqrt(np.mean((left_vals - mode)**2)))
    return float(np.std(times_clean))  # Fallback if distribution is weird


def compute_timing_statistics(times: Union[np.ndarray, List[float]],
                              name: str,
                              pre_cut: Optional[tuple] = None,
                              outlier_sigma: float = 3.0) -> Dict[str, float]:
    """Compute timing statistics with outlier rejection. Safely accepts raw lists."""
    # Convert list to array natively inside the function
    times_arr = np.asarray(times, dtype=np.float32)

    if len(times_arr) == 0:
        return {name: None, f"{name}_std": 0.0}

    if pre_cut is not None:
        times_arr = times_arr[(times_arr >= pre_cut[0]) & (times_arr <= pre_cut[1])]
        if len(times_arr) == 0:
            return {name: None, f"{name}_std": 0.0}
    
    # Outlier rejection
    mean_init = np.nanmean(times_arr)
    std_init = np.nanstd(times_arr)

    if std_init == 0:  # Protect against single-element or identical arrays
        return {name: round(float(mean_init), 3), f"{name}_std": 0.0}
        
    mask = np.abs(times_arr - mean_init) < (outlier_sigma * std_init)
    times_clean = times_arr[mask]

    if len(times_clean) == 0:
        return {name: None, f"{name}_std": 0.0}
    
    # Compute mode from histogram
    n, bins = np.histogram(times_clean, bins=100)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    mode = round(float(cbins[np.argmax(n)]), 3)
    if name == "t_s2_end":
        std = round(_compute_left_half_std(times_clean, mode), 3)
    else:
        std = round(float(np.std(times_clean)), 3)

    return {name: mode, f"{name}_std": std}

def find_s1(wf: Waveform, threshold: float = 1.0, t_max: float = -2.5) -> np.ndarray:
    """Vectorized S1 detection. Returns array of peak times (or NaN)."""

    wf = subtract_pedestal(wf, n_points=200)
    mask = wf.t < t_max
        
    t_sliced = wf.t[mask]
    v_sliced = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
    
    # Boolean matrix of threshold crossings
    above_thresh = v_sliced > threshold
    has_peak = np.any(above_thresh, axis=1)
    
    # Find rightmost peak (flip, find first, un-flip)
    first_flipped_idx = np.argmax(np.fliplr(above_thresh), axis=1)
    last_peak_idx = (v_sliced.shape[1] - 1) - first_flipped_idx
    
    # Build results
    s1_times = np.full(wf.nframes, np.nan, dtype=np.float32)
    s1_times[has_peak] = t_sliced[last_peak_idx[has_peak]]
    return s1_times

def find_s2(wf: Waveform, threshold_s2: float = 0.8, t_min: float = 0.0, 
            window_size: int = 9, threshold_bs: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized S2 boundary tracking. Returns (starts, ends) arrays."""

    wf = moving_average(wf, window=window_size)
    wf = threshold_clip(wf, threshold=threshold_bs)
    mask = wf.t > t_min
    if not np.any(mask):
        nan_arr = np.full(wf.nframes, np.nan, dtype=np.float32)
        return nan_arr, nan_arr
        
    t_sliced = wf.t[mask]
    v_sliced = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
    
    above_thresh = v_sliced > threshold_s2
    has_s2 = np.any(above_thresh, axis=1)
    
    # Leftmost (start) and rightmost (end) crossings
    start_idx = np.argmax(above_thresh, axis=1)
    end_idx = (v_sliced.shape[1] - 1) - np.argmax(np.fliplr(above_thresh), axis=1)
    
    starts = np.full(wf.nframes, np.nan, dtype=np.float32)
    ends = np.full(wf.nframes, np.nan, dtype=np.float32)
    
    starts[has_s2] = t_sliced[start_idx[has_s2]]
    ends[has_s2] = t_sliced[end_idx[has_s2]]
    return starts, ends