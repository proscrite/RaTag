"""Window extraction and peak detection."""

from dataclasses import replace

import numpy as np # type: ignore
from typing import Optional
from core.datatypes import PMTWaveform
from .preprocessing import moving_average, subtract_pedestal, threshold_clip

# ----------------------------------
# --- Generic Window extraction ----
# ----------------------------------
def extract_window(wf: PMTWaveform, t_start: float, t_end: float) -> PMTWaveform:
    """Extract time window from waveform, handling both FastFrame and single frame formats."""
    mask = (wf.t >= t_start) & (wf.t <= t_end)
    if wf.ff:
        v = wf.v[:, mask]  # Apply mask to all frames along time axis
    else:
        v = wf.v[mask]
    return replace(wf, t=wf.t[mask], v=v)


# ============================================================================
# FRAME-LEVEL S1 DETECTION
# ============================================================================

def detect_s1_in_frame(wf: PMTWaveform, threshold_s1: float = 1.0) -> Optional[float]:
    """
    Detect S1 peak time in a single frame (last peak before t=0).
    
    Args:
        wf: Single frame waveform (ff=False)
        threshold: Minimum peak height in mV
        
    Returns:
        S1 peak time in µs, or None if not found
    """
    # Preprocess
    wf = subtract_pedestal(wf, n_points=200)

    mask = wf.t < 0
    V_before_zero = wf.v[mask]
    t_before_zero = wf.t[mask]
    
    if len(V_before_zero) == 0:
        return None
    
    # Find indices where signal is above threshold
    above_threshold = V_before_zero > threshold_s1
    
    if not np.any(above_threshold):
        return None
    
    # Get the last (rightmost) peak above threshold
    indices_above = np.where(above_threshold)[0]
    last_peak_idx = indices_above[-1]
    
    return t_before_zero[last_peak_idx]

# ============================================================================
# FRAME-LEVEL S2 DETECTION
# ============================================================================

def detect_s2_in_frame(wf: PMTWaveform,
                       t_s1: float,
                       t_drift: float,
                       threshold_s2: float = 0.8,
                       window_size: int = 9,
                       threshold_bs: float = 0.02) -> Optional[tuple[float, float]]:
    """
    Detect S2 boundaries in a single frame.
    
    Args:
        wf: Single frame waveform (ff=False)
        t_s1: S1 time in µs
        t_drift: Expected drift time in µs
        threshold_s2: S2 detection threshold in mV
        window_size: Moving average window
        threshold_bs: Baseline threshold in mV
        
    Returns:
        (s2_start, s2_end) in µs, or None if not found
    """
    # Preprocess
    wf = moving_average(wf, window=window_size)
    wf = threshold_clip(wf, threshold=threshold_bs)
    
    # Look for S2 after drift time
    mask = wf.t > (t_s1 + t_drift * 0.3)
    s2_window = wf.v[mask]
    
    # Find boundaries
    positive_mask = s2_window > threshold_s2
    
    if not np.any(positive_mask):
        return None
    
    s2_start = wf.t[mask][positive_mask][0]
    s2_end = wf.t[mask][positive_mask][-1]
    
    return s2_start, s2_end

