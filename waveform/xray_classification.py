"""
X-ray event classification at the frame level.

This module provides pure functions for classifying X-ray events in individual
waveform frames. These functions check quality criteria and integrate accepted events.

All functions operate on single PMTWaveform objects and have no dependencies on
Set or Run-level structures.
"""

from typing import Optional
import numpy as np

from RaTag.core.datatypes import PMTWaveform
from RaTag.core.config import XRayConfig
from RaTag.waveform.preprocessing import moving_average, threshold_clip
from RaTag.waveform.s1s2_detection import extract_window


def excessive_s2_check(wf: PMTWaveform, 
                       s2_start: float, 
                       bs_threshold: float, 
                       max_area_s2: float = 1e5, 
                       debug: bool = False) -> tuple[bool, str]:
    """
    Check if waveform baseline before S2 is acceptable.
    
    Rejects frames where the integrated area in the S2 window exceeds the maximum
    allowed value, which typically indicates overlapping S2 signals or noise.
    
    Args:
        wf: PMTWaveform in µs/mV (after pedestal subtraction)
        s2_start: Expected S2 start time [µs]
        bs_threshold: Baseline threshold [mV]
        max_area_s2: Maximum allowed area in S2 window [mV·µs]
        debug: If True, print diagnostic information
        
    Returns:
        (passed, reason) - True if check passed, with descriptive reason
    """
    mask = (wf.t >= s2_start)
    Vs2 = wf.v[mask]
    
    area_above_threshold = Vs2[Vs2 > bs_threshold].sum()
    
    if area_above_threshold > max_area_s2:
        if debug:
            print(f'Area above baseline before S2: {area_above_threshold:.2e} (max allowed {max_area_s2})')
        return False, "excessive S2 signal above baseline"
    
    return True, "ok"


def separation_check(wf: PMTWaveform, 
                     t_s1: float, 
                     s2_start: float,
                     min_s2_sep: float, 
                     min_s1_sep: float) -> tuple[bool, str]:
    """
    Check if waveform has sufficient separation between S1 and X-ray signal.
    
    Verifies that the X-ray signal is isolated from both S1 (before) and S2 (after),
    ensuring clean integration without contamination from other signals.
    
    Args:
        wf: PMTWaveform in µs/mV (preprocessed: windowed, smoothed, clipped)
        t_s1: S1 peak time [µs]
        s2_start: Expected S2 start time [µs]
        min_s2_sep: Minimum required separation before S2 [µs]
        min_s1_sep: Minimum required separation after S1 [µs]
        
    Returns:
        (passed, reason) - True if check passed, with descriptive reason
    """
    t_thresh = wf.t[wf.v > 0] if np.any(wf.v > 0) else None
    
    if t_thresh is None:
        return False, "no signal above baseline"
    
    t_pre_s2 = s2_start - t_thresh[-1]
    t_post_s1 = t_thresh[0] - t_s1
    
    if not (t_pre_s2 > min_s2_sep and t_post_s1 > min_s1_sep):
        return False, f"insufficient separation (t_pre_s2={t_pre_s2:.2f}, t_post_s1={t_post_s1:.2f})"
    
    return True, "ok"


def classify_xray_in_frame(wf: PMTWaveform,
                          t_s1: float,
                          s2_start: float,
                          xray_config: XRayConfig,
                          debug: bool = False) -> tuple[bool, Optional[str], Optional[float]]:
    """
    Classify a single frame as X-ray event and integrate if accepted.
    
    This is the core classification pipeline for X-ray events:
    1. Check for excessive S2 signal (reject if too much baseline activity)
    2. Preprocess: extract window, smooth, clip to threshold
    3. Check signal separation (reject if too close to S1 or S2)
    4. Integrate accepted events
    
    Args:
        wf: PMTWaveform (already in µs/mV with pedestal subtracted)
        t_s1: S1 time [µs]
        s2_start: Expected S2 start time [µs]
        xray_config: Configuration parameters (thresholds, window sizes, etc.)
        debug: Enable debug output
        
    Returns:
        (accepted, rejection_reason, area) where:
        - accepted: True if event passes all criteria
        - rejection_reason: "excessive_s2", "insufficient_separation", or None if accepted
        - area: Integrated X-ray area [mV·µs], or None if rejected
        
    Example:
        >>> config = XRayConfig(bs_threshold=0.5, min_s2_sep=1.0, min_s1_sep=0.5)
        >>> wf_processed = subtract_pedestal(wf, n_points=200)
        >>> accepted, reason, area = classify_xray_in_frame(wf_processed, -0.7, 5.2, config)
        >>> if accepted:
        ...     print(f"X-ray area: {area:.2f} mV·µs")
    """
    # Check 1: Excessive S2 baseline
    passed, reason = excessive_s2_check(wf, s2_start, 
                                       xray_config.bs_threshold,
                                       xray_config.max_area_s2, 
                                       debug=debug)
    if not passed:
        return False, "excessive_s2", None
    
    # Preprocessing for integration
    wf = extract_window(wf, t_s1, s2_start)
    wf = moving_average(wf, window=xray_config.ma_window)
    wf = threshold_clip(wf, threshold=xray_config.bs_threshold)
    
    # Check 2: Signal separation
    passed, reason = separation_check(wf, t_s1, s2_start,
                                     xray_config.min_s2_sep, 
                                     xray_config.min_s1_sep)
    if not passed:
        return False, "insufficient_separation", None
    
    # Integration
    area = float(xray_config.integrator(wf, xray_config.dt))
    
    return True, None, area
