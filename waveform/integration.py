import numpy as np # type: ignore

from RaTag.core.datatypes import PMTWaveform
from RaTag.core.config import IntegrationConfig
from RaTag.waveform.preprocessing import subtract_pedestal, moving_average, threshold_clip
from RaTag.waveform.s1s2_detection import extract_window


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
        return np.trapezoid(wf.v, dx=dt, axis=1)  # Integrate each frame separately
    return np.array([np.trapezoid(wf.v, dx=dt)])  # Return single-element array


def integrate_s2_in_frame(wf: PMTWaveform,
                          s2_start: float,
                          s2_end: float,
                          config: IntegrationConfig) -> float:
    """
    Integrate S2 signal in a single frame.
    
    Args:
        wf: Single frame waveform (ff=False)
        s2_start: S2 window start (µs)
        s2_end: S2 window end (µs)
        config: Integration configuration
        
    Returns:
        Integrated S2 area (mV·µs)
    """
    # Preprocess
    wf = subtract_pedestal(wf, n_points=config.n_pedestal)
    wf = moving_average(wf, window=config.ma_window)
    wf = threshold_clip(wf, threshold=config.bs_threshold)
    
    # Extract S2 window and integrate
    wf_s2 = extract_window(wf, s2_start, s2_end)
    areas = config.integrator(wf_s2, config.dt)
    
    return float(areas[-1])  # Last element contains total area
