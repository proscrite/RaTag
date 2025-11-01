import numpy as np # type: ignore

from core.datatypes import PMTWaveform


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
