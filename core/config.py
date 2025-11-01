import numpy as np
from dataclasses import dataclass
from typing import Callable
from waveform.integration import integrate_trapz
from .datatypes import PMTWaveform

# -------------------------------
# General analysis thresholds
# -------------------------------
BASELINE_RMS_MAX = 2.0
AMPLITUDE_MIN = 10.0
AMPLITUDE_MAX = 500.0
PEAK_TIME_WINDOW = (100, 200)

# -------------------------------
# Gas / transport parameters
# -------------------------------
# Fit parameters for drift velocity model (Xe @ 2 bar, say)
DRIFT_VELOCITY_PARAMS = {
    "p0": 0.92809704,
    "p1": 17.17333489,
    "p2": 0.51193002,
    "p3": 0.30107278,
}

# Optionally: define common drift fields to evaluate
DRIFT_FIELDS = [35, 50, 70, 107, 142, 178, 214, 250, 285, 321, 357, 428]  # V/cm

@dataclass(frozen=True)
class IntegrationConfig:
    bs_threshold: float = 0.8          # (mV)  -- min baseline voltage to consider
    max_area_s2: float = 1e5          # (mV·µs) -- max area for S2 window
    min_s2_sep: float = 1.0           # (µs)   -- min separation before S2
    min_s1_sep: float = 1.0           # (µs)   -- min separation after S1
    n_pedestal: int = 2000            # number of pre-trigger samples for pedestal
    ma_window: int = 9                # moving average window length (samples)
    dt: float = 2e-4                  # default unless overridden by wf spacing
    integrator: Callable[[PMTWaveform, float], np.ndarray] = integrate_trapz


@dataclass(frozen=True)
class FitConfig:
    bin_cuts: tuple[float, float] = (0, 4)
    nbins: int = 100
    exclude_index: int = 1
