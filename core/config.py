import numpy as np
from dataclasses import dataclass, field
from typing import Callable
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

def _default_integrator():
    """Lazy import to avoid circular dependency."""
    from RaTag.waveform.integration import integrate_trapz
    return integrate_trapz


@dataclass(frozen=True)
class IntegrationConfig:
    bs_threshold: float = 0.8          # (mV)  -- min baseline voltage to consider
    max_area_s2: float = 1e5          # (mV·µs) -- max area for S2 window
    min_s2_sep: float = 1.0           # (µs)   -- min separation before S2
    min_s1_sep: float = 1.0           # (µs)   -- min separation after S1
    n_pedestal: int = 2000            # number of pre-trigger samples for pedestal
    ma_window: int = 9                # moving average window length (samples)
    dt: float = 2e-4                  # (µs) integration timestep: 0.2 ns = 0.0002 µs for 5 GS/s
    integrator: Callable[[PMTWaveform, float], np.ndarray] = field(default_factory=_default_integrator)


@dataclass(frozen=True)
class FitConfig:
    bin_cuts: tuple[float, float] = (0, 4)
    nbins: int = 100
    exclude_index: int = 1  # Deprecated - kept for backward compatibility
    bg_threshold: float = 0.3  # Background fraction threshold for two-stage fitting
    bg_cutoff: float = 1.0     # Upper limit for background fitting (mV·µs)
    n_sigma: float = 2.5       # Sigmas above background for signal region
    upper_limit: float = 5.0   # Upper limit for signal fitting (mV·µs)


@dataclass(frozen=True)
class XRayConfig:
    """Configuration for X-ray event classification and integration."""
    bs_threshold: float = 0.5          # (mV)  -- baseline threshold for signal detection
    max_area_s2: float = 1e5          # (mV·µs) -- max allowed area in S2 window (reject if exceeded)
    min_s2_sep: float = 1.0           # (µs)   -- min required separation before S2 window
    min_s1_sep: float = 0.5           # (µs)   -- min required separation after S1
    n_pedestal: int = 200             # number of pre-trigger samples for pedestal subtraction
    ma_window: int = 10               # moving average window length (samples)
    dt: float = 2e-4                  # integration timestep (µs)
    integrator: Callable[[PMTWaveform, float], np.ndarray] = field(default_factory=_default_integrator)

