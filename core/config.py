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
    max_area_s1: float = 1e5          # (mV·µs) -- max allowed area before S1 (reject if exceeded)
    max_area_s2: float = 1e5          # (mV·µs) -- max allowed area in S2 window (reject if exceeded)
    min_xray_area: float = 0.2        # (mV·µs) -- min required X-ray area (reject if below)
    min_s2_sep: float = 1.0           # (µs)   -- min required separation before S2 window
    min_s1_sep: float = 0.5           # (µs)   -- min required separation after S1
    n_pedestal: int = 200             # number of pre-trigger samples for pedestal subtraction
    ma_window: int = 10               # moving average window length (samples)
    dt: float = 2e-4                  # integration timestep (µs)
    integrator: Callable[[PMTWaveform, float], np.ndarray] = field(default_factory=_default_integrator)


# -------------------------------
# Alpha Spectrum Peak Definitions
# -------------------------------
# Literature energies and fitting windows for Th-232 decay chain alphas

# Main peaks for preliminary fitting (5 peaks in SCA scale)
ALPHA_PEAK_DEFINITIONS = [
    {'name': 'Th228', 'position': 4.5, 'window': (4.0, 4.7), 'sigma_init': 0.15, 'ref_energy': 5.42315},
    {'name': 'Ra224', 'position': 4.8, 'window': (4.65, 5.1), 'sigma_init': 0.15, 'ref_energy': 5.68537},
    {'name': 'Rn220', 'position': 5.4, 'window': (5.0, 5.5), 'sigma_init': 0.15, 'ref_energy': 6.40484},
    {'name': 'Po216', 'position': 5.9, 'window': (5.6, 6.1), 'sigma_init': 0.15, 'ref_energy': 6.90628},
    {'name': 'Po212', 'position': 7.5, 'window': (6.3, 8.0), 'sigma_init': 0.15, 'ref_energy': 8.785},
]

# Satellite peaks for hierarchical fitting (4 additional peaks)
ALPHA_SATELLITE_DEFINITIONS = [
    {'name': 'Th228_sat', 'ref_energy': 5.34036, 'branching_ratio': 0.385},
    {'name': 'Ra224_sat', 'ref_energy': 5.44860, 'branching_ratio': 0.054},
    {'name': 'Bi212', 'ref_energy': 6.207},  # Independent main peak
    {'name': 'Bi212_sat', 'ref_energy': 6.090, 'branching_ratio': 0.389},
]


@dataclass(frozen=True)
class AlphaCalibrationConfig:
    """Configuration for alpha spectrum calibration pipeline."""
    files_per_chunk: int = 10          # Waveform files per energy map chunk (10-100 typical)
    fmt: str = "8b"                    # Binary format: "8b" (accurate) or "6b" (compact)
    scale: float = 0.1                 # For "6b" format: keV per LSB
    pattern: str = "*Ch4.wfm"          # Glob pattern for alpha channel files
    nbins: int = 120                   # Number of histogram bins for energy spectra
    n_sigma: float = 2.0               # Number of sigmas for isotope range definition
    use_quadratic: bool = True         # Use quadratic (vs linear) energy calibration

