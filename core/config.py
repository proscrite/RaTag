import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
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
# Fit parameters for drift velocity model
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
    force: bool = False
    max_frames: Optional[int] = None  # Maximum number of frames to process (None = all)
    bs_threshold: float = 0.8          # (mV)  -- min baseline voltage to consider
    max_area_s2: float = 1e5          # (mV·µs) -- max area for S2 window
    min_s2_sep: float = 1.0           # (µs)   -- min separation before S2
    min_s1_sep: float = 1.0           # (µs)   -- min separation after S1
    n_pedestal: int = 2000            # number of pre-trigger samples for pedestal
    n_sigma_start: float = 1.0        # Number of sigmas to extend start of S2 window
    n_sigma_end: float = 0.5          # Number of sigmas to extend end of S2 window
    ma_window: int = 9                # moving average window length (samples)
    pre_time: float = 0.5             # µs - Time before S1 for integration window
    post_time: float = 0.5            # µs - Time after S1 for integration
    dt: float = 2e-4                  # (µs) integration timestep: 0.2 ns = 0.0002 µs for 5 GS/s


@dataclass(frozen=True)
class FitConfig:
    force: bool = False
    bin_cuts: tuple[float, float] = (0, 50)
    nbins: int = 100
    bg_threshold: float = 0.3  # Background fraction threshold for two-stage fitting
    bg_cutoff: float = 2.0     # Upper limit for background fitting (mV·µs)
    n_sigma: float = 2.5       # Sigmas above background for signal region
    smooth: int = 4            # Smoothing window for histogram counts (bins)
    max_lower_bound: float = 0.5 # Limit for lower bound of signal fitting range (mV·µs)

@dataclass(frozen=True)
class TimingConfig:
    """
    Configuration parameters for THGEM S1 and S2 timing extraction.
    """
    force: bool = False
    # S1 Search Parameters
    max_frames: int = None        # Maximum number of frames to consider for S1 search (None = all)
    s1_t_min: float = -4.0           # (µs) Start of empirical S1 search window
    s1_t_max: float = -2.5           # (µs) End of empirical S1 search window
    s1_v_min: float = 3.0            # (mV) Minimum height for valid S1
    s1_v_max: float = 15.0           # (mV) Maximum height for valid S1
    s1_max_area: float = 0.15        # (mV*µs) Max integrated area to reject alpha tails
    
    # S2 Search Parameters
    max_frames_s2: int = None        # Maximum number of frames to consider for S2 search (None = all)
    s2_margin: float = 0.9           # (µs) Multiplier for drift time to set S2 window
    s2_threshold: float = 230        # (mV) Maximum for frame clipping detection
    s2_fraction: float = 0.05        # Fractional threshold for S2 boundaries (e.g., 5% of peak)
    s2_min_area: float = 0.5         # (mV*µs) Minimum S2 area to consider valid
    s2_max_area: float = 50          # (mV*µs) Maximum S2 area to consider valid
    s2_min_width: float = 0.2        # (µs) Minimum S2 width to consider valid
    s2_start_max: float = 0.5        # (µs) Maximum S2 start time
    s2_start_min: float = -2.0       # (µs) Minimum S2 start time

    # Preprocessing Parameters
    n_pedestal: int = 200            # Samples for pedestal subtraction
    bs_threshold: float = 0.02       # (mV) Initial noise clipping threshold
    s1_window_ma: int = 10           # Samples (~2 ns) strictly for defeating digitizer quantization
    s2_window_ma: int = 1000          # Samples (~20 ns) for macroscopic S2 envelope tracking

@dataclass(frozen=True)
class XRayConfig:
    """Configuration for X-ray event classification and integration."""
    force: bool = False
    max_frames: Optional[int] = None  # Maximum number of frames to process (None = all)
    bs_threshold: float = 0.5          # (mV)  -- baseline threshold for signal detection
    max_area_s1: float = 100          # (mV·µs) -- max allowed area before S1 (reject if exceeded)
    max_area_s2: float = 100          # (mV·µs) -- max allowed area in S2 window (reject if exceeded)
    min_xray_area: float = 0.5        # (mV·µs) -- min required X-ray area (reject if below)
    min_s2_sep: float = 1.0           # (µs)   -- min required separation before S2 window
    min_s1_sep: float = 0.5           # (µs)   -- min required separation after S1
    n_pedestal: int = 200             # number of pre-trigger samples for pedestal subtraction
    ma_window: int = 10               # moving average window length (samples)
    dt: float = 2e-4                  # integration timestep (µs)
    max_v_clip: float = 150.0             # (mV) -- maximum voltage to consider for clipping detection



# -------------------------------
# Alpha Spectrum Peak Definitions
# -------------------------------
# Literature energies and fitting windows for Th-232 decay chain alphas

# Main peaks for preliminary fitting (5 peaks in SCA scale)
ALPHA_PEAK_DEFINITIONS = [
    {'name': 'Th228', 'position': 4.3, 'window': (3.5, 4.4), 'sigma_init': 0.05, 'ref_energy': 5.42315},
    {'name': 'Ra224', 'position': 4.6, 'window': (4.4, 4.7), 'sigma_init': 0.05, 'ref_energy': 5.68537},
    {'name': 'Bi212', 'position': 4.9, 'window': (4.7, 4.95), 'sigma_init': 0.05, 'ref_energy': 6.207},
    {'name': 'Rn220', 'position': 5.1, 'window': (4.95, 5.2), 'sigma_init': 0.05, 'ref_energy': 6.40484},
    {'name': 'Po216', 'position': 5.9, 'window': (5.2, 5.6), 'sigma_init': 0.05, 'ref_energy': 6.90628},
    {'name': 'Po212', 'position': 7.2, 'window': (6.7, 8.0), 'sigma_init': 0.07, 'ref_energy': 8.785},
]

# Satellite peaks for hierarchical fitting (4 additional peaks)
ALPHA_SATELLITE_DEFINITIONS = [
    {'name': 'Th228_sat', 'ref_energy': 5.34036, 'branching_ratio': 0.385},
    {'name': 'Ra224_sat', 'ref_energy': 5.44860, 'branching_ratio': 0.054},
    {'name': 'Bi212_sat', 'ref_energy': 6.090, 'branching_ratio': 0.389},
]


@dataclass(frozen=True)
class AlphaCalibrationConfig:
    """Configuration for alpha spectrum calibration pipeline."""
    force: bool = False
    max_frames: Optional[int] = None   # Number of frames to process (None = all)
    savgol_window: int = 501           # Savitzky-Golay window size (must be odd)
    pattern: str = "*Ch4.wfm"          # (Deprecated, now in bootstrap) Glob pattern for alpha channel files
    nbins: int = 120                   # Number of histogram bins for energy spectra
    n_sigma: float = 2.0               # Number of sigmas for isotope range definition
    use_quadratic: bool = True         # Use quadratic (vs linear) energy calibration
    energy_range: tuple[float, float] = (3.5, 8.2)  # Energy range for fitting (V)

