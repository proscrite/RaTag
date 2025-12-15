"""
Activity analysis for alpha-emitting isotopes from time-stamped energy maps.

This module provides functions for:
1. Time-stamped spectrum loading (per-set, not aggregated)
2. Activity calculation from count rates (with efficiency/branching corrections)
3. Decay curve fitting (exponential decay model)
4. Time-series visualization

Data flow:
    .wfm files (with timestamps) → energy_maps/*.bin → time-stamped spectra → activity vs time

Key use case: Track Ra-224 (T½ = 3.6 days) decay over multiple measurement sets (~2 days)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import lmfit
from scipy import stats

from RaTag.alphas.spectrum_fitting import (
    SpectrumData,
    IsotopeRange,
    load_spectrum_from_energy_maps
)


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class TimeStampedSpectrum:
    """
    Spectrum with acquisition timestamp.
    
    Attributes:
        spectrum: SpectrumData object
        timestamp: Unix timestamp (seconds since epoch) from .wfm file mtime
        set_name: Name of measurement set (e.g., "FieldScan_Gate0050_Anode1950")
        acquisition_time: Human-readable datetime
    """
    spectrum: SpectrumData
    timestamp: float
    set_name: str
    acquisition_time: datetime
    
    def count_in_range(self, E_min: float, E_max: float) -> int:
        """Count events in energy range [E_min, E_max] MeV."""
        mask = (self.spectrum.energies >= E_min) & (self.spectrum.energies <= E_max)
        return int(np.sum(mask))


@dataclass(frozen=True)
class ActivityMeasurement:
    """
    Activity measurement for single isotope at single time point.
    
    Attributes:
        isotope: Isotope name (e.g., "Ra224")
        timestamp: Unix timestamp (seconds since epoch)
        acquisition_time: Human-readable datetime
        counts: Raw counts in isotope energy range
        live_time: Measurement duration [hours]
        count_rate: Count rate [counts/hour]
        count_rate_err: Poisson uncertainty on count rate [counts/hour]
        activity: Activity [Bq] (if efficiency/branching provided)
        activity_err: Uncertainty on activity [Bq]
        set_name: Name of measurement set
    """
    isotope: str
    timestamp: float
    acquisition_time: datetime
    counts: int
    live_time: float
    count_rate: float
    count_rate_err: float
    activity: Optional[float] = None
    activity_err: Optional[float] = None
    set_name: str = ""


@dataclass(frozen=True)
class DecayFitResult:
    """
    Exponential decay fit result.
    
    Model: A(t) = A0 * exp(-λ*t) where λ = ln(2)/T½
    
    Attributes:
        isotope: Isotope name
        A0: Initial activity [Bq]
        A0_err: Uncertainty on A0 [Bq]
        lambda_decay: Decay constant [1/hour]
        lambda_err: Uncertainty on λ [1/hour]
        half_life: Fitted half-life [hours]
        half_life_err: Uncertainty on T½ [hours]
        half_life_literature: Literature value [hours]
        chi2_reduced: Reduced chi-squared
        fit_result: lmfit ModelResult for detailed diagnostics
    """
    isotope: str
    A0: float
    A0_err: float
    lambda_decay: float
    lambda_err: float
    half_life: float
    half_life_err: float
    half_life_literature: float
    chi2_reduced: float
    fit_result: lmfit.model.ModelResult


# ============================================================================
# TIMESTAMP EXTRACTION
# ============================================================================

def get_wfm_timestamps(wfm_dir: Path) -> Tuple[float, float, int]:
    """
    Extract timestamp range from .wfm files in directory.
    
    Uses modification time (mtime) which preserves original acquisition time
    even after copying files between disks.
    
    Args:
        wfm_dir: Directory containing .wfm files
        
    Returns:
        (t_start, t_end, n_files) where:
        - t_start: Earliest file mtime [Unix timestamp]
        - t_end: Latest file mtime [Unix timestamp]
        - n_files: Number of .wfm files found
        
    Raises:
        ValueError: If no .wfm files found
    """
    wfm_files = list(wfm_dir.glob("*.wfm"))
    
    if not wfm_files:
        raise ValueError(f"No .wfm files found in {wfm_dir}")
    
    # Extract modification times (preserves original creation time)
    mtimes = [f.stat().st_mtime for f in wfm_files]
    
    return min(mtimes), max(mtimes), len(wfm_files)


def compute_live_time(t_start: float, t_end: float) -> float:
    """
    Compute live time from timestamp range.
    
    Args:
        t_start: Start timestamp [Unix time]
        t_end: End timestamp [Unix time]
        
    Returns:
        Live time in hours
    """
    return (t_end - t_start) / 3600.0  # Convert seconds to hours


# ============================================================================
# TIME-STAMPED SPECTRUM LOADING
# ============================================================================

def load_timestamped_spectrum(run,
                              set_index: int,
                              energy_range: Tuple[float, float]) -> TimeStampedSpectrum:
    """
    Load spectrum from single set with timestamp from .wfm files.
    
    Args:
        run: Run object with initialized sets
        set_index: Index of set to load (0-based)
        energy_range: (E_min, E_max) ROI [keV]
        
    Returns:
        TimeStampedSpectrum with acquisition timestamp
        
    Example:
        >>> spectrum_t0 = load_timestamped_spectrum(run, set_index=0, energy_range=(4000, 8000))
        >>> print(spectrum_t0.acquisition_time)
        2025-11-04 11:43:46
    """
    set_pmt = run.sets[set_index]
    
    # Load spectrum from energy maps
    energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
    spectrum = load_spectrum_from_energy_maps(str(energy_maps_dir), energy_range)
    
    # Extract timestamp from .wfm files
    wfm_dir = set_pmt.source_dir
    t_start, t_end, n_files = get_wfm_timestamps(wfm_dir)
    
    # Use midpoint as representative timestamp
    timestamp = (t_start + t_end) / 2
    acquisition_time = datetime.fromtimestamp(timestamp)
    
    return TimeStampedSpectrum(spectrum=spectrum,
                               timestamp=timestamp,
                               set_name=set_pmt.source_dir.name,
                               acquisition_time=acquisition_time)


def load_all_timestamped_spectra(run,
                                 energy_range: Tuple[float, float]) -> List[TimeStampedSpectrum]:
    """
    Load time-stamped spectra from all sets in run.
    
    Args:
        run: Run object with initialized sets
        energy_range: (E_min, E_max) ROI [keV]
        
    Returns:
        List of TimeStampedSpectrum, sorted by timestamp
        
    Example:
        >>> spectra = load_all_timestamped_spectra(run, energy_range=(4000, 8000))
        >>> print(f"Loaded {len(spectra)} time points")
        >>> for s in spectra:
        ...     print(f"{s.set_name}: {s.acquisition_time}")
    """
    spectra = []
    
    for i in range(len(run.sets)):
        try:
            spectrum = load_timestamped_spectrum(run, i, energy_range)
            spectra.append(spectrum)
        except Exception as e:
            print(f"Warning: Failed to load set {i}: {e}")
    
    # Sort by timestamp
    spectra.sort(key=lambda s: s.timestamp)
    
    return spectra


# ============================================================================
# ACTIVITY CALCULATION
# ============================================================================

def compute_activity_from_counts(counts: int,
                                 live_time: float,
                                 efficiency: float = 1.0,
                                 branching_ratio: float = 1.0) -> Tuple[float, float]:
    """
    Compute activity from raw counts with Poisson uncertainty.
    
    Activity [Bq] = counts / (live_time [s] × efficiency × branching_ratio)
    
    Args:
        counts: Number of events in isotope energy range
        live_time: Measurement duration [hours]
        efficiency: Detection efficiency (0-1)
        branching_ratio: Alpha branching ratio (0-1)
        
    Returns:
        (activity, activity_err) in Bq
    """
    live_time_sec = live_time * 3600.0  # Convert hours to seconds
    
    # Activity and Poisson uncertainty
    activity = counts / (live_time_sec * efficiency * branching_ratio)
    activity_err = np.sqrt(counts) / (live_time_sec * efficiency * branching_ratio)
    
    return activity, activity_err


def measure_activity(timestamped_spectrum: TimeStampedSpectrum,
                    isotope_range: IsotopeRange,
                    efficiency: float = 1.0,
                    branching_ratio: float = 1.0) -> ActivityMeasurement:
    """
    Measure activity for single isotope at single time point.
    
    Args:
        timestamped_spectrum: Spectrum with timestamp
        isotope_range: Energy range for isotope [MeV]
        efficiency: Detection efficiency (default: 1.0 for relative activity)
        branching_ratio: Alpha branching ratio (default: 1.0)
        
    Returns:
        ActivityMeasurement with count rate and activity
        
    Example:
        >>> # Ra224: branching ratio 94.9% for main alpha line
        >>> activity = measure_activity(spectrum_t0, ra224_range, 
        ...                            efficiency=0.85, branching_ratio=0.949)
        >>> print(f"Activity: {activity.activity:.1f} ± {activity.activity_err:.1f} Bq")
    """
    # Count events in isotope range
    counts = timestamped_spectrum.count_in_range(isotope_range.E_min, isotope_range.E_max)
    
    # Compute live time from .wfm timestamps
    wfm_dir = Path(timestamped_spectrum.spectrum.source).parent.parent / timestamped_spectrum.set_name
    t_start, t_end, _ = get_wfm_timestamps(wfm_dir)
    live_time = compute_live_time(t_start, t_end)
    
    # Count rate [counts/hour]
    count_rate = counts / live_time
    count_rate_err = np.sqrt(counts) / live_time
    
    # Activity [Bq]
    activity, activity_err = compute_activity_from_counts(counts, live_time, efficiency, branching_ratio)
    
    return ActivityMeasurement(isotope=isotope_range.name,
                               timestamp=timestamped_spectrum.timestamp,
                               acquisition_time=timestamped_spectrum.acquisition_time,
                               counts=counts,
                               live_time=live_time,
                               count_rate=count_rate,
                               count_rate_err=count_rate_err,
                               activity=activity,
                               activity_err=activity_err,
                               set_name=timestamped_spectrum.set_name)


def measure_activity_timeseries(timestamped_spectra: List[TimeStampedSpectrum],
                                isotope_range: IsotopeRange,
                                efficiency: float = 1.0,
                                branching_ratio: float = 1.0) -> List[ActivityMeasurement]:
    """
    Measure activity time series for single isotope.
    
    Args:
        timestamped_spectra: List of time-stamped spectra (sorted by time)
        isotope_range: Energy range for isotope
        efficiency: Detection efficiency
        branching_ratio: Alpha branching ratio
        
    Returns:
        List of ActivityMeasurement, one per time point
    """
    measurements = []
    
    for spectrum in timestamped_spectra:
        measurement = measure_activity(spectrum, isotope_range, efficiency, branching_ratio)
        measurements.append(measurement)
    
    return measurements


# ============================================================================
# DECAY CURVE FITTING
# ============================================================================

def fit_exponential_decay(measurements: List[ActivityMeasurement],
                         half_life_literature: float) -> DecayFitResult:
    """
    Fit exponential decay to activity measurements.
    
    Model: A(t) = A0 * exp(-λ*t)
    where λ = ln(2) / T½
    
    Args:
        measurements: List of ActivityMeasurement (must have activity and activity_err)
        half_life_literature: Literature half-life [hours] for comparison
        
    Returns:
        DecayFitResult with fitted parameters
        
    Example:
        >>> # Ra-224: T½ = 3.6 days = 86.4 hours
        >>> decay_fit = fit_exponential_decay(activity_measurements, half_life_literature=86.4)
        >>> print(f"Fitted T½: {decay_fit.half_life:.1f} ± {decay_fit.half_life_err:.1f} hours")
        >>> print(f"Literature: {decay_fit.half_life_literature:.1f} hours")
    """
    # Extract data
    t = np.array([m.timestamp for m in measurements])
    t = (t - t[0]) / 3600.0  # Convert to hours since first measurement
    
    A = np.array([m.activity for m in measurements])
    A_err = np.array([m.activity_err for m in measurements])
    
    # Define exponential decay model
    def decay_model(t, A0, lambda_decay):
        return A0 * np.exp(-lambda_decay * t)
    
    # Fit with lmfit
    model = lmfit.Model(decay_model)
    params = model.make_params()
    
    # Initial guess: λ from literature, A0 from first data point
    lambda_init = np.log(2) / half_life_literature
    params['A0'].set(value=A[0], min=0)
    params['lambda_decay'].set(value=lambda_init, min=0)
    
    # Weighted fit using measurement uncertainties
    weights = 1.0 / A_err
    result = model.fit(A, params=params, t=t, weights=weights)
    
    # Extract fitted parameters
    A0 = result.params['A0'].value
    A0_err = result.params['A0'].stderr if result.params['A0'].stderr else 0.0
    
    lambda_fit = result.params['lambda_decay'].value
    lambda_err = result.params['lambda_decay'].stderr if result.params['lambda_decay'].stderr else 0.0
    
    # Compute fitted half-life
    half_life_fit = np.log(2) / lambda_fit
    half_life_err = (np.log(2) / lambda_fit**2) * lambda_err  # Error propagation
    
    return DecayFitResult(
        isotope=measurements[0].isotope,
        A0=A0,
        A0_err=A0_err,
        lambda_decay=lambda_fit,
        lambda_err=lambda_err,
        half_life=half_life_fit,
        half_life_err=half_life_err,
        half_life_literature=half_life_literature,
        chi2_reduced=result.redchi,
        fit_result=result
    )


# ============================================================================
# REFERENCE VALUES
# ============================================================================

# Literature half-lives [hours]
HALF_LIVES = {
    'Th228': 1.9 * 365.25 * 24,  # 1.9 years
    'Ra224': 3.66 * 24,           # 3.66 days
    'Rn220': 55.6 / 3600,         # 55.6 seconds
    'Po216': 0.145 / 3600,        # 0.145 seconds
    'Po212': 0.299e-6 / 3600,     # 0.299 microseconds
}

# Alpha branching ratios (main line)
BRANCHING_RATIOS = {
    'Th228': 0.717,   # 5.423 MeV line (71.7%)
    'Ra224': 0.949,   # 5.685 MeV line (94.9%)
    'Rn220': 0.999,   # 6.405 MeV line (99.9%)
    'Po216': 1.000,   # 6.906 MeV line (100%)
    'Po212': 1.000,   # 8.785 MeV line (100%)
}
