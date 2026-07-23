from __future__ import annotations
from dataclasses import dataclass, fields, field, replace
from pathlib import Path
from functools import lru_cache

from typing import Optional, Dict, Any, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import List as ListType

# -------------------------------#
# Dataclasses for waveforms    --#
# -------------------------------#

@dataclass
class Waveform:
    """Generic waveform with common properties."""
    t: np.ndarray          # time axis
    v: np.ndarray          # voltage (signal)
    source: Optional[str]  # filename or run ID (for traceability)
    
    # -- FastFrame tracing info --
    frame_idx: int = 0
    file_seq: int = 0
    
    # -- FastFrame properties --
    ff: bool = False  # whether loaded from FastFrames
    nframes: int = 1
    
    @property
    def uid(self) -> int:
        return self.file_seq * 64 + self.frame_idx
    
    @property
    def uids(self) -> np.ndarray:
        """Return array of UIDs for each frame (for FastFrame) or single UID for single-frame."""
        if self.ff and self.nframes > 1:
            return  self.file_seq * 64 + np.arange(self.nframes)
        else:
            return np.array([self.uid])

    @property
    def dt(self) -> float:
        """Calculate sampling interval from time axis."""
        if len(self.t) < 2:
            raise ValueError("Time array must have at least 2 points to calculate dt.")
        return self.t[1] - self.t[0]
    

    def __len__(self):
        return len(self.t)
    
    def __sizeof__(self):
        if self.ff:
            return self.v.shape
    def __print__(self):
        return f"Waveform(source={self.source}, ff={self.ff}, nframes={self.nframes}, file_seq={self.file_seq}, frame_idx={self.frame_idx})"

@dataclass
class PMTWaveform(Waveform):
    """PMT-specific waveform with PMT calibration info."""
    gain: Optional[float] = None

    def to_photoelectrons(self, t_min: float, t_max: float) -> float:
        """Convert integrated charge to photoelectrons (if gain set)."""
        area = self.area(t_min, t_max)
        return area / self.gain if self.gain else area


@dataclass
class SiliconWaveform(Waveform):
    """PIN diode waveform."""
    sensitivity: Optional[float] = None

    def to_energy(self, t_min: float, t_max: float) -> float:
        """Convert signal to energy (if sensitivity set)."""
        area = self.area(t_min, t_max)
        return area * self.sensitivity if self.sensitivity else area


# -----------------------------------------------#
# Tracker for individual frames in FF files   ---#
# -----------------------------------------------#


# -------------------------------
# Dataclasses for measurement sets
# -------------------------------

@dataclass(repr=False)
class SetPmt:

    # --- Provenance / housekeeping ---
    source_dir: Path
    filenames: list[str]     # lazy list of filenames (not waveforms!)
    multiiso: bool = False   # Multi-isotope set
    target_isotope: Optional[str] = None  # e.g., "Th228"

    
    # --- FastFrame properties ---
    ff: bool = False                 # Whether this set uses FastFrame files
    nframes: int = 1                 # Frames per file (1 for single-frame, typically 49 for FastFrame)

    gate: Optional[float] = None
    anode: Optional[float] = None
    sampling_rate: Optional[float] = None

    # --- Physics context ---
    drift_field: Optional[float] = None        # V/cm
    EL_field: Optional[float] = None           # V/cm
    red_drift_field: Optional[float] = None    # reduced drift field (Td)
    red_EL_field: Optional[float] = None       # reduced EL field (Td)
    speed_drift: Optional[float] = None        # mm/us
    time_drift: Optional[float] = None         # us
    diffusion_coefficient: Optional[float] = None    # mm/√cm 

    # --- Baseline noise properties ---
    baseline_median: Optional[float] = None
    baseline_std: Optional[float] = None

    # --- S1/S2 Timing Metadata ---
    t_s1: Optional[float] = None
    t_s1_std: Optional[float] = None
    t_s2_start: Optional[float] = None
    t_s2_start_std: Optional[float] = None
    t_s2_end: Optional[float] = None
    t_s2_end_std: Optional[float] = None

    filtered_drift_time: Optional[bool] = None

    # --- Integration & Fit Metadata ---
    n_areas_recoil: Optional[int] = None
    area_s2_mean: Optional[float] = None
    area_s2_sigma: Optional[float] = None
    area_s2_ci95: Optional[float] = None
    area_s2_fit_success: Optional[bool] = None
    s2_background_bound: Optional[float] = None

    # --- Standalone Analysis Metadata ---
    # xray_metadata: Optional[XRayMetadata] = None

    def __len__(self):
        """Return number of files."""
        return len(self.filenames)
    
    @property
    def name(self) -> str:
        """Dynamically append isotope name if this is a spawned subset."""
        base = self.source_dir.name
        if self.multiiso and self.target_isotope:
            return f"{base}_{self.target_isotope}"
        return base

    @property
    def n_waveforms(self) -> int:
        """Total number of waveforms (frames) in the set."""
        return len(self.filenames) * self.nframes
    
    @property
    def n_files(self) -> int:
        """Number of files (alias for len())."""
        return len(self.filenames)
    
    def __str__(self):
        ff_str = f"FastFrame({self.nframes} frames/file)" if self.ff else "single-frame"
        
        overrides = {
            'filenames': f"<{self.n_files} files, {self.n_waveforms} waveforms, {ff_str}>"
        }
        return format_dataclass_state(self, overrides=overrides)

    def __repr__(self):        
        return self.__str__()


@dataclass
class SetAlpha:
    """Data class for alpha detector sets."""
    source_dir: Path
    filenames: list[str]
    
    multiiso: bool = False   # Multi-isotope set
    target_isotope: Optional[str] = None  # e.g., "Th228"

    # --- FastFrame properties ---
    ff: bool = False                 # Whether this set uses FastFrame files
    nframes: int = 1                 # Frames per file (1 for single-frame, typically 49 for FastFrame)

    # --- Physics context (not mandatory, for logging) ---
    gate: Optional[int] = None
    anode: Optional[int] = None
    sampling_rate: Optional[float] = None
    
    # Alpha-specific artifacts
    n_alpha_energies: Optional[int] = None

    calib_a: Optional[float] = None             # Linear calibration slope (keV/LSB)
    calib_b: Optional[float] = None
    calib_c: Optional[float] = None          # For quadratic calibration
    calib_order: Optional[int] = None           # 1 for linear, 2 for quadratic, etc.

    mean_energy_resolution: Optional[float] = None
    isotope_ranges_V: Optional[Dict] = None     # SCA scale e.g. {"Th228": (V_min, V_max), "Ra224": (V_min, V_max), ...}
    isotope_ranges_E: Optional[Dict] = None     # Energy scale e.g. {"Th228": (E_min, E_max), "Ra224": (E_min, E_max), ...}
    
    @property
    def name(self) -> str:
        """Dynamically append isotope name if this is a spawned subset."""
        base = self.source_dir.name
        if self.multiiso and self.target_isotope:
            return f"{base}_{self.target_isotope}"
        return base

    @property
    def n_waveforms(self) -> int:
        """Total number of waveforms (frames) in the set."""
        return len(self.filenames) * self.nframes
    
    @property   
    def n_files(self) -> int:
        """Number of files (alias for len())."""
        return len(self.filenames)

    def __len__(self):
        """Return number of files."""
        return self.n_files
    
    def __str__(self) -> str:
        ff_str = f"FastFrame({self.nframes} frames/file)" if self.ff else "single-frame"
        
        overrides = {
            'filenames': f"<{self.n_files} files, {self.n_waveforms} waveforms, {ff_str}>"
        }
        return format_dataclass_state(self, overrides=overrides)
    def __repr__(self) -> str:
        return self.__str__()

# -------------------------------
# Dataclasses for runs
# -------------------------------

@dataclass(frozen=True)
class Run:
    root_directory: Path
    run_id: str
    el_field: float   # V/cm                    # Essential for basic organization, so we make it mandatory at the Run level

    multiiso: bool = False                      # Whether this run contains multi-isotope sets (detected during bootstrapping)
    target_isotope: Optional[str] = "Th228"     # Optional (can be None for Multiiso), but useful for organization and logging (e.g., "Th228", "Ra224", "Rn220")

    # -- Experimental conditions (for logging) --
    pressure: float = 2.0 # bar
    temperature: float = 293.0 # K
    sampling_rate: float = 1e9 
    el_gap: float = 0.8 # cm
    drift_gap: float = 1.4 # cm

    # -- Populated in bootstrapping --
    sets: List[SetPmt] = field(default_factory=list)
    alpha_sets: List[SetAlpha] = field(default_factory=list)

    # Orchestrate cut params here
    gas_density: Optional[float] = None  # cm^-3, to be filled in
    width_s2: float = 1.1 # in µs
    t_s1: float = 0.0  # can be refined by batch analysis

    # Recombination constants (to be deprecated)
    recoil_energy: float = 96.8         # keV (Th228 recoil)
    W_value: float = 22.0               # eV per e-ion pair (gas Xe @ 2 bar)
    E_gamma_xray: float = 12.3e3        # eV X-ray energy (for Th228 decay)
    g_S2: Optional[float] = None        # mV·µs per electron

    # Alpha spectrum calibration results (populated by alpha_calibration pipeline)
    alpha_calibration: Optional[dict] = None  # Contains: fit_results, calibration_linear, calibration_quad, spectrum, spectrum_calibrated
    isotope_ranges: Optional[dict] = None     # Contains: {isotope: (E_min, E_max)}

    def __len__(self):
        """Return number of PMT sets."""
        return len(self.sets)
    
    def __str__(self):
        overrides = {
            'sets': f"<{len(self.sets)} SetPmt objects>",
            'alpha_sets': f"<{len(self.alpha_sets)} SetAlpha objects>"
        }
        return format_dataclass_state(self, overrides=overrides)
    def __repr__(self):
        return self.__str__()
    
# -------------------------------
# Integration results
# -------------------------------
@dataclass(frozen=True)
class S2Areas:
    """Transient arrays representing dense, per-frame integration data."""
    uids: np.ndarray
    areas: np.ndarray

    def filter_by_range(self, min_val: float, max_val: float) -> 'S2Areas':
        """Safely masks BOTH areas and UIDs simultaneously to prevent desync."""
        mask = (self.areas >= min_val) & (self.areas <= max_val)
        return S2Areas(
            uids=self.uids[mask],
            areas=self.areas[mask]
        )
        
    @property
    def mean(self) -> float:
        """Read-only convenience metric. Does not mutate state."""
        return float(np.mean(self.areas)) if len(self.areas) > 0 else 0.0



@dataclass
class CalibrationResults:
    """Results of X-ray calibration and ion recombination analysis."""
    run_id: str
    A_x_mean: float
    N_e_exp: float
    g_S2: float
    # per_set: dict[str, dict[str, float]]  # e.g. {set_id: {"A_ion": ..., "N_e_meas": ..., "r": ..., "E_d": ...}}

# -------------------------------
# --- Run & set print formatting --#
# -------------------------------

def format_dataclass_state(obj: Any, exclude: List[str] = None, overrides: Dict[str, str] = None) -> str:
    """
    Creates a clean, readable string representation of a dataclass,
    separating populated fields from missing (None) fields.
    """
    exclude = exclude or []
    overrides = overrides or {}
    
    populated = []
    missing = []
    
    for f in fields(obj):
        if f.name in exclude:
            continue
            
        # Handle custom overrides (like n_sets or n_files)
        if f.name in overrides:
            populated.append(f"  {f.name} = {overrides[f.name]}")
            continue
            
        val = getattr(obj, f.name)
        if val is not None:
            populated.append(f"  {f.name} = {val}")
        else:
            missing.append(f.name)
            
    # Build the final string
    lines = [f"{obj.__class__.__name__} state:"]
    lines.extend(populated)
    
    if missing:
        lines.append("\n  Missing:")
        # Wraps the missing list cleanly
        lines.append(f"    {', '.join(missing)}")
        
    return "\n".join(lines)


