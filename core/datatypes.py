from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Any, List, TYPE_CHECKING
import numpy as np

import matplotlib.pyplot as plt     # type: ignore[import]
from matplotlib.ticker import ScalarFormatter # type: ignore[import]

if TYPE_CHECKING:
    from typing import List as ListType

# -------------------------------
# Dataclasses for waveforms
# -------------------------------

@dataclass
class Waveform:
    """Generic waveform with common properties."""
    t: np.ndarray          # time axis
    v: np.ndarray          # voltage (signal)
    source: Optional[str]  # filename or run ID (for traceability)
    ff: bool = False  # whether loaded from FastFrames
    nframes: int = 1

    def __len__(self):
        return len(self.t)
    
    def __sizeof__(self):
        if self.ff:
            return self.v.shape
    
    def plot(self, ax=None, **kwargs):
        """Plot waveform."""
        from .plotting import plot_waveform
        plot_waveform(self, ax=ax, **kwargs)
        return ax


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

# -------------------------------
# Dataclasses for measurement sets
# -------------------------------

@dataclass(repr=False)
class SetPmt:

    # --- Provenance / housekeeping ---
    source_dir: Path
    filenames: list[str]     # lazy list of filenames (not waveforms!)
    metadata: dict

    # --- FastFrame properties ---
    ff: bool = False                 # Whether this set uses FastFrame files
    nframes: int = 1                 # Frames per file (1 for single-frame, typically 49 for FastFrame)

    # --- Physics context ---
    drift_field: float = None        # V/cm
    EL_field: float = None           # V/cm
    red_drift_field: float = None    # reduced drift field (Td)
    red_EL_field: float = None       # reduced EL field (Td)
    speed_drift: float = None        # mm/us
    time_drift: float = None         # us
    diffusion_coefficient: float = None    # mm/√cm 

    # --- Cuts bookkeeping ---
    rejection_log: List[Any] = field(default_factory=list)

    def __len__(self):
        """Return number of files."""
        return len(self.filenames)
    
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
        return f"SetPmt(source_dir={self.source_dir.name}, n_files={self.n_files}, n_waveforms={self.n_waveforms}, {ff_str})"


# -------------------------------
# Dataclasses for runs
# -------------------------------

@dataclass(frozen=True)
class Run:
    root_directory: Path
    run_id: str
    el_field: float
    target_isotope: str = "Th228"
    pressure: float = 2.0 # bar
    temperature: float = 293.0 # K
    sampling_rate: float = 1e9 
    el_gap: float = 0.8 # cm
    drift_gap: float = 1.4 # cm
    sets: List[SetPmt] = field(default_factory=list)

    # Orchestrate cut params here
    gas_density: Optional[float] = None  # cm^-3, to be filled in
    width_s2: float = 1.1 # in µs
    t_s1: float = 0.0  # can be refined by batch analysis

    # Calibration constants
    W_value: float = 22.0               # eV per e-ion pair (gas Xe @ 2 bar)
    E_gamma_xray: float = 12.3e3        # eV X-ray energy (for Th228 decay)
    A_x_mean: Optional[float] = None    # mean X-ray S2 area
    N_e_exp: Optional[float] = None     # expected electrons
    g_S2: Optional[float] = None        # mV·µs per electron

    
# -------------------------------
# Integration results
# -------------------------------

@dataclass(frozen=True)
class S2Areas:
    source_dir: Path                    
    areas: np.ndarray               
    method: str                     
    params: dict[str, Any] = field(default_factory=dict)

    # Fit results
    mean: Optional[float] = None    
    sigma: Optional[float] = None   
    ci95: Optional[float] = None    
    fit_success: bool = False
    fit_result: Any = None

    def __repr__(self) -> str:
        return f"S2Areas(source_dir={self.source_dir.name}, n_areas={len(self.areas)}, method={self.method})"


# -------------------------------
# X-ray event identification
# -------------------------------

@dataclass
class XRayEvent:
    wf_id: str
    accepted: bool
    reason: str
    area: float = None

@dataclass
class XRayResults:
    set_id: Path
    events: list[XRayEvent]
    params: dict[str, Any]


@dataclass
class CalibrationResults:
    """Results of X-ray calibration and ion recombination analysis."""
    run_id: str
    A_x_mean: float
    N_e_exp: float
    g_S2: float
    # per_set: dict[str, dict[str, float]]  # e.g. {set_id: {"A_ion": ..., "N_e_meas": ..., "r": ..., "E_d": ...}}



# -------------------------------
# Deprecated: Cut results
# -------------------------------

@dataclass(frozen=True)
class RejectionLog:
    cut_name: str
    cut_fn: Callable[[Any], tuple[bool, np.ndarray, np.ndarray]]  # wf -> (bool, t_pass, V_pass)
    passed: list[int]
    rejected: list[int]
    reason: str = ""