from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from functools import lru_cache

from typing import Callable, Optional, Any, List, TYPE_CHECKING
import numpy as np

import matplotlib.pyplot as plt     # type: ignore[import]
from matplotlib.ticker import ScalarFormatter # type: ignore[import]

from RaTag.alphas.energy_map_reader import get_energy_for_frame
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


# -----------------------------------------------#
# Tracker for individual frames in FF files   ---#
# -----------------------------------------------#

@dataclass
class FrameProxy:
    file_path: Path 
    file_seq: int
    frame_idx: int
    chunk_dir: Optional[str] = None
    fmt: str = '8b'   # format of maps
    scale: float = 0.1

    # small cache of loaded waveform per-file handled by module-level loader
    def uid(self) -> int:
        return int(self.file_seq) * 64 + int(self.frame_idx)

    @property
    def energy(self) -> float:
        """Get energy from energy_map (cached by energy_map module)."""
        from RaTag.alphas.energy_map_reader import get_energy_for_frame
        if self.chunk_dir is None:
            raise RuntimeError("chunk_dir not provided to FrameProxy")
        e = get_energy_for_frame(self.chunk_dir, self.file_seq, self.frame_idx, fmt=self.fmt, scale=self.scale)
        return e

    @staticmethod
    def _check_channel(file_path: str, which: str = 'pmt'):
        if which == 'pmt':
            return 'Ch1' in file_path
        elif which == 'alpha':
            return 'Ch4' in file_path
        else:
            raise TypeError('Waveform type error')
        
    @staticmethod
    def _swap_ch_filepath(file_path: str)->Path:
        if 'Ch1' in file_path:
            return Path(file_path.replace('Ch1', 'Ch4'))
        elif 'Ch4' in file_path:
            return Path(file_path.replace('Ch4', 'Ch1'))

    # LRU loader for waveform arrays (per file path)
    @staticmethod
    @lru_cache(maxsize=64)
    def _load_file_waveforms_cached(file_path: str, which: str = 'pmt'):
        """
        Should return numpy array shape (n_frames, n_samples) for channel4 or whatever.
        Implement or import your existing loader here: e.g. load_alpha(file_path).v
        Keep this function small and replace the internals with your repo's loader.
        """
        from RaTag.core.dataIO import load_alpha, load_wfm
        if which == 'alpha':
            wf_alpha = load_alpha(file_path)
            return wf_alpha
        elif which == 'pmt':
            wf_pmt = load_wfm(file_path)
            return wf_pmt

    def load_alpha_frame(self) -> Waveform:
        """Return waveform array for this frame (1D)."""
        from RaTag.core.dataIO import extract_single_frame

        if FrameProxy._check_channel(str(self.file_path), which='alpha'):
            load_path = self.file_path
        else:
            load_path = FrameProxy._swap_ch_filepath(str(self.file_path))
        
        wf_alpha = FrameProxy._load_file_waveforms_cached(load_path, which='alpha')
        frame = extract_single_frame(wf_alpha, self.frame_idx)        # guard in case file has fewer frames done internally
        return frame

    def load_pmt_frame(self) -> Waveform:
        """Return waveform array for this frame (1D)."""
        from RaTag.core.dataIO import extract_single_frame
        if FrameProxy._check_channel(str(self.file_path), which='pmt'):
            load_path = self.file_path
        else:
            load_path = FrameProxy._swap_ch_filepath(str(self.file_path))
        
        wf_pmt = FrameProxy._load_file_waveforms_cached(load_path, which='pmt')
        frame = extract_single_frame(wf_pmt, self.frame_idx)        # guard in case file has fewer frames done internally
        return frame
# -------------------------------
# Dataclasses for measurement sets
# -------------------------------

@dataclass(repr=False)
class SetPmt:

    # --- Provenance / housekeeping ---
    source_dir: Path
    filenames: list[str]     # lazy list of filenames (not waveforms!)
    metadata: dict
    multiiso: bool = False   # Multi-isotope set

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