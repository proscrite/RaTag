from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Iterator, Any, List
import numpy as np
from .dataIO import load_wfm, parse_subdir_name

from typing import Optional
import matplotlib.pyplot as plt     # type: ignore[import]
from matplotlib.ticker import ScalarFormatter # type: ignore[import]

# -------------------------------
# Dataclasses for waveforms
# -------------------------------

@dataclass
class Waveform:
    """Generic waveform with common properties."""
    t: np.ndarray          # time axis
    v: np.ndarray          # voltage (signal)
    source: Optional[str]  # filename or run ID (for traceability)

    def __len__(self):
        return len(self.t)

    def slice(self, t_min: float, t_max: float) -> "Waveform":
        """Return a sub-waveform within [t_min, t_max]."""
        mask = (self.t >= t_min) & (self.t <= t_max)
        return Waveform(self.t[mask], self.v[mask], source=self.source)

    def baseline(self, t_min: float, t_max: float) -> float:
        """Compute baseline average in [t_min, t_max]."""
        return self.slice(t_min, t_max).v.mean()

    def area(self, t_min: float, t_max: float) -> float:
        """Compute area under the curve (trapezoid)."""
        w = self.slice(t_min, t_max)
        return np.trapz(w.v, w.t)
    
    def plot(self, ax=None, **kwargs):
        """Plot waveform."""
        if ax is None:
            ax = plt.gca()
        ax.plot(self.t, self.v, **kwargs)
        ax.set(xlabel='Time (µs)', ylabel='Signal (V)', title=Path(self.source).name if self.source else 'Waveform')
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-6, -6))
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

@dataclass(frozen=True)
class SetPmt:

    # --- Provenance / housekeeping ---
    source_dir: Path
    filenames: list[str]     # lazy list of filenames (not waveforms!)
    metadata: dict

    # --- Physics context ---
    drift_field: float = None        # V/cm
    EL_field: float = None           # V/cm
    red_drift_field: float = None    # reduced drift field (Td)
    red_EL_field: float = None       # reduced EL field (Td)
    speed_drift: float  = None              # mm/us
    time_drift: float  = None               # us
    diffusion_coefficient: float = None    # mm/√cm 

    # --- Cuts bookkeeping ---
    rejection_log: list["RejectionLog"] = field(default_factory=list)

    def __len__(self):
        return len(self.filenames)
    
    # --- Lazy loader ---
    def iter_waveforms(self) -> Iterator["PMTWaveform"]:
        """Yield PMTWaveform objects lazily, one by one."""
        
        for fn in self.filenames:
            yield load_wfm(self.source_dir / fn)

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
    sets: List[SetPmt] = None

    # Orchestrate cut params here
    bs_window: tuple[float, float] = (-1.5e-5, -1.0e-5)
    width_s2: float = 1.1 # in µs
    t_s1: float = 0.0  # can be refined by batch analysis

    
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
        return f"S2Areas(source_dir={self.set_id}, n_areas={len(self.areas)}, method={self.method})"


# -------------------------------
# Cut results
# -------------------------------

@dataclass(frozen=True)
class RejectionLog:
    cut_name: str
    cut_fn: Callable[[Any], tuple[bool, np.ndarray, np.ndarray]]  # wf -> (bool, t_pass, V_pass)
    passed: list[int]
    rejected: list[int]
    reason: str = ""

