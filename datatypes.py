from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Iterator, Any, List
import numpy as np
from .dataIO import load_wfm, parse_subdir_name
from .waveforms import PMTWaveform
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

    # --- Example operations ---
    def apply_cuts(self, cut_func: Callable[["PMTWaveform"], bool]) -> None:
        """
        Apply a cut function lazily across waveforms.
        The cut function returns True if waveform passes, False otherwise.
        Updates rejection_log with rejected indices.
        """
        rejected = []
        for i, wf in enumerate(self.iter_waveforms()):
            if not cut_func(wf):
                rejected.append(i)
        self.rejection_log.append(RejectionLog(
            cut_name=cut_func.__name__,
            rejected_indices=rejected,
            reason=f"Cut applied: {cut_func.__doc__ or ''}"
        ))

    def integrate_s2(self, integrator: Callable[["PMTWaveform"], float]) -> list[float]:
        """
        Compute S2 integrals for all waveforms, lazily.
        Returns a list of integrals (one per waveform).
        """
        return [integrator(wf) for wf in self.iter_waveforms()]

    # --- Static constructor ---
    @staticmethod
    def from_directory(path: str) -> "SetPmt":
        """
        Parse folder name and contained files to build SetPmt.
        Does NOT load waveforms, only stores filenames.
        """
        path = Path(path)
        # Example: FieldScan_5GSsec_Anode3000_Gate1600

        md = parse_subdir_name(path.name)
        
        # Filenames: RUN2_21082025_Gate70_Anode2470_P3_0006[_ch1].wfm
        filenames = [f.name for f in path.glob("*.wfm")]

        return SetPmt(source_dir=path, filenames=filenames, metadata=md)
# -------------------------------


@dataclass(frozen=True)
class RejectionLog:
    cut_name: str
    cut_fn: Callable[[Any], tuple[bool, np.ndarray, np.ndarray]]  # wf -> (bool, t_pass, V_pass)
    passed: list[int]
    rejected: list[int]
    reason: str = ""

    
# -------------------------------
# Integration results
# -------------------------------

@dataclass(frozen=True)
class S2Areas:
    set_id: str                    
    areas: np.ndarray               
    indices: List[int]              
    method: str                     
    params: dict[str, Any] = field(default_factory=dict)

    # Fit results
    mean: Optional[float] = None    
    sigma: Optional[float] = None   
    ci95: Optional[float] = None    
    fit_success: bool = False
    fit_result: Any = None  # Add this line to store the lmfit result

    def __repr__(self) -> str:
        return f"S2Areas(set_id={self.set_id}, n_areas={len(self.areas)}, method={self.method})"

