from dataclasses import dataclass, field, replace
import itertools
from typing import Iterator, Callable, List, Union
from pathlib import Path
from functools import reduce
from scipy.signal import find_peaks
import numpy as np

from .waveforms import PMTWaveform
from .transport import compute_reduced_field, redfield_to_speed
from .dataIO import parse_subdir_name, parse_filename, load_wfm
from .cuts import make_time_amplitude_cut, RejectionLog

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
# Functions to construct SetPmt
# -------------------------------

def set_fields(set_pmt: SetPmt, drift_length: float, EL_gap: float,
               gas_density: float = None) -> SetPmt:
    """
    Pure transformer: return a new SetPmt with drift/EL fields and (optionally) reduced fields.
    """
    v_gate = set_pmt.metadata.get("gate")
    v_anode = set_pmt.metadata.get("anode")

    drift_field = v_gate / drift_length if v_gate is not None else None
    EL_field = (v_anode - v_gate) / EL_gap if v_anode is not None and v_gate is not None else None

    red_drift = None
    red_EL = None
    if gas_density and drift_field and EL_field:
        red_drift = compute_reduced_field(drift_field, gas_density)
        red_EL = compute_reduced_field(EL_field, gas_density)

    return replace(set_pmt,
                   drift_field=drift_field,
                   EL_field=EL_field,
                   red_drift_field=red_drift,
                   red_EL_field=red_EL)

def set_transport_properties(set_pmt: SetPmt,
                             drift_length: float,
                             transport) -> SetPmt:
    """
    Given a SetPmt and geometry + transport model,
    return a new SetPmt with drift speed, drift time,
    and diffusion coefficient filled in.
    Args:
        set_pmt: input SetPmt with red_drift_field set
        drift_length: drift length in cm
        transport: module with transport functions (e.g. RaTag.transport)
    Returns: a new SetPmt with transport properties set.
    """
    if set_pmt.red_drift_field is None:
        raise ValueError("red_drift_field must be set before calling set_transport_properties")

    # calculate drift speed from reduced field
    speed = redfield_to_speed(set_pmt.red_drift_field)  # mm/us

    # drift time = L / v
    if not drift_length or drift_length <= 0:
        raise ValueError("drift_length must be positive")
    time_drift = drift_length * 10 / speed * 1e-6 if speed else None

    # diffusion coefficient (model-dependent)
    # diffusion = transport.redfield_to_diffusion(set_pmt.red_drift_field)
    diffusion = None  # Placeholder if no diffusion model provided

    return replace(set_pmt,
                   speed_drift=speed,
                   time_drift=time_drift,
                   diffusion_coefficient=diffusion)

# ---------------------------------
# Functions to analyze  SetPmt
# ---------------------------------

def batch_filenames(filenames: list[str], batch_size: int = 20):
    """Yield batches of filenames (lists) of size batch_size."""
    it = iter(sorted(filenames))
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch

def average_waveform(batch_files: list[str]):
    """Compute average waveform for a batch of files."""
    waveforms = [load_wfm(fn) for fn in batch_files]
    # All t should be aligned, take from first
    t = waveforms[0].t
    V_stack = np.stack([wf.v for wf in waveforms])
    V_avg = V_stack.mean(axis=0)
    return t, V_avg

def find_s1_in_avg(t, V, height_S1=0.001, min_distance=200):
    """Find S1 peak index in averaged waveform."""
    mask = t < 0
    inds = find_peaks(V[mask], height=height_S1, distance=min_distance)[0]
    if len(inds) == 0:
        return None
    return inds[np.argmax(V[inds])] if len(inds) > 1 else inds[0]


def estimate_s1_from_batches(set_pmt: SetPmt,
                             batch_size: int = 20,
                             height_S1=0.001,
                             min_distance=200) -> SetPmt:
    """Estimate average S1 time by analyzing averaged batches of waveforms."""
    s1_times = []
    for batch in batch_filenames(set_pmt.filenames, batch_size):
        t, V = average_waveform([set_pmt.source_dir / fn for fn in batch])
        idx = find_s1_in_avg(t, V, height_S1, min_distance)
        if idx is not None:
            s1_times.append(t[idx])

    if not s1_times:
        raise ValueError("No S1 peaks found in batches")

    t_s1_mean = float(np.mean(s1_times))
    t_s1_std  = float(np.std(s1_times))

    new_meta = {**set_pmt.metadata, "t_s1": t_s1_mean, "t_s1_std": t_s1_std}
    return replace(set_pmt, metadata=new_meta)

def estimate_s2_window_from_batches(set_pmt: SetPmt,
                                    batch_size: int = 20,
                                    baseline_window=(-1.5e-5, -1.0e-5),
                                    height_S2=0.002,
                                    min_distance=200):
    """Estimate average S2 window (start, end) by analyzing averaged batches of waveforms."""
    s2_starts, s2_ends = [], []
    t_s1 = set_pmt.metadata.get("t_s1")
    if t_s1 is None:
        raise ValueError("t_s1 must be estimated first")

    for batch in batch_filenames(set_pmt.filenames, batch_size):
        t, V = average_waveform([set_pmt.source_dir / fn for fn in batch],
                                baseline_window)

        # Look for peaks after drift time
        mask = (t > t_s1 + set_pmt.time_drift - 1e-6)  # small offset
        inds = find_peaks(V[mask], height=height_S2, distance=min_distance)[0]
        if len(inds) == 0:
            continue

        idx = inds[np.argmax(V[inds])]
        idx_global = np.where(mask)[0][idx]

        # crude start/end detection
        v_peak = V[idx_global]
        # start: first time rising above 10% of peak
        start_idx = np.argmax(V[:idx_global] > 0.1 * v_peak)
        # end: first time after peak that falls below 10% of peak
        after_peak = np.where(V[idx_global:] < 0.1 * v_peak)[0]
        end_idx = (after_peak[0] + idx_global) if len(after_peak) else None

        if end_idx is not None:
            s2_starts.append(t[start_idx])
            s2_ends.append(t[end_idx])

    if not s2_starts or not s2_ends:
        raise ValueError("No S2 windows found in batches")

    return {
        "s2_start_mean": float(np.mean(s2_starts)),
        "s2_start_std": float(np.std(s2_starts)),
        "s2_end_mean": float(np.mean(s2_ends)),
        "s2_end_std": float(np.std(s2_ends))
    }


# -------------------------------------
# Functions to implement cuts to SetPmt
# -------------------------------------

def drift_region_cut(set_pmt: SetPmt, t_tol: float = 1e-6, threshold: float = 0.02, nmax_grass: int = 5):
    t_s1 = set_pmt.metadata["t_s1"]
    t_drift = set_pmt.time_drift
    return make_time_amplitude_cut(t_start = t_s1 + t_tol, t_end = t_s1 + t_drift - t_tol, threshold=threshold, nmax_grass=nmax_grass)

def post_s2_cut(set_pmt: SetPmt, t_tol: float = 1e-6, width_s2: float = 1e-5, threshold: float = 0.02, nmax_grass: int = 5):
    t_s1 = set_pmt.metadata["t_s1"]
    t_drift = set_pmt.time_drift
    # width_s2 = set_pmt.metadata["width_s2"]
    t2_end = t_s1 + t_drift + width_s2 + t_tol

    return make_time_amplitude_cut(t_start=t2_end, t_end=None, threshold=threshold, nmax_grass=nmax_grass)

# -------------------------------
# Cut application and logging
# -------------------------------

def apply_cut(set_pmt: SetPmt,
              cut_fn: Callable[[PMTWaveform], bool],
              cut_name: str,
              reason: str = "") -> RejectionLog:
    """
    Apply a cut function to all waveforms in a SetPmt.

    Args:
        set_pmt: The set of waveforms to evaluate.
        cut_fn: A function that takes a PMTWaveform and returns True (pass) or False (reject).
        cut_name: A human-readable name for the cut.
        reason: Optional description of the physics motivation.

    Returns:
        RejectionLog with passed/rejected indices.
    """
    passed, rejected = [], []
    for idx, wf in enumerate(set_pmt.iter_waveforms()):
        ok, _, _ = cut_fn(wf)
        (passed if ok else rejected).append(idx)
    return RejectionLog(cut_name, cut_fn, passed, rejected)

def filter_set(set_pmt: SetPmt, logs: Union[RejectionLog, List[RejectionLog]]) -> SetPmt:
    """
    Return a new SetPmt containing only filenames that passed the cut(s).
    Accepts a single RejectionLog or a list of them.
    If a list is given, waveforms must pass ALL cuts to be included.
    """
    if isinstance(logs, RejectionLog):
        passed = set(logs.passed)
    else:  # list of logs
        passed_sets = [set(log.passed) for log in logs]
        passed = set.intersection(*passed_sets) if passed_sets else set()

    new_files = [fn for idx, fn in enumerate(set_pmt.filenames) if idx in passed]
    return replace(set_pmt, filenames=new_files)

# def filter_set(set_pmt: SetPmt, log: RejectionLog) -> SetPmt:
#     keep = [set_pmt.filenames[i] for i in log.passed]
#     return replace(set_pmt, filenames=keep)

def combine_logs(logs: List[RejectionLog]) -> RejectionLog:
    if not logs:
        return RejectionLog("all_cuts", [], [], lambda wf: True, reason="No cuts")

    passed = sorted(set.intersection(*(set(log.passed) for log in logs)))
    rejected = sorted(set.union(*(set(log.rejected) for log in logs)))

    # Combined cut: apply all cuts in sequence
    def combined_cut(wf: PMTWaveform) -> bool:
        return all(log.cut_fn(wf) for log in logs)

    return RejectionLog(
        cut_name="all_cuts",
        passed=passed,
        rejected=rejected,
        cut_fn=combined_cut,
        reason="Passed all cuts"
    )


# For reference, previous version of evaluate_cuts:

# def evaluate_cuts(set_pmt: SetPmt,
#                   bs_window: tuple[float, float],
#                   t_tol: float = 1e-6,
#                   threshold: float = 0.02,
#                   nmax_grass: int = 5,
#                   width_s2: float = 1e-5) -> list[RejectionLog]:
#     """Apply all defined cuts and return their logs."""
#     baseline = make_baseline_cut(bs_window)
#     drift_cut = drift_region_cut(set_pmt, t_tol, threshold, nmax_grass)
#     post_s2 = post_s2_cut(set_pmt, t_tol, width_s2, threshold, nmax_grass)

#     logs = [
#         apply_cut(set_pmt, baseline, "baseline", "Stable baseline"),
#         apply_cut(set_pmt, drift_cut, "drift_region", "Clean drift region"),
#         apply_cut(set_pmt, post_s2, "post_s2", "No noise after S2"),
#     ]
#     return logs


# -------------------------------
# High-level pipeline function
# -------------------------------

def pipe(value, *funcs):
    """Apply functions sequentially to a value."""
    return reduce(lambda v, f: f(v), funcs, value)
