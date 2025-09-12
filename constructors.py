import itertools
from dataclasses import replace
from pathlib import Path
from typing import List, 
from scipy.signal import find_peaks

from .dataIO import parse_subdir_name
from .datatypes import SetPmt, Run
from .transport import compute_reduced_field, redfield_to_speed
from .transformations import batch_filenames, average_waveform, find_s1_in_avg

# ------------------------
# --- Set constructors ---
# ------------------------

def set_from_dir(path: Path) -> SetPmt:
    """
    Parse folder name and contained files to build SetPmt.
    Does NOT load waveforms, only stores filenames.
    """
    # Example: FieldScan_5GSsec_Anode3000_Gate1600
    md = parse_subdir_name(path.name)
    
    # Filenames: RUN2_21082025_Gate70_Anode2470_P3_0006[_ch1].wfm
    filenames = [f.name for f in path.glob("*.wfm")]

    return SetPmt(source_dir=path, filenames=filenames, metadata=md)


def set_fields(set_pmt: SetPmt, drift_gap: float, el_gap: float,
               gas_density: float = None) -> SetPmt:
    """
    Pure transformer: return a new SetPmt with drift/EL fields and (optionally) reduced fields.
    """
    v_gate = set_pmt.metadata.get("gate")
    v_anode = set_pmt.metadata.get("anode")

    drift_field = v_gate / drift_gap if v_gate is not None else None
    EL_field = (v_anode - v_gate) / el_gap if v_anode is not None and v_gate is not None else None

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
                             drift_gap: float,
                             transport) -> SetPmt:
    """
    Given a SetPmt and geometry + transport model,
    return a new SetPmt with drift speed, drift time,
    and diffusion coefficient filled in.
    Args:
        set_pmt: input SetPmt with red_drift_field set
        drift_gap: drift length in cm
        transport: module with transport functions (e.g. RaTag.transport)
    Returns: a new SetPmt with transport properties set.
    """
    if set_pmt.red_drift_field is None:
        raise ValueError("red_drift_field must be set before calling set_transport_properties")

    # calculate drift speed from reduced field
    speed = redfield_to_speed(set_pmt.red_drift_field)  # mm/us

    # drift time = L / v
    if not drift_gap or drift_gap <= 0:
        raise ValueError("drift_gap must be positive")
    time_drift = drift_gap * 10 / speed * 1e-6 if speed else None

    # diffusion coefficient (model-dependent)
    # diffusion = transport.redfield_to_diffusion(set_pmt.red_drift_field)
    diffusion = None  # Placeholder if no diffusion model provided

    return replace(set_pmt,
                   speed_drift=speed,
                   time_drift=time_drift,
                   diffusion_coefficient=diffusion)


def estimate_s1_from_batches(set_pmt: SetPmt,
                             n_batches: int = 5,
                             batch_size: int = 20,
                             height_S1=0.001,
                             min_distance=200) -> SetPmt:
    """Estimate average S1 time by analyzing up to n_batches of averaged waveforms."""
    s1_times = []
    batch_iter = batch_filenames(set_pmt.filenames, batch_size)

    for batch in itertools.islice(batch_iter, n_batches):
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


# ------------------------
# --- Run constructors ---
# ------------------------

def populate_run(path: Path, run: Run) -> Run:
    """
    Populate Run.sets with SetPmt objects from a root directory.

    Args:
        path: Path to root directory containing FieldScan subdirectories.
        run: Existing Run object with experiment parameters.

    Returns:
        New Run with sets field populated.
    """
    sets: List[SetPmt] = [
        set_from_dir(sd)
        for sd in path.iterdir()
        if sd.is_dir() and sd.name.startswith("FieldScan")
    ]

    return replace(run, sets=sets)