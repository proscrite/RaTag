import itertools
from dataclasses import replace
from pathlib import Path
from typing import List
from scipy.signal import find_peaks
import numpy as np

from .dataIO import load_wfm, parse_subdir_name
from .datatypes import SetPmt, Run
from .units import *
from .transport import compute_reduced_field, redfield_to_speed
from .transformations import batch_filenames, average_waveform, find_s1_in_avg
from .plotting import plot_s1_time_distribution
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


def set_fields(set_pmt: SetPmt, drift_gap_cm: float, el_gap_cm: float,
               gas_density: float = None) -> SetPmt:
    """
    Pure transformer: return a new SetPmt with drift/EL fields and reduced fields.
    """
    v_gate = V(set_pmt.metadata.get("gate"))
    v_anode = V(set_pmt.metadata.get("anode"))

    drift_field = v_gate / drift_gap_cm if v_gate is not None else None
    EL_field = (v_anode - v_gate) / el_gap_cm if v_anode is not None and v_gate is not None else None

    red_drift_Td = None
    red_EL_Td = None
    if gas_density and drift_field and EL_field:
        red_drift_Vcm2 = compute_reduced_field(drift_field, gas_density) # V·cm²
        red_drift_Td = to_Td(red_drift_Vcm2)  # Convert to Td
        
        red_EL_Vcm2 = compute_reduced_field(EL_field, gas_density) # V·cm²
        red_EL_Td = to_Td(red_EL_Vcm2)  # Convert to Td

    return replace(set_pmt,
                   drift_field=drift_field,
                   EL_field=EL_field,
                   red_drift_field=red_drift_Td,
                   red_EL_field=red_EL_Td)

def set_transport_properties(set_pmt: SetPmt,
                             drift_gap_cm: float,
                             transport = None) -> SetPmt:
    """
    Given a SetPmt and geometry + transport model,
    return a new SetPmt with drift speed, drift time,
    and diffusion coefficient filled in.
    Args:
        set_pmt: input SetPmt with red_drift_field set
        drift_gap_cm: drift length in cm
        transport: module with transport functions (e.g. RaTag.transport)
    Returns: a new SetPmt with transport properties set.
    """
    if set_pmt.red_drift_field is None:
        raise ValueError("red_drift_field must be set before calling set_transport_properties")

    # calculate drift speed from reduced field
    speed_mmus = redfield_to_speed(set_pmt.red_drift_field)  # returns mm/us

    # drift time = L / v
    if not drift_gap_cm or drift_gap_cm <= 0:
        raise ValueError("drift_gap_cm must be positive")
    
    drift_gap_mm = cm_to_mm(drift_gap_cm)

    time_drift_us = drift_gap_mm / speed_mmus if speed_mmus else None

    # diffusion coefficient (model-dependent)
    # diffusion = transport.redfield_to_diffusion(set_pmt.red_drift_field)
    diffusion = None  # Placeholder if no diffusion model provided

    return replace(set_pmt,
                   speed_drift=speed_mmus,
                   time_drift=time_drift_us,
                   diffusion_coefficient=diffusion)


def _find_s1_in_batches(set_pmt: SetPmt,
                        n_batches: int = 5,
                        batch_size: int = 20,
                        height_S1: float = 0.001,
                        min_distance: int = 200) -> List[float]:
    """Helper function to find S1 times in batches of waveforms.
    For FastFrame files, processes multiple files with frames.
    For single frame files, processes batches of files.
    
    Returns:
        List of S1 peak times in microseconds
    """
    s1_times = []
    first_wf = load_wfm(set_pmt.source_dir / set_pmt.filenames[0])
    
    def process_waveform(t_s, V_v):
        """Helper to process a single averaged waveform"""
        t_us = s_to_us(t_s)
        V_mV = V_to_mV(V_v)
        idx = find_s1_in_avg(t_us, V_mV, height_S1, min_distance)
        if idx is not None:
            s1_times.append(t_us[idx])

    if first_wf.ff:
        # For FastFrame files, process n_batches files
        for i in range(min(n_batches, len(set_pmt.filenames))):
            t_s, V_v = average_waveform([set_pmt.source_dir / set_pmt.filenames[i]])
            process_waveform(t_s, V_v)
    else:
        # For single frames, process batches
        batch_iter = batch_filenames(set_pmt.filenames, batch_size)
        for batch in itertools.islice(batch_iter, n_batches):
            t_s, V_v = average_waveform([set_pmt.source_dir / fn for fn in batch])
            process_waveform(t_s, V_v)

    if not s1_times:
        raise ValueError("No S1 peaks found in batches")
        
    return s1_times
    

def estimate_s1_from_batches(set_pmt: SetPmt,
                           n_batches: int = 5,
                           batch_size: int = 20,
                           height_S1: float = 0.001,
                           min_distance: int = 200,
                           flag_plot: bool = True) -> SetPmt:
    """Estimate average S1 time by analyzing batches and optionally plot distribution."""
    s1_times = _find_s1_in_batches(
        set_pmt, n_batches, batch_size, height_S1, min_distance
    )

    if flag_plot:
        plot_s1_time_distribution(s1_times, 
            f"S1 Peak Times Distribution - {set_pmt.source_dir.name}")

    t_s1_mean = round(np.mean(s1_times), 3)
    t_s1_std = round(np.std(s1_times), 3)

    new_meta = {**set_pmt.metadata, "t_s1": t_s1_mean, "t_s1_std": t_s1_std}
    return replace(set_pmt, metadata=new_meta)

# Outdated function (incorrect parameter and units handling)
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
        mask = (t > t_s1 + set_pmt.time_drift)  # small offset
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

def populate_run(run: Run) -> Run:
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
        for sd in sorted(run.root_directory.iterdir())
        if sd.is_dir() and sd.name.startswith("FieldScan")
    ]

    return replace(run, sets=sets)