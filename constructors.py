import itertools
from dataclasses import replace
from pathlib import Path
from typing import List
from scipy.signal import find_peaks
import numpy as np

from .dataIO import load_wfm, parse_subdir_name, iter_waveforms
from .datatypes import PMTWaveform, SetPmt, Run
from .units import *
from .transport import compute_reduced_field, redfield_to_speed
from . import transformations as tf
from .plotting import plot_s1_time_distribution
# ------------------------
# --- Set constructors ---
# ------------------------

def set_from_dir(path: Path, nfiles: int = None) -> SetPmt:
    """
    Parse folder name and contained files to build SetPmt.
    Does NOT load waveforms, only stores filenames.
    """
    # Example: FieldScan_5GSsec_Anode3000_Gate1600
    md = parse_subdir_name(path.name)
    
    # Filenames: RUN2_21082025_Gate70_Anode2470_P3_0006[_ch1].wfm
    filenames = [f.name for f in path.glob("*.wfm")]
    if nfiles is not None:
        filenames = filenames[:nfiles]

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
        idx = tf.find_s1_in_avg(t_us, V_mV, height_S1, min_distance)
        if idx is not None:
            s1_times.append(t_us[idx])

    if first_wf.ff:
        # For FastFrame files, process n_batches files
        for i in range(min(n_batches, len(set_pmt.filenames))):
            t_s, V_v = tf.average_waveform([set_pmt.source_dir / set_pmt.filenames[i]])
            process_waveform(t_s, V_v)
    else:
        # For single frames, process batches
        batch_iter = tf.batch_filenames(set_pmt.filenames, batch_size)
        for batch in itertools.islice(batch_iter, n_batches):
            t_s, V_v = tf.average_waveform([set_pmt.source_dir / fn for fn in batch])
            process_waveform(t_s, V_v)

    if not s1_times:
        raise ValueError("No S1 peaks found in batches")

    return np.array(s1_times)


def estimate_s1_from_batches(set_pmt: SetPmt,
                           n_batches: int = 20,
                           batch_size: int = 50,
                           height_S1: float = 0.001,
                           min_distance: int = 200,
                           sigma_threshold: float = 1.0,
                           flag_plot: bool = True) -> SetPmt:
    """Estimate average S1 time by analyzing batches and optionally plot distribution."""

    t_s1_mean = 1
    t_s1_std = 0.9
    s1_times = _find_s1_in_batches(
        set_pmt, n_batches, batch_size, height_S1, min_distance
    )

    
    t_mean_init = round(np.mean(s1_times), 3)
    dt_init = round(np.std(s1_times), 3)
    mask = np.abs(s1_times - t_mean_init) < (sigma_threshold * dt_init)
    s1_times_clean = s1_times[mask]

    n, bins = np.histogram(s1_times_clean, bins=50)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    t_s1_mode = round(cbins[np.argmax(n)], 3)
    t_s1_std = round(np.std(s1_times_clean), 3)

    if flag_plot:
        plot_s1_time_distribution(s1_times_clean, 
            f"S1 Peak Times Distribution - {set_pmt.source_dir.name}")
    

    new_meta = {**set_pmt.metadata, "t_s1": t_s1_mode, "t_s1_std": t_s1_std}
    return replace(set_pmt, metadata=new_meta)

def _find_s2_window(wf: PMTWaveform,
                    t_s1: float,
                    t_drift: float,
                    threshold_s2: float = 0.4,) -> tuple[float, float]:
    """Estimate S2 window (start, end) in a clean waveform."""
  
    t, V = wf.t, wf.v
    mask = t > t_s1 + t_drift * 0.75
    s2_window = V[mask]
    positive_mask = s2_window > threshold_s2

    s2_start = t[mask][positive_mask][0]
    s2_end = t[mask][positive_mask][-1] 
    return s2_start, s2_end
    

def s2_window_pipeline(wf, t_s1: float, t_drift: float,
                        window_size: int = 9,
                       threshold_s2: float = 0.4,
                       threshold_clip: float = 0.02):
    """Estimate S2 window (start, end) in a waveform."""
    wf = tf.t_in_us(wf)
    wf = tf.v_in_mV(wf)
    wf = tf.moving_average(wf, window=window_size)
    wf = tf.threshold_clip(wf, threshold=threshold_clip)
    # wf = tf.subtract_pedestal(wf, n_points=200)

    t_start, t_end = _find_s2_window(wf, t_s1, t_drift, threshold_s2=threshold_s2)

    return t_start, t_end


def estimate_s2_window(set_pmt: SetPmt,
                      threshold_s2: float = 0.4,
                      window_size: int = 9,
                      threshold_clip: float = 0.02,
                      max_waveforms: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate S2 window (start, end, duration) for all waveforms in a set.
    
    Args:
        set_pmt: SetPmt with t_s1 and time_drift set
        threshold_s2: Amplitude threshold for S2 detection (mV)
        window_size: Moving average window size
        threshold_clip: Clipping threshold for noise
        max_waveforms: Maximum number of waveforms to process (None = all)
        
    Returns:
        Tuple of (t_starts, t_ends, s2_durations) in microseconds
    """
    t_s1 = set_pmt.metadata.get("t_s1")
    if t_s1 is None:
        raise ValueError("t_s1 must be estimated first (call estimate_s1_from_batches)")
    
    if set_pmt.time_drift is None:
        raise ValueError("time_drift must be set (call set_transport_properties)")

    t_starts, t_ends, durations = [], [], []
    
    waveforms = iter_waveforms(set_pmt)
    if max_waveforms is not None:
        waveforms = itertools.islice(waveforms, max_waveforms)
    
    for wf in waveforms:
        try:
            t_start, t_end = s2_window_pipeline(
                wf, t_s1, set_pmt.time_drift,
                window_size=window_size,
                threshold_s2=threshold_s2,
                threshold_clip=threshold_clip
            )
            if t_end > t_start:
                t_starts.append(t_start)
                t_ends.append(t_end)
                durations.append(t_end - t_start)
        except (ValueError, IndexError):
            continue
    
    if not t_starts:
        raise ValueError(f"No valid S2 windows found in {set_pmt.source_dir.name}")
    
    return np.array(t_starts), np.array(t_ends), np.array(durations)


def compute_s2_variance(s2_durations: np.ndarray,
                       duration_cuts: tuple[float, float] = None,
                       method: str = 'percentile') -> tuple[float, float]:
    """
    Compute robust variance estimate from S2 duration distribution.
    
    Args:
        s2_durations: Array of S2 durations in µs
        duration_cuts: Optional (min, max) cuts to remove outliers
        method: 'percentile' (16-84), 'mad', or 'std'
        
    Returns:
        (central_value, spread) in µs
    """
    if duration_cuts is not None:
        mask = (s2_durations >= duration_cuts[0]) & (s2_durations <= duration_cuts[1])
        durations_clean = s2_durations[mask]
    else:
        durations_clean = s2_durations
    
    if len(durations_clean) < 10:
        raise ValueError(f"Too few events after cuts: {len(durations_clean)}")
    
    if method == 'percentile':
        p16, p50, p84 = np.percentile(durations_clean, [16, 50, 84])
        sigma_lower = p50 - p16
        sigma_upper = p84 - p50
        spread = (sigma_upper + sigma_lower) / 2
        return p50, spread
    
    elif method == 'mad':
        median = np.median(durations_clean)
        mad = 1.4826 * np.median(np.abs(durations_clean - median))
        return median, mad
    
    elif method == 'std':
        return np.mean(durations_clean), np.std(durations_clean)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def s2_variance_run(run: Run,
                   s2_duration_cuts: tuple[float, float] = (5, 25),
                   threshold_s2: float = 0.4,
                   max_waveforms: int = None,
                   method: str = 'percentile') -> Run:
    """
    Compute S2 timing statistics for all sets in a run and store in metadata.
    
    Args:
        run: Run object with prepared sets
        s2_duration_cuts: (min, max) duration cuts in µs
        threshold_s2: S2 detection threshold in mV
        max_waveforms: Max waveforms per set (None = all)
        method: Variance estimation method
        
    Returns:
        Updated Run with S2 timing statistics stored in each set's metadata:
            - t_s2_start_mean, t_s2_start_std
            - t_s2_end_mean, t_s2_end_std
            - s2_duration_mean, s2_duration_std
    """
    updated_sets = []
    
    for set_pmt in run.sets:
        try:
            # Get raw S2 timing data
            t_starts, t_ends, durations = estimate_s2_window(
                set_pmt,
                threshold_s2=threshold_s2,
                max_waveforms=max_waveforms
            )
            
            # Compute statistics for each timing quantity
            timing_data = [
                ('t_s2_start', t_starts, None),              # No cuts for start times
                ('t_s2_end', t_ends, None),                  # No cuts for end times
                ('s2_duration', durations, s2_duration_cuts) # Cuts for durations
            ]
            
            new_metadata = {**set_pmt.metadata}
            
            for name, data, cuts in timing_data:
                mean, std = compute_s2_variance(data, duration_cuts=cuts, method=method)
                new_metadata[f'{name}_mean'] = mean
                new_metadata[f'{name}_std'] = std
            
            updated_sets.append(replace(set_pmt, metadata=new_metadata))
            
            print(f"✓ {set_pmt.source_dir.name}:")
            print(f"  S2 Start:    {new_metadata['t_s2_start_mean']:.2f} ± {new_metadata['t_s2_start_std']:.2f} µs")
            print(f"  S2 End:      {new_metadata['t_s2_end_mean']:.2f} ± {new_metadata['t_s2_end_std']:.2f} µs")
            print(f"  S2 Duration: {new_metadata['s2_duration_mean']:.2f} ± {new_metadata['s2_duration_std']:.2f} µs")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to process {set_pmt.source_dir.name}: {e}")
            updated_sets.append(set_pmt)  # Keep original if processing fails
            continue
    
    return replace(run, sets=updated_sets)