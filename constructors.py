import itertools
from dataclasses import replace
from pathlib import Path
from typing import List, Optional
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

def set_from_dir(source_dir: Path, nfiles: Optional[int] = None) -> SetPmt:
    """
    Create SetPmt from directory by lazy-loading filenames.
    
    Automatically detects FastFrame properties from the first file.
    
    Args:
        source_dir: Path to directory containing .wfm files
        nfiles: Optional limit on number of files to load
        
    Returns:
        SetPmt with filenames and FastFrame properties detected
    """
    source_dir = Path(source_dir)
    
    # Get all .wfm files
    all_files = sorted(source_dir.glob("*.wfm"))
    
    if not all_files:
        raise FileNotFoundError(f"No .wfm files found in {source_dir}")
    
    # Limit files if requested
    files_to_use = all_files[:nfiles] if nfiles is not None else all_files
    filenames = [f.name for f in files_to_use]
    
    # Detect FastFrame properties from first file
    first_wf = load_wfm(files_to_use[0])
    ff = first_wf.ff
    nframes = first_wf.nframes if ff else 1
    
    # Parse metadata from directory name
    metadata = parse_subdir_name(source_dir.name)
    
    return SetPmt(
        source_dir=source_dir,
        filenames=filenames,
        metadata=metadata,
        ff=ff,
        nframes=nframes
    )


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


def _find_s1_in_frames(set_pmt: SetPmt,
                       max_files: int,
                       threshold_s1: float = 1.0) -> List[float]:
    """Find S1 times by looking at individual frames.
    
    Optimized for FastFrame data - processes each frame independently
    looking for the last peak before t=0 above threshold.
    
    Args:
        set_pmt: Measurement set
        max_files: Maximum number of files to process
        threshold_s1: S1 detection threshold in mV (default: 0.1)
        
    Returns:
        List of S1 peak times in microseconds
    """
    s1_times = []
    
    for wf in itertools.islice(iter_waveforms(set_pmt), max_files):
        # Convert units
        wf = tf.t_in_us(wf)
        wf = tf.v_in_mV(wf)
        wf = tf.subtract_pedestal(wf, n_points=200)
        
        # Handle FastFrame: iterate over frames
        if wf.ff:
            for frame_v in wf.v:
                frame_wf = replace(wf, v=frame_v, ff=False)
                t_s1 = tf.find_s1_in_frame(frame_wf.t, frame_wf.v, threshold_s1)
                if t_s1 is not None:
                    s1_times.append(t_s1)
        else:
            # Single frame
            t_s1 = tf.find_s1_in_frame(wf.t, wf.v, threshold_s1)
            if t_s1 is not None:
                s1_times.append(t_s1)
    
    if not s1_times:
        raise ValueError("No S1 peaks found in frames")
    
    return np.array(s1_times)


def estimate_s1_from_frames(set_pmt: SetPmt, 
                           max_frames: int = 1000,
                           threshold_s1: float = 1.0, 
                           flag_plot: bool = False) -> SetPmt:
    """
    Estimate S1 timing from individual frames (optimized for FastFrame).
    
    Processes frames individually looking for S1-like peaks before t=0.
    No batching or averaging - each frame analyzed independently.
    
    Args:
        set_pmt: Measurement set with ff and nframes properties
        max_frames: Target number of frames to process (default: 1000)
                    Actual frames = ceil(max_frames/nframes) × nframes
        threshold_s1: Minimum peak height in mV (default: 0.1)
        flag_plot: If True, show diagnostic plot
        
    Returns:
        SetPmt with t_s1 and t_s1_std in metadata
    """
    # Compute how many files to process (rounds up to complete files)
    max_files = int(np.ceil(max_frames / set_pmt.nframes))
    actual_frames = max_files * set_pmt.nframes
    
    print(f"  S1 estimation: processing {max_files} files (~{actual_frames} frames)")
    
    # Find S1 in individual frames
    s1_times = _find_s1_in_frames(set_pmt, max_files, threshold_s1)
    
    # Outlier rejection
    t_mean_init = round(np.mean(s1_times), 3)
    dt_init = round(np.std(s1_times), 3)
    mask = np.abs(s1_times - t_mean_init) < (3 * dt_init)
    s1_times_clean = s1_times[mask]
    
    # Compute mode from histogram
    n, bins = np.histogram(s1_times_clean, bins=50)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    t_s1_mode = round(cbins[np.argmax(n)], 3)
    t_s1_std = round(np.std(s1_times_clean), 3)
    
    print(f"  → t_s1 = {t_s1_mode} ± {t_s1_std} µs (from {len(s1_times_clean)} frames)")
    
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
                       threshold_s2: float = 0.8,
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
                      threshold_s2: float = 0.8,
                      window_size: int = 9,
                      threshold_clip: float = 0.02,
                      max_frames: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate S2 window (start, end, duration) for waveforms in a set.
    Supports both single-frame and FastFrame waveforms.
    
    Args:
        set_pmt: SetPmt with t_s1 and time_drift set
        threshold_s2: Amplitude threshold for S2 detection (mV)
        window_size: Moving average window size
        threshold_clip: Clipping threshold for noise
        max_frames: Target number of frames to process (None = all)
                    Actual frames = ceil(max_frames/nframes) × nframes
        
    Returns:
        Tuple of (t_starts, t_ends, s2_durations) in microseconds
    """
    t_s1 = set_pmt.metadata.get("t_s1")
    if t_s1 is None:
        raise ValueError("t_s1 must be estimated first (call estimate_s1_from_frames)")
    
    if set_pmt.time_drift is None:
        raise ValueError("time_drift must be set (call set_transport_properties)")

    t_starts, t_ends, durations = [], [], []
    
    # Compute max files to process (rounds up to complete files)
    if max_frames is not None:
        max_files = int(np.ceil(max_frames / set_pmt.nframes))
        waveforms = itertools.islice(iter_waveforms(set_pmt), max_files)
        print(f"  S2 estimation: processing {max_files} files (~{max_frames} frames)")
    else:
        waveforms = iter_waveforms(set_pmt)
    
    for wf in waveforms:
        # Handle FastFrame waveforms: process each frame individually
        if wf.ff and wf.nframes > 1:
            for frame_idx in range(wf.nframes):
                try:
                    # Extract single frame
                    from .dataIO import extract_single_frame
                    single_wf = extract_single_frame(wf, frame_idx)
                    
                    t_start, t_end = s2_window_pipeline(
                        single_wf, t_s1, set_pmt.time_drift,
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
        else:
            # Single frame waveform
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
                   s2_duration_cuts: tuple = (5, 25),
                   threshold_s2: float = 0.8,
                   max_frames: int = 200,
                   method: str = 'percentile') -> Run:
    """
    Estimate S2 timing windows for all sets in a run.
    
    Args:
        run: Run object with prepared sets
        s2_duration_cuts: (min, max) duration cuts in µs
        threshold_s2: S2 detection threshold in mV
        max_files: Maximum files per set to process (default: 200)
        method: Method for computing statistics ('percentile' or 'std')
        
    Returns:
        Run with S2 timing statistics in each set's metadata
    """
    updated_sets = []
    
    for set_pmt in run.sets:
        try:
            # Get raw S2 timing data
            t_starts, t_ends, durations = estimate_s2_window(
                set_pmt,
                threshold_s2=threshold_s2,
                max_frames=max_frames
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
                if name != 's2_duration':
                    assert std / mean < 0.2, f"⚠ Warning: High relative error in {name} for {set_pmt.source_dir.name}"
            
            updated_sets.append(replace(set_pmt, metadata=new_metadata))
            
            print(f"✓ {set_pmt.source_dir.name}:")
            print(f"  S2 Start:    {new_metadata['t_s2_start_mean']:.2f} ± {new_metadata['t_s2_start_std']:.2f} µs, rel. error: {100 * new_metadata['t_s2_start_std'] / new_metadata['t_s2_start_mean']:.2f} %")
            print(f"  S2 End:      {new_metadata['t_s2_end_mean']:.2f} ± {new_metadata['t_s2_end_std']:.2f} µs, rel. error: {100 * new_metadata['t_s2_end_std'] / new_metadata['t_s2_end_mean']:.2f} %")
            print(f"  S2 Duration: {new_metadata['s2_duration_mean']:.2f} ± {new_metadata['s2_duration_std']:.2f} µs")

        except Exception as e:
            print(f"⚠ Warning: Failed to process {set_pmt.source_dir.name}: {e}")
            updated_sets.append(set_pmt)  # Keep original if processing fails
            continue
    
    return replace(run, sets=updated_sets)


def populate_run(run: Run, nfiles: Optional[int] = None) -> Run:
    """
    Populate a Run with all measurement sets from subdirectories.
    
    Each subdirectory in run.root_directory becomes a SetPmt.
    FastFrame properties are automatically detected per set.
    
    Args:
        run: Run object with root_directory
        nfiles: Optional limit on files per set
        
    Returns:
        Run with sets populated
    """
    sets = []
    subdirs = [d for d in run.root_directory.iterdir() if (d.is_dir() and 'FieldScan' in d.name)]
    
    for subdir in sorted(subdirs):
        try:
            set_pmt = set_from_dir(subdir, nfiles=nfiles)
            sets.append(set_pmt)
            
            # Log FastFrame info
            ff_info = f"FastFrame ({set_pmt.nframes} frames/file)" if set_pmt.ff else "single-frame"
            print(f"  Loaded: {subdir.name} - {len(set_pmt)} files ({set_pmt.n_waveforms} waveforms) [{ff_info}]")
        except Exception as e:
            print(f"  Warning: Failed to load {subdir.name}: {e}")
    
    return replace(run, sets=sets)