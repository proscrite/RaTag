import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

from core.dataIO import iter_frames, save_set_metadata, load_set_metadata
from core.datatypes import SetPmt, Run
from waveform.s1s2_detection import detect_s1_in_frame, detect_s2_in_frame
from plotting import plot_time_histograms

# ============================================================================
# SHARED UTILITIES (private)
# ============================================================================

def _extract_timing_from_frames(set_pmt: SetPmt,
                                max_frames: int,
                                detector_func,
                                **detector_kwargs) -> np.ndarray:
    """
    Generic function to extract timing data from frames.

    1. Read: iter_frames() - unified frame iteration
    2. Extract: detector_func() - frame-level detection
    3. Transform: collect results into array
    
    Args:
        set_pmt: Set to process
        max_frames: Target number of frames
        detector_func: Frame-level detection function
        **detector_kwargs: Arguments for detector_func
        
    Returns:
        Array of detected times (may contain None values)
    """

    # Compute how many files to process (rounds up to complete files)
    max_files = int(np.ceil(max_frames / set_pmt.nframes))
    actual_frames = max_files * set_pmt.nframes
    
    print(f"  Processing {max_files} files (~{actual_frames} frames)")
    
    # Iterate over frames and apply detector
    results = []
    for frame_wf in iter_frames(set_pmt, max_files=max_files):
        result = detector_func(frame_wf, **detector_kwargs)
        if result is not None:
            results.append(result)
    
    if not results:
        raise ValueError(f"No valid detections in {set_pmt.source_dir.name}")
    
    return np.array(results)

def _compute_timing_statistics(times: np.ndarray,
                               name: str,
                               pre_cut: Optional[tuple] = None,
                               outlier_sigma: float = 3.0) -> dict:
    """Compute timing statistics with outlier rejection."""
    if pre_cut is not None:
        times = times[(times >= pre_cut[0]) & (times <= pre_cut[1])]
    
    # Outlier rejection
    mean_init = np.mean(times)
    std_init = np.std(times)
    mask = np.abs(times - mean_init) < (outlier_sigma * std_init)
    times_clean = times[mask]
    
    # Compute mode from histogram
    n, bins = np.histogram(times_clean, bins=50)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    mode = round(cbins[np.argmax(n)], 3)
    std = round(np.std(times_clean), 3)
    
    print(f"  → {name} = {mode} ± {std} µs (from {len(times_clean)} frames)")
    
    return {name: mode, f"{name}_std": std}


def _estimate_timing_in_run(run: Run,
                           workflow_func,
                           param_name: str,
                           **workflow_kwargs) -> Run:
    """
    Generic run-level timing estimation with caching.
    
    Args:
        run: Run to process
        workflow_func: Set-level workflow function (e.g., s1_set_workflow)
        param_name: Name for logging (e.g., "s1")
        **workflow_kwargs: Arguments passed to workflow_func
        
    Returns:
        Updated Run
    """
    print("\n" + "="*60)
    print(f"{param_name.upper()} TIMING ESTIMATION")
    print("="*60)
    
    # Setup directories
    plots_dir = run.root_directory / "plots" / f"{param_name}_timing"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = run.root_directory / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    updated_sets = []
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check cache (metadata contains timing stats)
        cache_key = f"t_{param_name}" if param_name == "s1" else f"t_{param_name}_start"
        data_file = data_dir / f"{set_pmt.source_dir.name}_{param_name}.npz"

        loaded = load_set_metadata(set_pmt)
        if loaded and cache_key in loaded.metadata and data_file.exists():
            print(f"  📂 Loaded from cache")
            updated_sets.append(loaded)
            continue
        
        # Run complete workflow
        try:
            updated_set = workflow_func(set_pmt,
                                       plots_dir=plots_dir,
                                       data_dir=data_dir,
                                       **workflow_kwargs)
            updated_sets.append(updated_set)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            updated_sets.append(set_pmt)
    
    return replace(run, sets=updated_sets)



# ============================================================================
# GENERIC SIDE EFFECTS (modular)
# ============================================================================

def save_timing_results(set_pmt: SetPmt, 
                       timing_data: Union[np.ndarray, dict],
                       data_dir: Path,
                       signal_type: str) -> None:
    """
    Save timing results to disk (metadata + raw data).
    
    Args:
        set_pmt: SetPmt with updated metadata
        timing_data: Raw timing data (array for S1, dict for S2)
        data_dir: Directory for processed data
        signal_type: "s1" or "s2"
    """
    # Save metadata
    save_set_metadata(set_pmt)
    
    # Save raw data as npz (consistent format)
    data_file = data_dir / f"{set_pmt.source_dir.name}_{signal_type}.npz"
    
    if isinstance(timing_data, np.ndarray):
        # S1: single array
        np.savez(data_file, times=timing_data)
    else:
        # S2: dict with multiple arrays
        np.savez(data_file, **timing_data)
    
    print(f"    💾 Saved to {data_file.name}")

# ============================================================================
# S1 COMPUTATION (pure)
# ============================================================================

def compute_s1(set_pmt: SetPmt,
               max_frames: int = 200,
               threshold_s1: float = 1.0) -> tuple[SetPmt, np.ndarray]:
    """
    Compute S1 timing for a single set (pure computation).
    
    Returns:
        (updated_set, s1_times) - Set with metadata AND raw timing array
    """
    print(f"  Computing S1...")
    
    s1_times = _extract_timing_from_frames(set_pmt,
                                           max_frames=max_frames,
                                           detector_func=detect_s1_in_frame,
                                           threshold_s1=threshold_s1)
    
    s1_times = s1_times[s1_times < -0.5]
    
    stats = _compute_timing_statistics(s1_times, 
                                      name="t_s1", 
                                      outlier_sigma=3.0)
    
    new_metadata = {**set_pmt.metadata, **stats}
    updated_set = replace(set_pmt, metadata=new_metadata)
    
    return updated_set, s1_times


# ============================================================================
# S2 COMPUTATION (pure)
# ============================================================================

def compute_s2(set_pmt: SetPmt,
               max_frames: int = 500,
               threshold_s2: float = 0.8,
               window_size: int = 9,
               threshold_bs: float = 0.02,
               s2_duration_cuts: tuple = (3, 35)) -> tuple[SetPmt, dict]:
    """
    Compute S2 timing for a single set (pure computation).
    
    Returns:
        (updated_set, s2_data) - Set with metadata AND raw timing dict
    """
    # Validate prerequisites
    t_s1 = set_pmt.metadata.get("t_s1")
    if t_s1 is None:
        raise ValueError("t_s1 must be estimated first")
    
    if set_pmt.time_drift is None:
        raise ValueError("time_drift must be set")
    
    expected_s2_start = t_s1 + set_pmt.time_drift
    print(f"  Computing S2 (expected start: {expected_s2_start:.2f} µs)...")
    
    s2_boundaries = _extract_timing_from_frames(set_pmt,
                                                max_frames=max_frames,
                                                detector_func=detect_s2_in_frame,
                                                t_s1=t_s1,
                                                t_drift=set_pmt.time_drift,
                                                threshold_s2=threshold_s2,
                                                window_size=window_size,
                                                threshold_bs=threshold_bs)
    
    t_starts = np.array([b[0] for b in s2_boundaries])
    t_ends = np.array([b[1] for b in s2_boundaries])
    durations = t_ends - t_starts
    
    timing_data = [
        ("t_s2_start", t_starts, (expected_s2_start * 0.8, expected_s2_start * 1.3)),
        ("t_s2_end", t_ends, (expected_s2_start * 1.2, 35)),
        ("s2_duration", durations, s2_duration_cuts)
    ]
    
    new_metadata = {**set_pmt.metadata}
    for name, data, cuts in timing_data:
        stats = _compute_timing_statistics(data, name, pre_cut=cuts)
        new_metadata.update(stats)
    
    s2_data = {
        't_s2_start': t_starts, 
        't_s2_end': t_ends,     
        's2_duration': durations
    }
    
    return replace(set_pmt, metadata=new_metadata), s2_data


# ============================================================================
# COMPLETE SET-LEVEL WORKFLOWS (composable)
# ============================================================================

def workflow_s1_set(set_pmt: SetPmt,
                    max_frames: int = 200,
                    threshold_s1: float = 1.0,
                    plots_dir: Optional[Path] = None,
                    data_dir: Optional[Path] = None) -> SetPmt:
    """
    Complete S1 workflow for a single set: compute → save → plot.
    
    Use this for interactive work on single sets.
    """
    # Compute
    updated_set, s1_times = compute_s1(set_pmt,
                                       max_frames=max_frames,
                                       threshold_s1=threshold_s1)
    
    # Default directories
    if data_dir is None:
        data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if plots_dir is None:
        plots_dir = set_pmt.source_dir.parent / "plots" / "s1_timing"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    save_timing_results(updated_set, s1_times, data_dir, "s1")
    
    # Plot
    fig = plot_time_histograms(s1_times, 
                               title=f"{'S1'} - {set_pmt.source_dir.name}",
                               mean=updated_set.metadata.get("t_s1", None),
                               std=updated_set.metadata.get("t_s1_std", None),
                               xlabel = "Time (µs)", color='blue', ax = None)

    fig.savefig(plots_dir / f"{set_pmt.source_dir.name}_s1.png")
    plt.close(fig)
    return updated_set


def workflow_s2_set(set_pmt: SetPmt,
                    max_frames: int = 500,
                    threshold_s2: float = 0.8,
                    s2_duration_cuts: tuple = (3, 35),
                    plots_dir: Optional[Path] = None,
                    data_dir: Optional[Path] = None) -> SetPmt:
    """
    Complete S2 workflow for a single set: compute → save → plot.
    
    Use this for interactive work on single sets.
    """
    # Compute
    updated_set, s2_data = compute_s2(set_pmt,
                                      max_frames=max_frames,
                                      threshold_s2=threshold_s2,
                                      s2_duration_cuts=s2_duration_cuts)
    
    # Default directories
    if data_dir is None:
        data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if plots_dir is None:
        plots_dir = set_pmt.source_dir.parent / "plots" / "s2_timing"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    save_timing_results(updated_set, s2_data, data_dir, "s2")
    
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    for a, time_data in zip(ax, ['t_s2_start', 't_s2_end', 's2_duration']):
        plot_time_histograms(s2_data[time_data], 
                             title=f"{time_data.replace('t_', ' ').replace('_', ' ').title()} - {set_pmt.source_dir.name}",
                             mean=updated_set.metadata.get(f"{time_data}", None),
                             std=updated_set.metadata.get(f"{time_data}_std", None),
                             xlabel = "Time (µs)", color='blue', ax = a)

    fig.savefig(str(plots_dir / f"{set_pmt.source_dir.name}_s2.png"))
    plt.close(fig)

    return updated_set


# ============================================================================
# RUN-LEVEL WORKFLOWS (simple iteration with helper)
# ============================================================================

def estimate_s1_in_run(run: Run,
                       max_frames: int = 200,
                       threshold_s1: float = 1.0) -> Run:
    """Estimate S1 timing for all sets in a run."""
    return _estimate_timing_in_run(run,
                                   workflow_func=workflow_s1_set,
                                   param_name="s1",
                                   max_frames=max_frames,
                                   threshold_s1=threshold_s1)


def estimate_s2_in_run(run: Run,
                       max_frames: int = 500,
                       threshold_s2: float = 0.8,
                       s2_duration_cuts: tuple = (3, 35)) -> Run:
    """Estimate S2 timing for all sets in a run."""
    return _estimate_timing_in_run(run,
                                   workflow_func=workflow_s2_set,
                                   param_name="s2",
                                   max_frames=max_frames,
                                   threshold_s2=threshold_s2,
                                   s2_duration_cuts=s2_duration_cuts)