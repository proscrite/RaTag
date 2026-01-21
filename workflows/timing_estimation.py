import numpy as np # type: ignore
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union, Dict

from RaTag.core.dataIO import iter_frameproxies, save_set_metadata, load_set_metadata, store_isotope_df, save_figure
from RaTag.core.datatypes import SetPmt, Run
from RaTag.core.uid_utils import make_uid
from RaTag.core.functional import apply_workflow_to_run, map_isotopes_in_run, compute_max_files
from RaTag.alphas.energy_join import map_results_to_isotopes, generic_multiiso_workflow
from RaTag.waveform.s1s2_detection import detect_s1_in_frame, detect_s2_in_frame
from RaTag.plotting import plot_time_histograms, plot_n_waveforms, plot_timing_vs_drift_field, plot_grouped_histograms
# ============================================================================
# SHARED UTILITIES (private)
# ============================================================================

def _extract_timing_from_frames(set_pmt: SetPmt,
                                max_frames: int,
                                detector_func,
                                **detector_kwargs) -> tuple[np.ndarray, np.ndarray]:
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
    max_files, actual_frames = compute_max_files(max_frames, set_pmt.nframes)
    
    print(f"  Processing {max_files} files (~{actual_frames} frames)")
    
    # Iterate over frames and apply detector
    results = []
    uids = []

    # for frame_wf in iter_frames(set_pmt, max_files=max_files):
    for frame_wf in iter_frameproxies(set_pmt, chunk_dir=None, max_files=max_files):

        uid = make_uid(frame_wf.file_seq, frame_wf.frame_idx)
        frame_pmt = frame_wf.load_pmt_frame()
        val = detector_func(frame_pmt, **detector_kwargs)
        if val is not None:
            uids.append(uid)
            results.append(val)
    
    if not results:
        raise ValueError(f"No valid detections in {set_pmt.source_dir.name}")
    
    if len(uids) == 0:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)   # keep dtype explicit
    return np.array(uids, dtype=np.uint64), np.array(results)


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
    n, bins = np.histogram(times_clean, bins=100)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    mode = round(cbins[np.argmax(n)], 3)
    std = round(np.std(times_clean), 3)
    
    print(f"  → {name} = {mode} ± {std} µs (from {len(times_clean)} frames)")
    
    return {name: mode, f"{name}_std": std}


# ============================================================================
# SET-LEVEL WORKFLOWS (complete ETL with side effects)
# ============================================================================

def save_timing_results(set_pmt: SetPmt,
                        uids: np.ndarray, 
                       timing_data: Union[np.ndarray, dict],
                       data_dir: Path,
                       signal_type: str) -> None:
    """
    Save timing results to disk (metadata + raw data).
    
    Args:
        set_pmt: SetPmt with updated metadata
        timing_data: Raw timing data (array for S1, dict for S2)
        data_dir: Directory for processed data (base directory)
        signal_type: "s1" or "s2"
    """
    # Save metadata (at root level)
    save_set_metadata(set_pmt)
    
    # Save raw data as npz in all/ subdirectory
    all_dir = data_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    data_file = all_dir / f"{set_pmt.source_dir.name}_{signal_type}.npz"
    
    if isinstance(timing_data, np.ndarray):
        # S1: single array
        np.savez_compressed(data_file, uids=uids.astype(np.uint32), t_s1=timing_data)
        # np.savez(data_file, times=timing_data)
    else:
        # S2: dict with multiple arrays
        np.savez_compressed(data_file, uids=uids.astype(np.uint32), **timing_data)
    
    print(f"    💾 Saved to {data_file.relative_to(data_dir.parent)}")


def _plot_exists(plot_path: Path) -> bool:
    """
    Helper to check whether a plot file already exists on disk.

    Returns True if the file exists (so plotting can be skipped), False otherwise.
    """
    try:
        return plot_path.exists()
    except Exception:
        # Be conservative: if we cannot determine, assume we should regenerate
        return False

# ============================================================================
# S1 COMPUTATION (pure)
# ============================================================================

def compute_s1(set_pmt: SetPmt,
               max_frames: int = 200,
               threshold_s1: float = 1.0) -> tuple[SetPmt, np.ndarray, np.ndarray]:
    """
    Compute S1 timing for a single set (pure computation).
    
    Returns:
        (updated_set, s1_times) - Set with metadata AND raw timing array
    """
    print(f"  Computing S1...")
    
    uids, s1_times = _extract_timing_from_frames(set_pmt,
                                           max_frames=max_frames,
                                           detector_func=detect_s1_in_frame,
                                           threshold_s1=threshold_s1)
    
    # Filter both uids and s1_times with the same mask
    mask = s1_times < -2.5
    uids = uids[mask]
    s1_times = s1_times[mask]
    
    stats = _compute_timing_statistics(s1_times, 
                                      name="t_s1", 
                                      outlier_sigma=3.0)
    
    new_metadata = {**set_pmt.metadata, **stats}
    updated_set = replace(set_pmt, metadata=new_metadata)
    
    return updated_set, s1_times, uids


# ============================================================================
# S2 COMPUTATION (pure)
# ============================================================================

def compute_s2(set_pmt: SetPmt,
               max_frames: int = 500,
               threshold_s2: float = 0.8,
               window_size: int = 9,
               threshold_bs: float = 0.02,
               s2_duration_cuts: tuple = (3, 35)) -> tuple[SetPmt, dict, np.ndarray]:
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
    
    uids, s2_boundaries = _extract_timing_from_frames(set_pmt,
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
    updated_set = replace(set_pmt, metadata=new_metadata)
    return updated_set, s2_data, uids


# ============================================================================
# COMPLETE SET-LEVEL WORKFLOWS (composable)
# ============================================================================

def workflow_s1_timing(set_pmt: SetPmt,
                    max_frames: int = 200,
                    threshold_s1: float = 1.0) -> SetPmt:
    """Complete S1 workflow for a single set: compute → save → plot."""
    
    # Compute
    updated_set, s1_times, uids_s1 = compute_s1(set_pmt,
                                       max_frames=max_frames,
                                       threshold_s1=threshold_s1)
    
    # Default directories
    data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = set_pmt.source_dir.parent / "plots" / "all" / "t_s1"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving S1 timing results in {data_dir.relative_to(set_pmt.source_dir.parent)}")
    # Save
    save_timing_results(updated_set, uids_s1, s1_times, data_dir, signal_type='t_s1')
    
    # Plot
    fig = plot_time_histograms(s1_times, 
                            title=f"{'S1'} - {set_pmt.source_dir.name}",
                            mean=updated_set.metadata.get("t_s1", None),
                            std=updated_set.metadata.get("t_s1_std", None),
                            xlabel = "Time (µs)", color='blue', ax = None)

    save_figure(fig, plots_dir / f"{set_pmt.source_dir.name}_t_s1.png")
    plt.close(fig)
    return updated_set

def workflow_s2_timing(set_pmt: SetPmt,
                    max_frames: int = 500,
                    threshold_s2: float = 0.8,
                    s2_duration_cuts: tuple = (3, 35),) -> SetPmt:
    """Complete S2 workflow for a single set: compute → save → plot."""
    # Compute
    updated_set, s2_data, uids_s2 = compute_s2(set_pmt,
                                      max_frames=max_frames,
                                      threshold_s2=threshold_s2,
                                      s2_duration_cuts=s2_duration_cuts)
    
    # Default directories
    data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = set_pmt.source_dir.parent / "plots" / "all" / "t_s2"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    save_timing_results(updated_set, uids_s2, s2_data, data_dir, signal_type='t_s2')

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    for a, time_data in zip(ax, ['t_s2_start', 't_s2_end', 's2_duration']):
        plot_time_histograms(s2_data[time_data], 
                             title=f"{time_data.replace('t_', ' ').replace('_', ' ').title()} - {set_pmt.source_dir.name}",
                             mean=updated_set.metadata.get(f"{time_data}", None),
                             std=updated_set.metadata.get(f"{time_data}_std", None),
                             xlabel = "Time (µs)", color='blue', ax = a)

    save_figure(fig, plots_dir / f"{set_pmt.source_dir.name}_t_s2.png")
    plt.close(fig)
    return updated_set


def workflow_s1_multiiso(set_pmt: SetPmt,
                         isotope_ranges: Dict[str, tuple]) -> pd.DataFrame:
    """Multi-isotope S1 workflow: load → map → plot."""
    return generic_multiiso_workflow(set_pmt,
                                     data_filename="t_s1.npz",
                                     value_keys=["t_s1"],
                                     isotope_ranges=isotope_ranges,
                                     output_suffix="t_s1_multi",
                                     plot_columns=["t_s1"],
                                     bins=40)


def workflow_s2_multiiso(set_pmt: SetPmt,
                         isotope_ranges: Dict[str, tuple]) -> pd.DataFrame:
    """Multi-isotope S2 workflow: load → map → plot."""
    return generic_multiiso_workflow(set_pmt,
                                     data_filename="t_s2.npz",
                                     value_keys=["t_s2_start", "t_s2_end"],
                                     isotope_ranges=isotope_ranges,
                                     output_suffix="t_s2_multi",
                                     plot_columns=["t_s2_start", "t_s2_end"],
                                     bins=40)


# ============================================================================
# RUN-LEVEL WORKFLOWS (simple iteration with helper)
# ============================================================================

def estimate_s1_in_run(run: Run,
                       max_frames: int = 200,
                       threshold_s1: float = 1.0) -> Run:
    """Estimate S1 timing for all sets in a run."""
    return apply_workflow_to_run(run,
                                 workflow_func=workflow_s1_timing,
                                 workflow_name="S1 timing estimation",
                                 cache_key="t_s1",
                                 data_file_suffix="t_s1.npz",
                                 max_frames=max_frames,
                                 threshold_s1=threshold_s1)


def estimate_s2_in_run(run: Run,
                       max_frames: int = 500,
                       threshold_s2: float = 0.8,
                       s2_duration_cuts: tuple = (3, 35)) -> Run:
    """Estimate S2 timing for all sets in a run."""
    return apply_workflow_to_run(run,
                                 workflow_func=workflow_s2_timing,
                                 workflow_name="S2 timing estimation",
                                 cache_key="t_s2_start",
                                 data_file_suffix="t_s2.npz",
                                 max_frames=max_frames,
                                 threshold_s2=threshold_s2,
                                 s2_duration_cuts=s2_duration_cuts)


# ============================================================================
# MULTI-ISOTOPE RUN-LEVEL WORKFLOWS
# ============================================================================

def run_s1_multiiso(run: Run, isotope_ranges: dict) -> Run:
    """Run-level wrapper for distributing S1 timings by isotope."""
    return map_isotopes_in_run(run,
                               workflow_func=workflow_s1_multiiso,
                               workflow_name="S1 isotope mapping",
                               isotope_ranges=isotope_ranges)


def run_s2_multiiso(run: Run, isotope_ranges: dict) -> Run:
    """Run-level wrapper for distributing S2 timings by isotope."""
    return map_isotopes_in_run(run,
                               workflow_func=workflow_s2_multiiso,
                               workflow_name="S2 isotope mapping",
                               isotope_ranges=isotope_ranges)

# ============================================================================
# VALIDATION STEP WITH PLOTTING (pure QA)
# ============================================================================

def validate_timing_windows(run: Run, n_waveforms: int = 5, force: bool = False) -> Run:
    """
    Visual validation of timing windows across all sets.
    
    Plots sample waveforms with S1/S2 windows overlaid.
    This is QA, not computation - doesn't modify the Run.
    
    Args:
        run: Run with timing estimates
        n_waveforms: Number of random waveforms to plot per set
        force: If True, regenerate plots even if the PNG already exists (overwrites)
        
    Returns:
        Same Run (unchanged)
    """
    print("\n" + "="*60)
    print("TIMING VALIDATION")
    print("="*60)
    
    validation_dir = run.root_directory / "plots" / "all" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check if timing is estimated
        if "t_s1" not in set_pmt.metadata or "t_s2_start" not in set_pmt.metadata:
            print("  ⚠ Skipping (missing timing estimates)")
            continue
        
        try:
            plot_path = validation_dir / f"{set_pmt.source_dir.name}_validation.png"
            # Skip plotting if the validation image already exists (unless force=True)
            if _plot_exists(plot_path) and not force:
                print(f"  ⏭ Skipping (plot exists): {plot_path.name}")
                continue
            if _plot_exists(plot_path) and force:
                print(f"  ⚡ Force enabled — overwriting existing plot: {plot_path.name}")

            fig, ax = plot_n_waveforms(set_pmt, n_waveforms=n_waveforms)
            save_figure(fig, plot_path)
            print(f"  ✓ Saved validation plot")
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    return run  # Unchanged run


# ============================================================================
# SUMMARY PLOT OF TIMING (pure QA)
# ============================================================================


def _collect_timing_data(sets: list[SetPmt], 
                        param_names: list[str]) -> tuple[list, dict]:
    """
    Collect timing data from all sets for specified parameters.
    
    Helper function - extracts data with validation.
    
    Args:
        sets: List of SetPmt objects
        param_names: List of parameter names to extract (e.g., ['t_s1', 't_s2_start'])
        
    Returns:
        (drift_fields, timing_dict) where timing_dict maps param -> {mean: [], std: []}
    """
    drift_fields = []
    
    # Initialize storage for each parameter
    timing_dict = {param: {'mean': [], 'std': []} for param in param_names}

    for set_pmt in sets:
        # Check if ALL required parameters are present
        missing = [p for p in param_names 
                  if p not in set_pmt.metadata or set_pmt.metadata[p] is None]
        
        if missing:
            print(f"  ⚠ Skipping {set_pmt.source_dir.name} (missing {missing})")
            continue
        
        # Collect drift field
        drift_fields.append(set_pmt.drift_field)
        
        # Collect each parameter's mean and std
        for param in param_names:
            timing_dict[param]['mean'].append(set_pmt.metadata[param])
            timing_dict[param]['std'].append(set_pmt.metadata.get(f"{param}_std", 0))
        
        print(f"  ✓ {set_pmt.source_dir.name}: E_drift = {set_pmt.drift_field:.1f} V/cm")
    
    # Convert lists to arrays
    drift_fields = np.array(drift_fields)
    for param in param_names:
        timing_dict[param]['mean'] = np.array(timing_dict[param]['mean'])
        timing_dict[param]['std'] = np.array(timing_dict[param]['std'])
    
    return drift_fields, timing_dict


def summarize_timing_vs_field(run: Run, 
                               plots_dir: Optional[Path] = None) -> Run:
    """Create summary plot of timing estimates vs drift field."""
    print("\n" + "="*60)
    print("TIMING VS FIELD SUMMARY")
    print("="*60)
    
    # Set up output directory
    if plots_dir is None:
        plots_dir = run.root_directory / "plots" / "all" / "summary_preparation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all timing parameters
    param_names = ['t_s1', 't_s2_start', 't_s2_end']
    drift_fields, timing_data = _collect_timing_data(run.sets, param_names)
    
    if len(drift_fields) == 0:
        print("  ⚠ No sets with complete timing data - skipping plot")
        return run
    
    # Create plot
    fig, ax = plot_timing_vs_drift_field(
        drift_fields=drift_fields,
        timing_data=timing_data,
        title=f"Timing vs Drift Field - {run.run_id}"
    )
    
    # Save
    output_file = plots_dir / f"{run.run_id}_timing_vs_field.png"
    save_figure(fig, output_file)

    print(f"\n✓ Summary plot saved to {output_file}")
    print(f"  Plotted {len(drift_fields)} sets with complete timing")
    
    return run  # Unchanged