"""
X-ray event classification and integration workflow.

This module provides workflows for:
1. Complete set-level X-ray classification workflow (ETL: extract → classify → load)
2. Run-level orchestration with caching
3. X-ray area integration for calibration purposes

Structure mirrors recoil_integration.py and timing_estimation.py:
- Set-level workflows handle complete ETL including immediate persistence
- Run-level functions orchestrate with caching
- Uses FrameProxy iteration for consistency with other workflows
- Frame-level classification functions are in waveform/xray_classification.py
"""

from typing import Optional, Dict
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from RaTag.core.datatypes import SetPmt, Run, S2Areas
from RaTag.core.config import XRayConfig, FitConfig
from RaTag.core.dataIO import iter_frameproxies, save_set_metadata, store_s2area, save_figure, save_run_metadata
from RaTag.core.uid_utils import make_uid, decode_uid
from RaTag.core.functional import apply_workflow_to_run, compute_max_files, map_isotopes_in_run
from RaTag.core.fitting import fit_gaussian_to_histogram, plot_gaussian_fit
from RaTag.alphas.energy_join import  generic_multiiso_workflow
from RaTag.waveform.preprocessing import subtract_pedestal
from RaTag.waveform.xray_classification import classify_xray_in_frame


# ============================================================================
# REJECTION STATISTICS TRACKER
# ============================================================================

class _RejectionTracker:
    """
    Lightweight tracker for X-ray classification rejection statistics.
    
    Encapsulates counting logic for accepted/rejected frames to keep the
    main classification loop clean and readable. This is a local accumulator
    pattern - not global state - which is compatible with functional principles.
    """
    
    def __init__(self):
        self.n_total = 0
        self.n_accepted = 0
        self.n_excessive_s2 = 0
        self.n_insufficient_sep = 0
        self.n_errors = 0
    
    def record(self, is_accepted: bool, rejection_reason: Optional[str] = None) -> None:
        """
        Record a classification result.
        
        Args:
            is_accepted: Whether the frame was accepted
            rejection_reason: Reason for rejection (if not accepted)
        """
        self.n_total += 1
        if is_accepted:
            self.n_accepted += 1
        elif rejection_reason == "excessive_s2":
            self.n_excessive_s2 += 1
        elif rejection_reason == "insufficient_separation":
            self.n_insufficient_sep += 1
    
    def record_error(self) -> None:
        """Record a frame that failed during classification."""
        self.n_total += 1
        self.n_errors += 1
    
    def get_stats(self) -> dict:
        """
        Compile rejection statistics into a dictionary.
        
        Returns:
            Dictionary with counts and rates for all rejection categories
        """
        return {
            'n_total': self.n_total,
            'n_accepted': self.n_accepted,
            'n_rejected_excessive_s2': self.n_excessive_s2,
            'n_rejected_insufficient_sep': self.n_insufficient_sep,
            'n_rejected_errors': self.n_errors,
            'acceptance_rate': self.n_accepted / self.n_total if self.n_total > 0 else 0.0,
            'rejection_rate_excessive_s2': self.n_excessive_s2 / self.n_total if self.n_total > 0 else 0.0,
            'rejection_rate_insufficient_sep': self.n_insufficient_sep / self.n_total if self.n_total > 0 else 0.0,
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of classification results."""
        stats = self.get_stats()
        n_rejected = self.n_total - self.n_accepted
        
        print(f"    ✓ Classified {self.n_total} frames: {self.n_accepted} accepted, {n_rejected} rejected")
        print(f"      - Excessive S2: {self.n_excessive_s2} ({stats['rejection_rate_excessive_s2']:.1%})")
        print(f"      - Insufficient separation: {self.n_insufficient_sep} ({stats['rejection_rate_insufficient_sep']:.1%})")
        if self.n_errors > 0:
            print(f"      - Errors: {self.n_errors}")


# ============================================================================
# DIRECTORY MANAGEMENT HELPERS
# ============================================================================

def _setup_set_directories(set_pmt: SetPmt) -> tuple[Path, Path]:
    """Setup output directories for set-level processing."""
    plots_dir = set_pmt.source_dir.parent / "plots" / "all" / "xrays"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir


def _get_xray_window(set_pmt: SetPmt) -> tuple[float, float]:
    """
    Check preconditions and determine X-ray integration window.
    
    Uses t_s2_start from metadata if available (preferred), otherwise falls back
    to t_s1 + time_drift. Raises ValueError if required metadata is missing.
    
    Args:
        set_pmt: Set to extract timing window from
        
    Returns:
        (t_s1, s2_start) - X-ray window boundaries in µs
        
    Raises:
        ValueError: If t_s1 is missing, or both t_s2_start and time_drift are missing
    """
    # Check that t_s1 is available
    if 't_s1' not in set_pmt.metadata:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing t_s1 - run timing estimation first")
    
    t_s1 = set_pmt.metadata['t_s1']
    
    # Determine S2 start: prefer t_s2_start, fallback to t_s1 + time_drift
    if 't_s2_start' in set_pmt.metadata:
        s2_start = set_pmt.metadata['t_s2_start']
    elif set_pmt.time_drift is not None:
        s2_start = t_s1 + set_pmt.time_drift
        print(f"  ⚠ Using fallback: t_s2_start not in metadata, using t_s1 + time_drift")
    else:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing both t_s2_start and time_drift")
    
    return t_s1, s2_start


# ============================================================================
# SET-LEVEL CLASSIFICATION (with iteration)
# ============================================================================

def _classify_xrays_in_set(set_pmt: SetPmt,
                          t_s1: float,
                          s2_start: float,
                          max_frames: Optional[int],
                          xray_config: XRayConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Classify X-ray events for all frames in a set.
    
    Uses FrameProxy iteration for consistency with other workflows.
    Only returns ACCEPTED events (rejected events are counted but not saved).
    
    Args:
        set_pmt: Source set
        t_s1: S1 time [µs]
        s2_start: Expected S2 start time [µs]
        max_frames: Optional limit on number of frames (None = process all)
        xray_config: Configuration parameters
        
    Returns:
        (uids, areas, rejection_stats) where:
        - uids: UIDs of accepted events only
        - areas: Integrated areas of accepted events only
        - rejection_stats: Dict with per-criterion rejection counts
    """
    print(f"  Classifying X-rays in window: [{t_s1:.2f}, {s2_start:.2f}] µs")
    
    accepted_uids = []
    accepted_areas = []
    tracker = _RejectionTracker()
    
    # Compute how many files to process (rounds up to complete files)
    max_files, actual_frames = compute_max_files(max_frames, set_pmt.nframes)
    
    if max_frames is not None:
        print(f"  Processing {max_files} files (~{actual_frames} frames)")
    
    # Type ignore: iter_frameproxies accepts max_files=None per its signature
    for frame_wf in iter_frameproxies(set_pmt, max_files=max_files):  # type: ignore[arg-type]
        try:
            uid = make_uid(frame_wf.file_seq, frame_wf.frame_idx)
            frame_pmt = frame_wf.load_pmt_frame()  # type: ignore[assignment]
            
            # Frames from load_pmt_frame() are already in correct units (µs, mV)
            # Just need to subtract pedestal
            frame_pmt = subtract_pedestal(frame_pmt, n_points=xray_config.n_pedestal)  # type: ignore[arg-type]
            
            # Classify and integrate
            is_accepted, rejection_reason, area = classify_xray_in_frame(frame_pmt,
                                                                         t_s1, s2_start,
                                                                           xray_config)
            
            if is_accepted:
                accepted_uids.append(uid)
                accepted_areas.append(area)
            
            tracker.record(is_accepted, rejection_reason)
            
        except Exception as e:
            tracker.record_error()
            if tracker.n_errors <= 3:  # Only print first few errors
                print(f"    ⚠ Frame classification failed: {e}")
            continue
    
    if len(accepted_uids) == 0:
        print(f"    ⚠ No frames accepted (all {tracker.n_total} rejected)")
    
    uids = np.array(accepted_uids, dtype=np.uint32)
    areas = np.array(accepted_areas, dtype=np.float32)
    
    # Print summary and return statistics
    tracker.print_summary()
    
    return uids, areas, tracker.get_stats()


# ============================================================================
# SET-LEVEL CLASSIFICATION WORKFLOW (Complete ETL with immediate persistence)
# ============================================================================

def workflow_xray_classification(set_pmt: SetPmt,
                                 max_frames: Optional[int] = None,
                                 xray_config: XRayConfig = XRayConfig()) -> SetPmt:
    """
    Complete X-ray classification workflow for a single set.
    
    1. Classify X-ray events in all frames
    2. Save accepted events to disk (UIDs, areas) using S2Areas format
    3. Update metadata with acceptance/rejection statistics
    
    Prerequisites:
    - Set metadata must contain t_s1
    - Set metadata should contain t_s2_start (fallback to t_s1 + time_drift)
    
    Args:
        set_pmt: Set with timing metadata
        max_frames: Optional limit on number of frames (None = process all)
        xray_config: X-ray classification configuration
        plots_dir: Directory for plots (for compatibility, unused)
        data_dir: Directory for data output
        
    Returns:
        Updated SetPmt with X-ray statistics in metadata
    """
    # Setup directories
    plots_dir, data_dir = _setup_set_directories(set_pmt)
    
    # Check preconditions and get X-ray window
    t_s1, s2_start = _get_xray_window(set_pmt)
    
    # Classify (returns only accepted events + rejection stats)
    uids, areas, rejection_stats = _classify_xrays_in_set(
        set_pmt, t_s1, s2_start, max_frames, xray_config
    )    # Create S2Areas object for accepted X-ray events
    # This allows us to reuse the same I/O infrastructure as recoil S2 integration
    
    xray_areas = S2Areas(source_dir=set_pmt.source_dir,
                         areas=areas, uids=uids,
                         method="xray_classification",
                         params={
                            "xray_window": (t_s1, s2_start),
                            "bs_threshold": xray_config.bs_threshold,
                            "max_area_s2": xray_config.max_area_s2,
                            "min_s2_sep": xray_config.min_s2_sep,
                            "min_s1_sep": xray_config.min_s1_sep,
                            "n_pedestal": xray_config.n_pedestal,
                            "ma_window": xray_config.ma_window,
                            "dt": xray_config.dt,
                        }
                    )
    
    # Save X-ray areas using unified storage function
    store_s2area(xray_areas, set_pmt, data_dir, suffix="xray_areas")
    
    # Update metadata with statistics
    new_metadata = {
        **set_pmt.metadata,
        **rejection_stats,  # Includes all rejection stats from _classify_xrays_in_set
        'xray_area_mean': float(np.mean(areas)) if len(areas) > 0 else np.nan,
        'xray_area_std': float(np.std(areas)) if len(areas) > 0 else np.nan,
    }
    
    updated_set = replace(set_pmt, metadata=new_metadata)
    save_set_metadata(updated_set)
    
    return updated_set


def workflow_xray_area_multiiso(set_pmt: SetPmt,
                                isotope_ranges: Dict[str, tuple]) -> pd.DataFrame:
    """Multi-isotope X-ray area workflow (set-level): load → map → plot."""
    return generic_multiiso_workflow(set_pmt,
                                     data_filename="xray_areas.npz",
                                     value_keys=["xray_areas"],
                                     isotope_ranges=isotope_ranges,
                                     output_suffix="xray_areas_multi",
                                     plot_columns=["xray_areas"],
                                     bins=40)



# ============================================================================
# VALIDATION AND QA
# ============================================================================

def _load_accepted_uids(set_pmt: SetPmt) -> np.ndarray:
    """
    Load accepted X-ray UIDs from stored NPZ file.
    
    Args:
        set_pmt: Set to load results from
        
    Returns:
        Array of accepted UIDs (empty if file doesn't exist)
    """
    data_dir = set_pmt.source_dir.parent / "processed_data" / "all"
    xray_file = data_dir / f"{set_pmt.source_dir.name}_xray_areas.npz"
    
    if not xray_file.exists():
        return np.array([], dtype=np.uint32)
    
    data = np.load(xray_file)
    return data['uids']


def _generate_all_possible_uids(set_pmt: SetPmt) -> np.ndarray:
    """
    Generate all possible UIDs for a set.
    
    Computes the complete UID space: all (file_seq, frame_idx) combinations.
    
    Args:
        set_pmt: Set to generate UIDs for
        
    Returns:
        Array of all possible UIDs for this set
    """
    from RaTag.alphas.energy_map_writer import parse_file_seq_from_name
    
    file_seqs = np.array([parse_file_seq_from_name(fn) for fn in set_pmt.filenames])
    frame_indices = np.arange(set_pmt.nframes)
    
    # Create all possible UIDs using list comprehension
    all_uids = np.array([make_uid(fs, fi) 
                        for fs in file_seqs 
                        for fi in frame_indices], dtype=np.uint32)
    
    return all_uids


def _sample_uids(uids: np.ndarray, n_frames: int) -> list:
    """
    Sample random UIDs and decode to (file_seq, frame_idx) tuples.
    
    Args:
        uids: Array of UIDs to sample from
        n_frames: Number of frames to sample
        
    Returns:
        List of (file_seq, frame_idx) tuples
    """
    if len(uids) == 0:
        return []
    
    n_sample = min(n_frames, len(uids))
    sampled_uids = np.random.choice(uids, size=n_sample, replace=False)
    return [decode_uid(uid) for uid in sampled_uids]


def _sample_xray_frames(set_pmt: SetPmt, n_frames: int = 5) -> tuple[list, list]:
    """
    Sample random accepted and rejected X-ray candidate frames.
    
    Exploits UID system: accepted UIDs are stored, rejected = all_possible - accepted.
    No iteration needed - pure set arithmetic and random sampling.
    
    Args:
        set_pmt: Set to sample from
        n_frames: Number of frames to sample for each category
        
    Returns:
        (accepted_sample, rejected_sample) - lists of (file_seq, frame_idx) tuples
    """
    # Load accepted UIDs
    accepted_uids = _load_accepted_uids(set_pmt)
    
    # Generate all possible UIDs and compute rejected set
    all_uids = _generate_all_possible_uids(set_pmt)
    rejected_uids = np.setdiff1d(all_uids, accepted_uids)
    
    # Sample both categories
    accepted_sample = _sample_uids(accepted_uids, n_frames)
    rejected_sample = _sample_uids(rejected_uids, n_frames)
    
    return accepted_sample, rejected_sample


def validate_xray_classification(run: Run,
                                 n_waveforms: int = 5) -> Run:
    """
    Visual validation of X-ray classification across all sets.
    
    Plots sample waveforms showing accepted vs rejected X-ray candidates.
    This is QA, not computation - doesn't modify the Run.
    
    Args:
        run: Run with X-ray classification completed
        n_waveforms: Number of waveforms to show per category (accepted/rejected)
        
    Returns:
        Same Run (unchanged)
    """
    print("\n" + "="*60)
    print("X-RAY CLASSIFICATION VALIDATION")
    print("="*60)
    
    validation_dir = run.root_directory / "plots" / "all" / "xray_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check if classification was done
        if 'n_accepted' not in set_pmt.metadata:
            print(f"  ⚠ No X-ray classification results - skipping")
            continue
        
        # Sample frames using UID system
        accepted_sample, rejected_sample = _sample_xray_frames(set_pmt, n_frames=n_waveforms)
        
        if len(accepted_sample) == 0 and len(rejected_sample) == 0:
            print(f"  ⚠ No frames to plot")
            continue
        
        # Import here to avoid circular dependency
        from RaTag.plotting import plot_xray_validation
        
        # Generate validation plot
        fig = plot_xray_validation(set_pmt, accepted_sample, rejected_sample,
                                   title=f"X-ray Classification — {set_pmt.source_dir.name}")
        
        # Save
        save_figure(fig, validation_dir / f"{set_pmt.source_dir.name}_xray_validation.png")
        
        plt.close(fig)
        
        print(f"  ✓ Plotted {len(accepted_sample)} accepted, {len(rejected_sample)} rejected")
    
    print(f"\n  📊 Saved validation plots to {validation_dir.relative_to(run.root_directory)}")
    
    return run  # Unchanged run


# ============================================================================
# FITTING AND PLOTTING
# ============================================================================

def fit_xray_areas(run: Run,
                   fit_config: FitConfig = FitConfig()) -> Run:
    """
    Fit Gaussian to combined X-ray area distribution.
    
    Loads aggregated X-ray areas, fits Gaussian, and saves histogram plot.
    
    Args:
        run: Run with X-ray classification and aggregation completed
        fit_config: Fit configuration (bin_cuts, nbins, etc.)
        
    Returns:
        Updated Run with fit results in metadata
    """
    print("\n" + "="*60)
    print("X-RAY AREA FITTING")
    print("="*60)
    
    # Load combined X-ray areas
    data_dir = run.root_directory / "processed_data"
    combined_file = data_dir / f"{run.run_id}_xray_areas_combined.npz"
    
    if not combined_file.exists():
        print(f"  ⚠ No combined X-ray areas found - run aggregation first")
        return run
    
    data = np.load(combined_file)
    xray_areas = data['xray_areas']
    print(f"  📂 Loaded {len(xray_areas)} combined X-ray events")
    
    # Fit Gaussian using existing function
    mean, sigma, ci95, cbins, n, fit_result = fit_gaussian_to_histogram(data=xray_areas,
                                                                         bin_cuts=fit_config.bin_cuts,
                                                                         nbins=fit_config.nbins,
                                                                         exclude_index=fit_config.exclude_index)
    
    print(f"  ✓ Fit: μ={mean:.3f} ± {ci95:.3f} mV·µs, σ={sigma:.3f} mV·µs")
    
    # Plot using existing function
    fig = plot_gaussian_fit(data=xray_areas,
                            bin_cuts=fit_config.bin_cuts,
                            nbins=fit_config.nbins,
                            fit_result=fit_result,
                            title=f'{run.run_id} — X-ray S2 Area Distribution',
                            xlabel='S2 Area (mV·µs)',
                            color='green')
    
    # Save plot
    plots_dir = run.root_directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / f"{run.run_id}_xray_area_histogram.png")
    
    # Save fit results to run metadata JSON
    metadata_dir = run.root_directory / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    metadata_file = metadata_dir / "xray_fit_results.json"
    
    fit_metadata = {
        'xray_area_mean': float(mean),
        'xray_area_sigma': float(sigma),
        'xray_area_ci95': float(ci95),
        'xray_n_events': int(len(xray_areas)),
        'bin_cuts': fit_config.bin_cuts,
        'nbins': fit_config.nbins
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(fit_metadata, f, indent=2)
    
    print(f"  💾 Saved fit results to metadata/xray_fit_results.json")
    
    return run


# ============================================================================
# RUN-LEVEL ORCHESTRATION (with caching)
# ============================================================================

def classify_xrays_in_run(run: Run,
                         range_sets: Optional[slice] = None,
                         max_frames: Optional[int] = None,
                         xray_config: XRayConfig = XRayConfig()) -> Run:
    """
    Classify X-ray events for all sets in a run.
    
    Args:
        run: Run object with timing already estimated
        range_sets: Optional slice to process subset of sets
        max_frames: Optional limit on frames per set (None = process all)
        xray_config: X-ray classification configuration
        
    Returns:
        Updated Run with X-ray statistics in set metadata
    """
    # Filter sets if range specified
    if range_sets is not None:
        filtered_run = replace(run, sets=run.sets[range_sets])
    else:
        filtered_run = run
    
    return apply_workflow_to_run(filtered_run,
                                 workflow_func=workflow_xray_classification,
                                 workflow_name="X-ray classification",
                                 cache_key="n_accepted",
                                 data_file_suffix="xray_areas.npz",
                                 max_frames=max_frames,
                                 xray_config=xray_config)


def run_xray_multiiso(run: Run, 
                      isotope_ranges: dict) -> Run:
    """Run-level wrapper for distributing X-ray areas by isotope (per-set)."""
    return map_isotopes_in_run(run,
                               workflow_func=workflow_xray_area_multiiso,
                               workflow_name="X-ray area isotope mapping",
                               isotope_ranges=isotope_ranges)
