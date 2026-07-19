from typing import Optional, Dict, Union, List
import math

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import replace
import matplotlib.pyplot as plt
from pathlib import Path

from RaTag.core.datatypes import Run
from RaTag.core.paths import get_output_root
from RaTag.core.datatypes import Run, SetPmt, Waveform
from RaTag.core.functional import map_over
from RaTag.core.decorators import *
from RaTag.io.file_ops import iter_waveforms, load_npz_arrays, load_random_waveform
from RaTag.waveform.preprocessing import subtract_pedestal, moving_average, threshold_clip
from RaTag.core.config import TimingConfig

from RaTag.plotting import (
    build_fig_grid, 
    catch_plot_errors,
    plot_set_windows,
    plot_timing_histograms, 
    plot_run_timing_vs_field,
    plot_window_validation,
)
# ====================================================================
# Note for Devs: Use `plot_n_waveforms(set_pmt)` from plotting for deep-dive QA
# ====================================================================


# ================================================================
# 1. Helpers for timing statistics
# ================================================================

def _compute_left_half_std(times_clean: np.ndarray, mode: float) -> float:
    """
    Computes a robust standard deviation for heavily right-skewed distributions 
    by mirroring the variance of the left half of the peak. 
    Used to prevent trailing tails from artificially widening the S2 End window.
    """
    left_vals = times_clean[times_clean <= mode]
    if len(left_vals) > 1:
        return float(np.sqrt(np.mean((left_vals - mode)**2)))
    return float(np.std(times_clean))  # Fallback if distribution is weird


def compute_timing_statistics(times: Union[np.ndarray, List[float]],
                              name: str,
                              pre_cut: Optional[tuple] = None,
                              outlier_sigma: float = 3.0) -> Dict[str, float]:
    """Compute timing statistics with outlier rejection. Safely accepts raw lists."""
    # Convert list to array natively inside the function
    times_arr = np.asarray(times, dtype=np.float32)

    if len(times_arr) == 0:
        return {name: None, f"{name}_std": 0.0}

    if pre_cut is not None:
        times_arr = times_arr[(times_arr >= pre_cut[0]) & (times_arr <= pre_cut[1])]
        if len(times_arr) == 0:
            return {name: None, f"{name}_std": 0.0}
    
    # Outlier rejection
    mean_init = np.nanmean(times_arr)
    std_init = np.nanstd(times_arr)

    if std_init == 0:  # Protect against single-element or identical arrays
        return {name: round(float(mean_init), 3), f"{name}_std": 0.0}
        
    mask = np.abs(times_arr - mean_init) < (outlier_sigma * std_init)
    times_clean = times_arr[mask]

    if len(times_clean) == 0:
        return {name: None, f"{name}_std": 0.0}
    
    # Compute mode from histogram
    n, bins = np.histogram(times_clean, bins=100)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    mode = round(float(cbins[np.argmax(n)]), 3)
    if name == "t_s2_end":
        std = round(_compute_left_half_std(times_clean, mode), 3)
    else:
        std = round(float(np.std(times_clean)), 3)

    return {name: mode, f"{name}_std": std}

# ================================================================
# 2. Vectorized S1 and S2 detection functions
# ================================================================


def find_s1(wf: Waveform, config: TimingConfig) -> np.ndarray:
    """Vectorized S1 detection. Returns array of peak times (or NaN)."""

    wf = subtract_pedestal(wf, config.n_pedestal)
    mask = wf.t < config.s1_t_max
        
    t_sliced = wf.t[mask]
    v_sliced = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
    
    # Boolean matrix of threshold crossings
    above_thresh = v_sliced > config.s1_v_min
    has_peak = np.any(above_thresh, axis=1)
    
    # Find rightmost peak (flip, find first, un-flip)
    first_flipped_idx = np.argmax(np.fliplr(above_thresh), axis=1)
    last_peak_idx = (v_sliced.shape[1] - 1) - first_flipped_idx
    
    # Build results
    s1_times = np.full(wf.nframes, np.nan, dtype=np.float32)
    s1_times[has_peak] = t_sliced[last_peak_idx[has_peak]]
    return s1_times

def find_s2(wf: Waveform, config: TimingConfig, t_min_s2: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized S2 boundary tracking. Returns (starts, ends) arrays."""

    wf = moving_average(wf, window=config.s2_window_ma)
    wf = threshold_clip(wf, threshold=config.bs_threshold)
    mask = wf.t > t_min_s2
    if not np.any(mask):
        nan_arr = np.full(wf.nframes, np.nan, dtype=np.float32)
        return nan_arr, nan_arr
        
    t_sliced = wf.t[mask]
    v_sliced = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
    
    above_thresh = v_sliced > config.s2_threshold
    has_s2 = np.any(above_thresh, axis=1)
    
    # Leftmost (start) and rightmost (end) crossings
    start_idx = np.argmax(above_thresh, axis=1)
    end_idx = (v_sliced.shape[1] - 1) - np.argmax(np.fliplr(above_thresh), axis=1)
    
    starts = np.full(wf.nframes, np.nan, dtype=np.float32)
    ends = np.full(wf.nframes, np.nan, dtype=np.float32)
    
    starts[has_s2] = t_sliced[start_idx[has_s2]]
    ends[has_s2] = t_sliced[end_idx[has_s2]]
    return starts, ends

# ============================================================================
# SET-LEVEL WORKFLOW (One-Pass Logic)
# ============================================================================

@allow_force
@load_cached_metadata(target_attr='t_s2_end')
@load_cached_npz(signal_type='timing')
@require_attributes('time_drift')
@write_metadata(target_attr='t_s2_end')
@write_npz_arrays(signal_type='timing')
@limit_frames
def resolve_set_timing(set_pmt: SetPmt, 
                       max_files: int, 
                       config: TimingConfig = TimingConfig(),
                       force: bool = False) -> tuple[SetPmt, dict]:
    """Resolves S1 and S2 timing for a single set. Returns updated SetPmt and timing arrays."""
    
    # 1. Buffers 
    out_uids, out_s1, out_s2_starts, out_s2_ends = [], [], [], []
    s1_anchor = -5.0 
    t_drift_margin = (set_pmt.time_drift or 0.0) * config.s2_margin

    # 2. VECTORIZED LOOP
    for wf in iter_waveforms(set_pmt, max_files=max_files, show_progress=True):
        
        t_s1 = find_s1(wf, config=config)
        
        # Update Anchor dynamically
        if not np.all(np.isnan(t_s1)):
            s1_anchor = np.nanmedian(t_s1) 
        
        t_min = s1_anchor + t_drift_margin   # Ensure we start looking for S2 after the expected drift time, with margin (typically 90% of the drift time to account for variations)
        t_s2_start, t_s2_end = find_s2(wf, config=config, t_min_s2=t_min)
        
        # Bulk Append
        out_uids.append(wf.uids)
        out_s1.append(t_s1)
        out_s2_starts.append(t_s2_start)
        out_s2_ends.append(t_s2_end)

    # 3. Dynamic Concatenation & Stats Computation
    arrays = {"uids": np.concatenate(out_uids) if out_uids else np.array([])}
    stats = {}
    
    # Iterate over the raw buffer lists directly
    for name, buffer in [("t_s1", out_s1), 
                         ("t_s2_start", out_s2_starts), 
                         ("t_s2_end", out_s2_ends)]:
        
        # Concatenate once per timing type
        arr_concat = np.concatenate(buffer) if buffer else np.array([])
        
        # Populate both payloads in one shot
        arrays[name] = arr_concat
        stats.update(compute_timing_statistics(arr_concat, name=name))

    return replace(set_pmt, **stats), arrays

# ============================================================================
# PUBLIC APIs (The "map" functions called by the pipelines)
# ============================================================================

def map_time_windows(run: Run, max_frames: int = 500, 
                   config: TimingConfig = TimingConfig(), force: bool = False) -> Run:
    
    """Entry point: Maps timing workflow over all sets in the Run."""
    bound_timing = lambda s: resolve_set_timing(s, max_frames=max_frames, 
                                                 config=config, 
                                                 force=force)
    
    new_sets = map_over(run.sets, bound_timing, catch_errors=True)
    return replace(run, sets=new_sets)

# ============================================================================
# QA & VALIDATION WORKFLOWS
# ============================================================================

@allow_force
@load_cached_plots(subfolder="timing_qa", expected_suffixes=["histograms", "validation", "timing_vs_field"])
@write_plots(subfolder="timing_qa")
def map_timing_plots(run: Run, force: bool = False) -> Run:
    print("\n" + "="*60)
    print(f"GENERATING TIMING PLOTS: {run.run_id}")
    print("="*60)
    
    figs = {}
    
    # ==========================================
    # 1. Histograms Grid
    # ==========================================
    fig_hist, hist_cells = build_fig_grid(run, f"Timing Histograms - {run.run_id}")
    for set_pmt, ax in hist_cells:
        with catch_plot_errors(ax, set_pmt.source_dir.name):
            
            arrays = load_npz_arrays(set_pmt, signal_type='timing')
            plot_timing_histograms(ax, set_pmt.source_dir.name, arrays)
            
    figs["histograms"] = fig_hist
    
    # ==========================================
    # 2. Validation Grid 
    # ==========================================
    fig_val, val_cells = build_fig_grid(run, f"Timing Window Validation - {run.run_id}")
    for set_pmt, ax in val_cells:
        with catch_plot_errors(ax, set_pmt.source_dir.name):
            # A. Abstracted I/O & Sampling
            wf, frame = load_random_waveform(set_pmt)


            # B. Pure Presentation
            plot_window_validation(ax, wf, frame, set_pmt)
            
    figs["validation"] = fig_val
    
    # ==========================================
    # 3. Global Summary
    # ==========================================
    figs["timing_vs_field"] = plot_run_timing_vs_field(run)
    
    return run, figs