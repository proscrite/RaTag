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
    """
    Fallback S1 Detection.
    Uses a strict height cut and a simple fixed-window Area integration 
    as an efficient proxy for pulse width.
    """
    wf_sub = subtract_pedestal(wf, config.n_pedestal)
    
    # 1. Global Search Window
    
    mask = (wf_sub.t >= config.s1_t_min) & (wf_sub.t <= config.s1_t_max)
    
    t_sliced = wf_sub.t[mask]
    v_sliced = wf_sub.v[:, mask] if wf_sub.ff else wf_sub.v[mask][np.newaxis, :]
    
    # 2. Find first peak in the window
    over_thresh = v_sliced > config.s1_v_min
    first_cross_idx = np.argmax(over_thresh, axis=1)

    
    s1_times = t_sliced[first_cross_idx]
    s1_heights = v_sliced[np.arange(len(v_sliced)), first_cross_idx]
    
    dt = wf_sub.t[1] - wf_sub.t[0]
    t_2d = wf_sub.t[np.newaxis, :]
    s1_times_2d = s1_times[:, np.newaxis]
    
    # FIXED local window: +/- 50 ns around the found peak
    local_mask = (t_2d >= s1_times_2d - 0.05) & (t_2d <= s1_times_2d + 0.05)
    v_full = wf_sub.v if wf_sub.ff else wf_sub.v[np.newaxis, :]
    
    # Area = sum of voltages in the window * dt. (Computationally O(N), very fast)
    s1_areas = np.sum(v_full * local_mask, axis=1) * dt
    
    # 4.  Cuts
    valid_height = (s1_heights >= config.s1_v_min) & (s1_heights <= config.s1_v_max)
        
    valid_area = (s1_areas >= 0.0) & (s1_areas <= config.s1_max_area)
    valid_cut = valid_height & valid_area

    s1_times[~valid_cut] = np.nan
    
    return s1_times

def find_s2(wf: Waveform, config: TimingConfig, t_min_s2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Peak-Anchored Vectorized S2 boundary tracking.
    Now includes a heavy macroscopic envelope filter and a strict Minimum Area cut 
    to reject shifted S1s and noise. Returns (starts, ends).
    """
    # 1. HEAVY Macroscopic Smoothing (1500 samples / 300 ns)
    wf_smooth = moving_average(wf, window=config.s2_window_ma)
    wf_clip = threshold_clip(wf_smooth, threshold=config.bs_threshold)
    
    mask = wf_clip.t > t_min_s2
    if not np.any(mask):
        nan_arr = np.full(wf_clip.nframes, np.nan, dtype=np.float32)
        return nan_arr, nan_arr
        
    t_sliced = wf_clip.t[mask]
    v_sliced = wf_clip.v[:, mask] if wf_clip.ff else wf_clip.v[mask][np.newaxis, :]
    
    # We need the raw waveform to compute the true, undiminished S2 area
    v_raw = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
    
    # 2. Anchor to the Global Maximum
    peak_idx = np.argmax(v_sliced, axis=1)
    peak_heights = v_sliced[np.arange(len(v_sliced)), peak_idx]
    peak_times = t_sliced[peak_idx]
    
    has_s2_height = peak_heights > config.s2_threshold
    
    # 3. Dynamic Outward Boundary Search
    dynamic_thresh = np.maximum(peak_heights * config.s2_fraction, config.s2_threshold)[:, np.newaxis]
    below_thresh = v_sliced <= dynamic_thresh
    
    idx_2d = np.arange(v_sliced.shape[1])[np.newaxis, :]
    peak_idx_2d = peak_idx[:, np.newaxis]
    
    # Search Left
    left_mask = idx_2d <= peak_idx_2d
    valid_below_left = below_thresh & left_mask
    first_below_left_rev = np.argmax(np.fliplr(valid_below_left), axis=1)
    start_below_idx = (v_sliced.shape[1] - 1) - first_below_left_rev
    has_start_drop = np.any(valid_below_left, axis=1)
    start_idx = np.where(has_start_drop, np.minimum(start_below_idx + 1, v_sliced.shape[1] - 1), 0)
                         
    # Search Right
    right_mask = idx_2d >= peak_idx_2d
    valid_below_right = below_thresh & right_mask
    end_below_idx = np.argmax(valid_below_right, axis=1)
    has_end_drop = np.any(valid_below_right, axis=1)
    end_idx = np.where(has_end_drop, np.maximum(end_below_idx - 1, 0), v_sliced.shape[1] - 1)
    
    # 4. Area Integration using the RAW waveform
    dt = wf.t[1] - wf.t[0]
    t_2d = wf.t[np.newaxis, :]
    peak_times_2d = peak_times[:, np.newaxis]
    
    local_mask = (t_2d >= peak_times_2d - 1.5) & (t_2d <= peak_times_2d + 1.5)
    v_full = wf.v if wf.ff else wf.v[np.newaxis, :]
    s2_areas = np.sum(v_full * local_mask, axis=1) * dt
    
    widths = t_sliced[end_idx] - t_sliced[start_idx]
    # 5. Apply the Cuts
    # The candidate must be tall enough AND have a massive area (rejecting S1s)
    valid_cut = (
        has_s2_height & 
        (peak_idx > 0) &                            # VETO: We sliced into a falling tail
        (widths >= config.s2_min_width) &           # VETO: The boundaries collapsed
        (s2_areas >= config.s2_min_area) &          # VETO: Too small (shifted S1)
        (s2_areas <= config.s2_max_area) &           # VETO: Too large (X-rays / alphas)
        (t_sliced[start_idx] <= config.s2_start_max)  # VETO: S2 start time too late
    )

    starts = np.full(wf_clip.nframes, np.nan, dtype=np.float32)
    ends = np.full(wf_clip.nframes, np.nan, dtype=np.float32)
    
    starts[valid_cut] = t_sliced[start_idx[valid_cut]]
    ends[valid_cut] = t_sliced[end_idx[valid_cut]]
    
    return starts, ends
# ============================================================================
# SET-LEVEL WORKFLOW 
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
    t_min = s1_anchor + t_drift_margin  # Default minimum S2 search time if no S1 found

    # 2. Find s1 statistically
    for wf in iter_waveforms(set_pmt, max_files=max_files):
        t_s1 = find_s1(wf, config=config)
        out_s1.append(t_s1)
        
    if not np.all(np.isnan(t_s1)):
        s1_anchor = np.nanmedian(t_s1) - np.nanstd(t_s1)  
        t_min = s1_anchor + t_drift_margin   # Ensure we start looking for S2 after the expected drift time, with margin (typically 90% of the drift time to account for variations)

    # 3. Find s2 boundaries
    for wf in iter_waveforms(set_pmt, max_files=max_files):        
        t_s2_start, t_s2_end = find_s2(wf, config=config, t_min_s2=t_min)
        
        # Bulk Append
        out_uids.append(wf.uids)
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
def map_timing_plots(run: Run, force: bool = False) -> Tuple[Run, dict]:
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