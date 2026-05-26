from ast import Tuple
import math

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import replace
import matplotlib.pyplot as plt
from pathlib import Path

from RaTag.core.datatypes import Run
from RaTag.core.paths import get_output_root
from RaTag.io import file_ops

from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.functional import map_over, compute_max_files
from RaTag.core.decorators import *
from RaTag.el_tpc.waveform_features import find_s1, find_s2, compute_timing_statistics
from RaTag.io.file_ops import iter_waveforms

from RaTag.plotting import (
    map_plot_to_grid, 
    plot_set_windows,
    plot_combined_timing_histograms, 
    plot_run_timing_vs_field
)
# ====================================================================
# Note for Devs: Use `plot_n_waveforms(set_pmt)` from plotting for deep-dive QA
# ====================================================================


# ============================================================================
# SET-LEVEL WORKFLOW (One-Pass Logic)
# ============================================================================
def _concat_buffers(*buffers) -> tuple:
    """Helper to concatenate multiple lists of arrays in one line."""
    return tuple(np.concatenate(b) for b in buffers)

@disk_cache(target_attr='t_s2_end')
@require_attributes('time_drift')
@persist_results(signal_type='timing')
@limit_frames
def resolve_set_timing(set_pmt: SetPmt, 
                       max_files: int, 
                       threshold_s1: float, 
                       threshold_s2: float, 
                       window_size: int = 9, 
                       threshold_bs: float = 0.02, 
                       t_s1_max: float = -2.5,
                       t_drift_margin: float = 0.9,
                       force: bool = False) -> tuple[SetPmt, dict]:
    
    # 1. Buffers 
    out_uids, out_s1, out_s2_starts, out_s2_ends = [], [], [], []
    s1_anchor = -5.0 
    t_drift_margin = (set_pmt.time_drift or 0.0) * t_drift_margin

    # 2. VECTORIZED LOOP
    for wf in iter_waveforms(set_pmt, max_files=max_files):
        
        t_s1 = find_s1(wf, threshold=threshold_s1, t_max=t_s1_max)
        
        # Update Anchor dynamically
        if not np.all(np.isnan(t_s1)):
            s1_anchor = np.nanmedian(t_s1) 
        
        t_min = s1_anchor + t_drift_margin   # Ensure we start looking for S2 after the expected drift time, with margin (typically 90% of the drift time to account for variations)
        t_s2_start, t_s2_end = find_s2(wf, threshold_s2=threshold_s2, t_min=t_min, window_size=window_size, threshold_bs=threshold_bs)
        
        # Bulk Append
        out_uids.append(wf.uids)
        out_s1.append(t_s1)
        out_s2_starts.append(t_s2_start)
        out_s2_ends.append(t_s2_end)

    # 3.  Concatenation Helper and statistics computation (not important)
    uids, t_s1, t_s2_start, t_s2_end = _concat_buffers(out_uids, out_s1, out_s2_starts, out_s2_ends)

    stats = {}
    for name, arr in [("t_s1", t_s1), ("t_s2_start", t_s2_start), ("t_s2_end", t_s2_end)]:
        stats.update(compute_timing_statistics(arr, name=name))   # This function is robust to NaNs and empty arrays, returning 0.0 for std in those cases.


    # 5. Dense Payload (One UID array for both S1 and S2, aligned with timing arrays)
    payload = {
        "uids": uids,
        "t_s1": t_s1,
        "t_s2_start": t_s2_start,
        "t_s2_end": t_s2_end,
    }

    return replace(set_pmt, **stats), payload

# ============================================================================
# PUBLIC APIs (The "map" functions called by the pipelines)
# ============================================================================

def map_run_timing(run: Run, max_frames: int = 500, 
                   threshold_s1: float = 1.0, threshold_s2: float = 0.8, 
                   t_s1_max: float = -2.5, force: bool = False) -> Run:
    
    """Entry point: Maps timing workflow over all sets in the Run."""
    bound_timing = lambda s: resolve_set_timing(s, max_frames=max_frames, 
                                                 threshold_s1=threshold_s1, 
                                                 threshold_s2=threshold_s2,
                                                 t_s1_max=t_s1_max, 
                                                 force=force)
    
    new_sets = map_over(run.sets, bound_timing, catch_errors=True)
    return replace(run, sets=new_sets)

# ============================================================================
# QA & VALIDATION WORKFLOWS
# ============================================================================
@persist_plots(subfolder="timing_qa", expected_suffixes=["histograms", "validation", "timing_vs_field"])
def map_timing_plots(run: Run, force: bool = False) -> Tuple[Run, dict]:
    """Generates the Run-level grid dashboards."""
    print("\n" + "="*60)
    print(f"GENERATING TIMING PLOTS: {run.run_id}")
    print("="*60)
    figs = {}
    
    # 1. Histogram Dashboard
    figs["histograms"] = map_plot_to_grid(
        run, plot_combined_timing_histograms, f"Timing Histograms - {run.run_id}"
    )
    
    # 2. Timing vs Field Summary
    figs["timing_vs_field"] = plot_run_timing_vs_field(run)

    # 3. Validation Dashboard
    figs["validation"] = map_plot_to_grid(
            run, plot_set_windows, f"Timing Window Validation - {run.run_id}"
        )
        
    return run, figs