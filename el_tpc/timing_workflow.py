import numpy as np
from dataclasses import replace
from typing import Dict, Any
from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.functional import map_over, compute_max_files
from RaTag.core.decorators import *
from RaTag.el_tpc.waveform_features import find_s1, find_s2, compute_timing_statistics
from RaTag.io.file_ops import iter_waveforms
from RaTag.plotting import plot_combined_timing_histograms, plot_n_waveforms, summarize_timing_vs_field


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
                       window_size: int, 
                       threshold_bs: float, 
                       t_s1_max: float = -2.5) -> tuple[SetPmt, dict]:
    
    # 1. Buffers 
    out_uids, out_s1, out_s2_starts, out_s2_ends = [], [], [], []
    s1_anchor = -5.0 
    t_drift_margin = (set_pmt.time_drift or 0.0) * 0.3

    # 2. THE VECTORIZED LOOP
    for wf in iter_waveforms(set_pmt, max_files=max_files):
        
        t_s1 = find_s1(wf, threshold=threshold_s1, t_max=t_s1_max)
        
        # Update Anchor dynamically
        if not np.all(np.isnan(t_s1)):
            s1_anchor = np.nanmean(t_s1) 
            
        t_s2_start, t_s2_end = find_s2(wf, threshold_s2=threshold_s2, t_min=s1_anchor + t_drift_margin, window_size=window_size, threshold_bs=threshold_bs)
        
        # Bulk Append
        out_uids.append(wf.uids)
        out_s1.append(t_s1)
        out_s2_starts.append(t_s2_start)
        out_s2_ends.append(t_s2_end)

    print(f"t_s1 range: {np.nanmin(out_s1):.2f} to {np.nanmax(out_s1):.2f} µs")
    print(f"t_s2_start range: {np.nanmin(out_s2_starts):.2f} to {np.nanmax(out_s2_starts):.2f} µs")
    print(f"t_s2_end range: {np.nanmin(out_s2_ends):.2f} to {np.nanmax(out_s2_ends):.2f} µs")
    # 3.  Concatenation Helper and statistics computation
    uids, t_s1, t_s2_start, t_s2_end = _concat_buffers(out_uids, out_s1, out_s2_starts, out_s2_ends)

    print(f"Concat result: range {np.nanmin(t_s1):.2f} to {np.nanmax(t_s1):.2f} µs")
    stats = {}
    for name, arr in [("t_s1", t_s1), ("t_s2_start", t_s2_start), ("t_s2_end", t_s2_end)]:
        stats.update(compute_timing_statistics(arr, name=name))

    print(f"Timing stats: {stats}")
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
                   t_s1_max: float = -2.5) -> Run:
    
    """Entry point: Maps timing workflow over all sets in the Run."""
    bound_timing = lambda s: resolve_set_timing(s, max_frames=max_frames, 
                                                 threshold_s1=threshold_s1, 
                                                 threshold_s2=threshold_s2,
                                                 t_s1_max=t_s1_max)
    
    new_sets = map_over(run.sets, bound_timing, catch_errors=True)
    return replace(run, sets=new_sets)

# ============================================================================
# QA & VALIDATION WORKFLOWS
# ============================================================================

@persist_plots(subfolder="timing_qa")
def generate_timing_qa(set_pmt: SetPmt, n_waveforms: int = 5) -> tuple[SetPmt, dict]:
    """Loads timing payload and generates QA figures."""
    
    # 1. Load Data
    data_file = get_output_root(set_pmt.source_dir.parent) / f"{set_pmt.source_dir.name}_timing.npz"
    payload = dict(np.load(data_file)) if data_file.exists() else {}
    
    # 2. Generate Plots (Physics logic only)
    fig_hist = None
    if payload:
        fig_hist, _ = plot_combined_timing_histograms(payload, set_name=set_pmt.source_dir.name)
        
    fig_val, _ = plot_n_waveforms(set_pmt, n_waveforms=n_waveforms)
    
    # 3. Pass to decorator for saving
    figures = {
        "histograms": fig_hist,
        "validation": fig_val
    }
    
    return set_pmt, figures

def map_timing_plots(run: Run, n_waveforms: int = 5) -> Run:
    """Entry point: Maps QA plots across the run and creates the global summary."""
    bound_qa = lambda s: generate_timing_qa(s, n_waveforms=n_waveforms)
    
    run_with_plots = replace(run, sets=map_over(run.sets, bound_qa, catch_errors=True))
    
    # Global summary plot handles its own saving natively
    summarize_timing_vs_field(run_with_plots)
    
    return run_with_plots