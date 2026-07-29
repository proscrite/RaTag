# RaTag/el_tpc/recoil_workflow.py
import numpy as np
from typing import Optional, Tuple
from dataclasses import replace
from scipy.ndimage import maximum_filter1d

from RaTag.core.datatypes import Run, SetPmt, S2Areas
from RaTag.core.config import IntegrationConfig, FitConfig, TimingConfig
from RaTag.core.paths import get_output_root
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.io.file_ops import iter_waveforms, load_s2areas, load_fit_result
from RaTag.waveform.preprocessing import subtract_pedestal
# from RaTag.core.fitting import fit_set_s2
from RaTag.plotting import (
    plot_s2areas_summary, plot_run_s2_vs_field, 
    catch_plot_errors, build_fig_grid
)
from RaTag.el_tpc.fit_s2_area import fit_s2_crystalball, v_crystalball_right
from RaTag.thgem_tpc.timing_workflow import compute_timing_statistics

# ============================================================================
# 1. PURE MATH & PHYSICS (The Vectorized Engine)
# ============================================================================

def calculate_s2_areas(set_pmt: SetPmt, 
                       max_files: Optional[int] = None, 
                       config: IntegrationConfig = IntegrationConfig()) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-pass 2D integration of S2 areas.
    Returns (areas, uids).
    """
    static_start, static_end = _compute_static_bounds(set_pmt, config)
    print(f"  Integrating set {set_pmt.source_dir.name} with Static Window: [{static_start:.2f}, {static_end:.2f}] µs")

    out_uids, out_areas = [], []

    for wf in iter_waveforms(set_pmt, max_files=max_files):
        # 1. Native 2D Preprocessing
        wf = standard_preprocessing(wf, n_pedestal=int(config.n_pedestal),
                                    ma_window=int(config.ma_window),
                                    threshold=float(config.bs_threshold))

        # 2. Slice the Time Dimension for Integration
        mask = (wf.t >= static_start) & (wf.t <= static_end)
        v_window = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]

        # 3. Native 2D Trapezoidal Integration
        wf_areas = np.trapezoid(v_window, dx=config.dt, axis=1)

        out_uids.append(wf.uids)
        out_areas.append(wf_areas)

    print(f"  🔹 Processed {len(out_areas)} waveforms for S2 area integration.")

    if not out_areas:
        raise ValueError("No waveforms processed.")

    return np.concatenate(out_areas), np.concatenate(out_uids)


def find_s2(wf, config = TimingConfig(), t_start_delay: float = 0, integ_interval: float = 1.5) -> dict:
    # 1. Base parameters & preprocessing
    is_clipped = np.any(wf.v >= config.s2_threshold, axis=1)
    has_no_clips = ~is_clipped

    wf_sub = subtract_pedestal(wf, n_points=config.n_pedestal)
    v_envelope = maximum_filter1d(wf_sub.v, size=config.s2_window_ma, axis=1)
    wf_smooth = replace(wf_sub, v=v_envelope, nframes=wf_sub.nframes)

    mask = wf_smooth.t > config.s2_start_min
    t_sliced = wf_smooth.t[mask]
    v_sliced = wf_smooth.v[:, mask] if wf_smooth.ff else wf_smooth.v[mask][np.newaxis, :]

    # 2. Anchor to Global Maximum & Find Boundaries
    peak_idx = np.argmax(v_sliced, axis=1)
    peak_heights = v_sliced[np.arange(len(v_sliced)), peak_idx]
    peak_times = t_sliced[peak_idx]
        
    dynamic_thresh = (peak_heights * config.s2_fraction)[:, np.newaxis]
    below_thresh = v_sliced <= dynamic_thresh
    idx_2d = np.arange(v_sliced.shape[1])[np.newaxis, :]

    # Left boundary search
    left_mask = idx_2d <= peak_idx[:, np.newaxis]
    valid_left = below_thresh & left_mask
    first_below_left_rev = np.argmax(np.fliplr(valid_left), axis=1)
    start_time = t_sliced[np.clip(v_sliced.shape[1] - 1 - first_below_left_rev, 0, v_sliced.shape[1] - 1)]

    # Right boundary search
    right_mask = idx_2d >= peak_idx[:, np.newaxis]
    valid_right = below_thresh & right_mask
    last_below_right = np.argmax(valid_right, axis=1)
    end_time = t_sliced[np.clip(last_below_right - 1, 0, v_sliced.shape[1] - 1)]

    # 3. Area Integration
    dt = wf_sub.t[1] - wf_sub.t[0]
    t_2d = wf_sub.t[np.newaxis, :]
    # s2_mask = (t_2d >= start_times[:, np.newaxis]) & (t_2d <= end_time[:, np.newaxis])
    s2_mask = (t_2d >= peak_times[:, np.newaxis] + t_start_delay) & (t_2d <= end_time[:, np.newaxis] )  

    s2_areas = np.sum(wf_sub.v * s2_mask, axis=1) * dt

    # 4. COMBINE ALL CUTS VECTORIALLY
    has_valid_left = np.any(valid_left, axis=1)
    has_valid_right = np.any(valid_right, axis=1)

    # Assemble the final acceptance filter mask
    accepted_events = (
        has_no_clips & 
        has_valid_left & 
        has_valid_right & 
        (s2_areas > config.s2_min_area) & 
        (s2_areas < config.s2_max_area)
    )

    
    filtered_areas = s2_areas[accepted_events] if np.sum(accepted_events) > 0 else np.array([])
    filtered_peak_times = peak_times[accepted_events] if np.sum(accepted_events) > 0 else np.array([])
    filtered_start_times = start_time[accepted_events] if np.sum(accepted_events) > 0 else np.array([])
    filtered_end_times = end_time[accepted_events] if np.sum(accepted_events) > 0 else np.array([])
    filtered_uids = wf.uids[accepted_events] if np.sum(accepted_events) > 0 else np.array([])
    return {
        's2_areas': np.atleast_1d(filtered_areas),
        'peak_times': np.atleast_1d(filtered_peak_times),
        'start_times': np.atleast_1d(filtered_start_times),
        'end_times': np.atleast_1d(filtered_end_times),
        'uids': np.atleast_1d(filtered_uids),
        'n_accepted': int(np.sum(accepted_events))
    }
# ============================================================================
# 2. SET-LEVEL ETL (Cached Solver)
# ============================================================================

@allow_force
@load_cached_metadata(target_attr='n_areas_recoil')
@load_cached_npz(signal_type='s2_areas')
# @require_attributes('t_s2_start', 't_s2_end')
@write_metadata(target_attr='n_areas_recoil')
@write_npz_arrays(signal_type='s2_areas')
@limit_frames
def resolve_set_recoils(set_pmt: SetPmt, 
                        max_files: Optional[int] = None, 
                        config: TimingConfig = TimingConfig()) -> tuple[SetPmt, dict]:
    """
    Executes S2 detection and integration and formats the arrays for storage.
    """
    total_frames = 0
    accepted_frames = 0
    accum_areas, accum_uids, accum_starts, accum_ends, accum_peaks = [], [], [], [], []
    for wf in iter_waveforms(set_pmt, max_files=max_files, show_progress=True):
        result_dict = find_s2(wf, config=config, t_start_delay = 0, integ_interval=1.5)
        total_frames += wf.nframes
        accepted_frames += result_dict['n_accepted']
        
        if result_dict['n_accepted'] > 0:
            accum_areas.append(result_dict['s2_areas'])
            accum_uids.append(result_dict['uids'])
            accum_starts.append(result_dict['start_times'])
            accum_ends.append(result_dict['end_times'])
            accum_peaks.append(result_dict['peak_times'])

    retention = float((accepted_frames / total_frames * 100) if total_frames > 0 else 0.0)
    print(f"  {set_pmt.source_dir.name}: {accepted_frames}/{total_frames} events ({retention:.1f}%)")
    
    stats = {}
    area_arrays = {}
    timing_arrays = {}
    if len(result_dict['s2_areas']) > 0:
        start_concat = np.concatenate(accum_starts)
        end_concat = np.concatenate(accum_ends)
        uids_concat = np.concatenate(accum_uids)

        area_arrays = {
            "s2_areas": np.concatenate(accum_areas),
            "uids": uids_concat
        }

        timing_arrays = {
            "uids": uids_concat,
            "t_s2_start": start_concat,
            "t_s2_end": end_concat,
            "t_s2_peak":  np.concatenate(accum_peaks)
        }

        stats.update(compute_timing_statistics(start_concat, name='t_s2_start'))
        stats.update(compute_timing_statistics(end_concat, name='t_s2_end'))

        stats['n_areas_recoil'] = len(uids_concat)
        

    # Update Set Metadata
    file_ops.save_npz_arrays(set_pmt, 'timing', timing_arrays)
    updated_set = replace(set_pmt, **stats)
    
    return updated_set, area_arrays


# ============================================================================
# 3. RUN-LEVEL ORCHESTRATOR
# ============================================================================

def map_recoil_integration(run: Run, 
                    max_frames: Optional[int] = None, 
                    config: TimingConfig = TimingConfig(),
                    force: bool = False) -> Run:
    """Entry point: Maps the Recoil Integration workflow across all sets."""
    print("\n" + "="*60)
    print(f"INTEGRATING S2 RECOILS: {run.run_id}")
    print("="*60)

    bound_recoils = lambda s: resolve_set_recoils(s, max_frames=max_frames, config=config, force=force)
    
    # Execute the map
    updated_sets = map_over(run.sets, bound_recoils, catch_errors=True)
    
    return replace(run, sets=updated_sets)


# ============================================================================
#  4. Fitting workflow functions
# ============================================================================

@allow_force
@load_cached_metadata(target_attr='area_s2_fit_success')
@load_cached_fit(suffix='s2_areas_hist_fit')
@require_attributes('n_areas_recoil')
@write_metadata(target_attr='area_s2_fit_success')
@write_fit(suffix='s2_areas_hist_fit')
def resolve_set_s2_fit(set_pmt: SetPmt, 
                       config: FitConfig = FitConfig()) -> Tuple[SetPmt, Any]:
    """
    Loads dense S2 areas from disk, executes the statistical fit, 
    and updates the SetPmt metadata.
    """
    s2_areas = load_s2areas(set_pmt)
    if len(s2_areas.areas) == 0:
        return replace(set_pmt, area_s2_fit_success=False), None
    
    try:
        result = fit_s2_crystalball(s2_areas.areas, bin_cuts=config.bin_cuts,
                                    nbins=config.nbins, max_lower_bound=config.max_lower_bound,
                                    smooth=config.smooth )
        metadata_updates = {
            'area_s2_mean': result['peak_position'],
            'area_s2_ci95': result['ci95'],
            'area_s2_sigma': result['sigma'],
            'area_s2_fit_success': True,
            's2_background_bound': result['lower_bound']
        }
        
        print(f"  ✓ Fit: μ={result['peak_position']:.3f} ± {result['ci95']:.3f} mV·µs")
        
        # Return the updated SetPmt and the raw lmfit model for the @persist_fit decorator
        return replace(set_pmt, **metadata_updates), result['result']
        
    except Exception as e:
        print(f"  ✗ Fit failed for {set_pmt.source_dir.name}: {e}")
        return replace(set_pmt, area_s2_fit_success=False), None

def map_recoil_fits(run: Run, 
                    config: FitConfig = FitConfig(), 
                    force: bool = False) -> Run:
    """Entry point: Maps the statistical fitting workflow across all sets."""
    print("\n" + "="*60)
    print(f"FITTING S2 AREA DISTRIBUTIONS: {run.run_id}")
    print("="*60)

    bound_fitter = lambda s: resolve_set_s2_fit(s, config=config, force=force)
    
    updated_sets = map_over(run.sets, bound_fitter, catch_errors=False)
    
    return replace(run, sets=updated_sets)


# ============================================================================
# QA & VALIDATION WORKFLOWS
# ============================================================================
@allow_force
@load_cached_plots(subfolder="s2_areas", expected_suffixes=["histograms", "s2_vs_field"])
@write_plots(subfolder="s2_areas")
def map_recoil_plots(run: Run, config: FitConfig = FitConfig(), force: bool = False,) -> tuple[Run, dict]:
    print("\n" + "="*60 + f"\nGENERATING RECOIL S2 AREAS QA PLOTS: {run.run_id}\n" + "="*60)
    
    figs = {}
    fig_hist, grid_cells = build_fig_grid(run, f"S2 Area Fits - {run.run_id}")
    
    for set_pmt, ax in grid_cells:
        with catch_plot_errors(ax, set_pmt.source_dir.name): # This simply adds an error message to the plot instead of crashing
            s2_areas = load_s2areas(set_pmt)
            
            fit_model = None
            if set_pmt.area_s2_fit_success:
                fit_path = get_output_root(set_pmt.source_dir.parent) / "fits" / f"{set_pmt.source_dir.name}_s2_areas_hist_fit.json"
                fit_model = load_fit_result(fit_path, funcdefs={'v_crystalball_right': v_crystalball_right})
                print(f"  Loaded fit model for {set_pmt.source_dir.name} with lower_bound={set_pmt.s2_background_bound}")
            plot_s2areas_summary(ax=ax, set_name=set_pmt.source_dir.name, 
                                 s2_areas=s2_areas, 
                                 bin_cuts = config.bin_cuts,
                                 fit_model=fit_model,
                                 lower_bound=set_pmt.s2_background_bound)

    figs["histograms"] = fig_hist
    figs["s2_vs_field"] = plot_run_s2_vs_field(run)
        
    return run, figs