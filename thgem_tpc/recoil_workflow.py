# RaTag/el_tpc/recoil_workflow.py
import numpy as np
from typing import Optional, Tuple
from dataclasses import replace
from scipy.ndimage import maximum_filter1d

from RaTag.core.datatypes import Run, SetPmt, S2Areas, Waveform
from RaTag.core.config import IntegrationConfig, FitConfig, TimingConfig
from RaTag.core.paths import get_output_root
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.io.file_ops import iter_waveforms, load_s2areas, load_fit_result
from RaTag.waveform.preprocessing import subtract_pedestal, threshold_clip, moving_average
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

def _check_s2_clipping(v_window: np.ndarray, max_v: float) -> np.ndarray:
    """Anti-alpha cut: Returns 1D boolean mask of frames avoiding amplifier saturation."""
    return np.max(v_window, axis=1) < max_v

def _find_left_boundary(v_sliced: np.ndarray, t_sliced: np.ndarray, peak_idx: np.ndarray, 
                        peak_heights: np.ndarray, peak_times: np.ndarray, 
                        fraction_left: float, fallback_window: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized search for the left boundary crossing with static fallback."""
    dynamic_thresh_left = (peak_heights * fraction_left)[:, np.newaxis]
    below_thresh_left = v_sliced <= dynamic_thresh_left
    idx_2d = np.arange(v_sliced.shape[1])[np.newaxis, :]

    left_mask = idx_2d <= peak_idx[:, np.newaxis]
    valid_left = below_thresh_left & left_mask
    first_below_left_rev = np.argmax(np.fliplr(valid_left), axis=1)
    
    has_valid_left = np.any(valid_left, axis=1)
    
    start_times = np.where(has_valid_left,
                           t_sliced[np.clip(v_sliced.shape[1] - 1 - first_below_left_rev, 0, v_sliced.shape[1] - 1)],
                           peak_times - fallback_window)

    return start_times, has_valid_left

def _find_right_boundary(v_sliced: np.ndarray, t_sliced: np.ndarray, peak_idx: np.ndarray, 
                         peak_heights: np.ndarray, peak_times: np.ndarray, 
                         fraction_right: float, fallback_window: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized search for the right boundary crossing with static fallback."""
    dynamic_thresh_right = (peak_heights * fraction_right)[:, np.newaxis]
    below_thresh_right = v_sliced <= dynamic_thresh_right
    idx_2d = np.arange(v_sliced.shape[1])[np.newaxis, :]

    right_mask = idx_2d >= peak_idx[:, np.newaxis]
    valid_right = below_thresh_right & right_mask
    last_below_right = np.argmax(valid_right, axis=1)

    has_valid_right = np.any(valid_right, axis=1)

    end_times = np.where(has_valid_right,
                         t_sliced[np.clip(last_below_right - 1, 0, v_sliced.shape[1] - 1)],
                         peak_times + fallback_window)

    return end_times, has_valid_right



def _integrate_s2_areas(wf_sub: Waveform, start_times: np.ndarray, end_times: np.ndarray, t_start_delay: float = 0.0) -> np.ndarray:
    t_2d = wf_sub.t[np.newaxis, :]
    s2_mask = (t_2d >= start_times[:, np.newaxis] + t_start_delay) 
    # 4. Area Integration
    dt = wf_sub.t[1] - wf_sub.t[0]
    s2_areas = np.sum(wf_sub.v * s2_mask, axis=1) * dt
    return s2_areas

def _check_area_bounds(areas: np.ndarray, min_area: float, max_area: float) -> np.ndarray:
    """Returns 1D boolean mask of frames passing the area thresholds."""
    return (areas > min_area) & (areas < max_area)

def _update_s2_stats(stats: dict, pass_clip: np.ndarray, pass_left: np.ndarray, 
                     pass_right: np.ndarray, pass_area: np.ndarray, final_mask: np.ndarray) -> dict:
    """Accumulates the cut-flow statistics for the audit."""
    stats['pass_clip'] += int(pass_clip.sum())
    stats['pass_left'] += int(pass_left.sum())
    stats['pass_right'] += int(pass_right.sum())
    stats['pass_area'] += int(pass_area.sum())
    
    # Track how many times we relied on the static fallback window
    stats['fallback_right_bound'] += int((~pass_right & pass_left & pass_clip).sum())
    stats['accepted'] += int(final_mask.sum())
    return stats

def _print_audit_stats(set_pmt: SetPmt, set_stats: dict) -> None:
    total = set_stats['total']
    acc = set_stats['accepted']
    retention = (acc / total * 100) if total > 0 else 0.0
    print(f"  {set_pmt.source_dir.name} Audit: {acc}/{total} events ({retention:.1f}%)")
    print(f"    - Pass Clip:  {set_stats['pass_clip']}/{total}")
    print(f"    - Pass Left:  {set_stats['pass_left']}/{set_stats['pass_clip']}")
    print(f"    - Pass Right: {set_stats['pass_right']}/{set_stats['pass_left']} (Fallback used: {set_stats['fallback_right_bound']})")
    print(f"    - Pass Area:  {set_stats['pass_area']}/{total}")

# ===========================================================================
#  MATH ORCHESTRATION (Set-Level ETL)
# ===========================================================================
def find_s2(wf: Waveform, config: TimingConfig, t_start_s2: float) -> dict:
    """
    Vectorized engine for S2 detection. 
    Maintains strict array alignment and exports the exact boolean ledger.
    """
    stats = {
        'total': wf.nframes, 'pass_clip': 0, 'pass_left': 0, 'pass_right': 0, 
        'pass_v_min': 0, 'fallback_left': 0, 'fallback_right': 0, 'accepted': 0
    }
    wf_sub0 = subtract_pedestal(wf, n_points=config.n_pedestal)
    wf_clipped = threshold_clip(wf_sub0, threshold=config.bs_threshold)

    wf_smooth = moving_average(wf, window=config.s2_window_ma)
    wf_sub = subtract_pedestal(wf_smooth, n_points=config.n_pedestal)

    mask = wf_smooth.t > t_start_s2
    t_sliced = wf_smooth.t[mask]
    v_sliced = wf_sub.v[:, mask] if wf_sub.ff else wf_sub.v[mask][np.newaxis, :]

    if not np.any(mask) or v_sliced.shape[1] == 0:
        return {'s2_areas': np.array([]), 'uids': np.array([]), 'n_accepted': 0, 'stats': stats}

    # 2. Extract Global Peaks
    peak_idx = np.argmax(v_sliced, axis=1)
    peak_heights = v_sliced[np.arange(len(v_sliced)), peak_idx]
    peak_times = t_sliced[peak_idx]

    # 3. Apply Decoupled Vectorized Cuts
    pass_clip = _check_s2_clipping(wf.v, config.s2_threshold) 

    pass_v_min = peak_heights > config.s2_v_min
    start_times, pass_left = _find_left_boundary(
        v_sliced, t_sliced, peak_idx, peak_heights, peak_times, 
        config.s2_fraction_left, config.s2_fallback_window_left
    )
    
    end_times, pass_right = _find_right_boundary(
        v_sliced, t_sliced, peak_idx, peak_heights, peak_times, 
        config.s2_fraction_right, config.s2_fallback_window
    )

    # 4. Area Integration
    dt = wf_clipped.t[1] - wf_clipped.t[0]
    t_2d = wf_clipped.t[np.newaxis, :]
    s2_mask = (t_2d >= start_times[:, np.newaxis]) & (t_2d <= end_times[:, np.newaxis])
    s2_areas = np.sum(wf_clipped.v * s2_mask, axis=1) * dt

    pass_area = _check_area_bounds(s2_areas, config.s2_min_area, config.s2_max_area)

    # 5. Compile Final Acceptance
    final_mask = pass_clip & pass_v_min & pass_area

    # 6. Extract Diagnostic N-1 UIDs
    strict_v_min_failure = (~pass_v_min)
    n_minus_one_uids = wf.uids[strict_v_min_failure] if np.sum(strict_v_min_failure) > 0 else np.array([])

    # 7. Update Stats Ledger
    stats['pass_clip'] += int(pass_clip.sum())
    stats['pass_left'] += int(pass_left.sum())
    stats['pass_right'] += int(pass_right.sum())
    stats['pass_v_min'] += int(pass_v_min.sum())
    stats['fallback_left'] += int((~pass_left & pass_clip).sum())
    stats['fallback_right'] += int((~pass_right & pass_clip).sum())
    stats['accepted'] += int(final_mask.sum())

    # 8. Return Strictly Aligned Output
    return {
        's2_areas': np.atleast_1d(s2_areas[final_mask]) if np.sum(final_mask) > 0 else np.array([]),
        'raw_areas': np.atleast_1d(s2_areas),
        'peak_times': np.atleast_1d(peak_times[final_mask]) if np.sum(final_mask) > 0 else np.array([]),
        'start_times': np.atleast_1d(start_times[final_mask]) if np.sum(final_mask) > 0 else np.array([]),
        'end_times': np.atleast_1d(end_times[final_mask]) if np.sum(final_mask) > 0 else np.array([]),
        'uids': np.atleast_1d(wf.uids[final_mask]) if np.sum(final_mask) > 0 else np.array([]),
        # 'n_minus_one_uids': np.atleast_1d(n_minus_one_uids),
        'n_accepted': int(np.sum(final_mask)),
        'stats': stats,
    }

# ============================================================================
# 2. SET-LEVEL ETL (Cached Solver)
# ============================================================================

@allow_force
@load_cached_metadata(target_attr='n_areas_recoil')
@load_cached_npz(signal_type='s2_areas')
@require_attributes('time_drift')
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
    # accum_n_minus_one = []
    # accum_areas, accum_uids = [], []

    set_stats = {
        'total': 0, 'pass_clip': 0, 'pass_left': 0, 'pass_right': 0, 
        'pass_area': 0, 'fallback_right_bound': 0, 'accepted': 0
    }
    t_s1 = -1.5
    t_start_s2 = t_s1 + set_pmt.time_drift * config.s2_margin
    for wf in iter_waveforms(set_pmt, max_files=max_files, show_progress=True):
        result_dict = find_s2(wf, config=config, t_start_s2=t_start_s2)
        total_frames += wf.nframes
        accepted_frames += result_dict['uids'].size

        for k in set_stats:
            set_stats[k] += result_dict['stats'].get(k, 0)
        
        if result_dict['n_accepted'] > 0:
            accum_areas.append(result_dict['s2_areas'])
            accum_uids.append(result_dict['uids'])
            accum_starts.append(result_dict['start_times'])
            accum_ends.append(result_dict['end_times'])
            accum_peaks.append(result_dict['peak_times'])
            # accum_n_minus_one.append(result_dict['n_minus_one_uids'])

    # _print_audit_stats(set_pmt, set_stats)

    stats = {}
    area_arrays = {}
    # timing_arrays = {}
    if len(result_dict['s2_areas']) > 0:
        start_concat = np.concatenate(accum_starts)
        end_concat = np.concatenate(accum_ends)
        uids_concat = np.concatenate(accum_uids)

        area_arrays = {
            "s2_areas": np.concatenate(accum_areas),
            "uids": uids_concat,
            "stats": set_stats
        }

        timing_arrays = {
            "uids": uids_concat,
            "t_s2_start": start_concat,
            "t_s2_end": end_concat,
            "t_s2_peak":  np.concatenate(accum_peaks),
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