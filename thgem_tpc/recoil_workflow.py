# RaTag/el_tpc/recoil_workflow.py
import numpy as np
from typing import Optional, Tuple
from dataclasses import replace

from RaTag.core.datatypes import Run, SetPmt, S2Areas
from RaTag.core.config import IntegrationConfig, FitConfig
from RaTag.core.paths import get_output_root
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.io.file_ops import iter_waveforms, load_s2areas, load_fit_result
from RaTag.waveform.preprocessing import standard_preprocessing
# from RaTag.core.fitting import fit_set_s2
from RaTag.plotting import (
    plot_s2areas_summary, plot_run_s2_vs_field, 
    catch_plot_errors, build_fig_grid
)
from RaTag.el_tpc.fit_s2_area import fit_s2_crystalball, v_crystalball_right
# ============================================================================
# 1. PURE MATH & PHYSICS (The Vectorized Engine)
# ============================================================================

def _compute_static_bounds(set_pmt: SetPmt, config: IntegrationConfig) -> tuple[float, float]:
    """Calculates the global static integration window for the set."""
    if set_pmt.t_s2_start is None or set_pmt.t_s2_end is None:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing S2 timing metadata. Run Timing Pipeline first.")
        
    t_start = float(set_pmt.t_s2_start)
    t_end = float(set_pmt.t_s2_end)
    std_start = float(set_pmt.t_s2_start_std or 0.0)
    std_end = float(set_pmt.t_s2_end_std or 0.0)
    
    static_start = t_start - (config.n_sigma_start * std_start)
    static_end = t_end + (config.n_sigma_end * std_end)
    
    return static_start, static_end

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


# ============================================================================
# 2. SET-LEVEL ETL (Cached Solver)
# ============================================================================

@allow_force
@load_cached_metadata(target_attr='n_areas_recoil')
@load_cached_npz(signal_type='s2_areas')
@require_attributes('t_s2_start', 't_s2_end')
@write_metadata(target_attr='n_areas_recoil')
@write_npz_arrays(signal_type='s2_areas')
@limit_frames
def resolve_set_recoils(set_pmt: SetPmt, 
                        max_files: Optional[int] = None, 
                        config: IntegrationConfig = IntegrationConfig()) -> tuple[SetPmt, dict]:
    """
    Executes S2 integration and formats the arrays for storage.
    """
    areas, uids = calculate_s2_areas(set_pmt, max_files=max_files, config=config)
    
    # Update Set Metadata
    updated_set = replace(set_pmt, n_areas_recoil=len(areas))
    
    arrays = {
        "s2_areas": areas,
        "uids": uids
    }
    
    return updated_set, arrays


# ============================================================================
# 3. RUN-LEVEL ORCHESTRATOR
# ============================================================================

def map_recoil_integration(run: Run, 
                    max_frames: Optional[int] = None, 
                    config: IntegrationConfig = IntegrationConfig(),
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
                                       nbins=config.nbins, smooth=config.smooth )
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
    
    updated_sets = map_over(run.sets, bound_fitter, catch_errors=True)
    
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