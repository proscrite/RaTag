import numpy as np
import random
import json
import matplotlib.pyplot as plt
from typing import Optional, Any
from dataclasses import replace

from RaTag.core.datatypes import Run, SetPmt, S2Areas
from RaTag.core.config import XRayConfig, FitConfig
from RaTag.core.paths import get_output_root
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.core.uid_utils import sample_validation_waveforms
from RaTag.io import file_ops
from RaTag.waveform.preprocessing import standard_preprocessing
from RaTag.plotting import plot_xray_candidate, plot_s2areas_summary, plot_xray_histogram, plot_xray_validation, catch_plot_errors
from RaTag.el_tpc.fit_s2_area import fit_s2_crystalball, v_crystalball_right, compute_fit_ci

# ============================================================================
# 1. VECTORIZED HELPERS (Private)
# ============================================================================
def _get_window_mask(t: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    """Returns a 1D boolean mask for the specified drift time window."""
    return (t >= t_start) & (t < t_end)

def _check_clipping(v_window: np.ndarray, max_v: float) -> np.ndarray:
    """Anti-alpha cut: Returns a 1D boolean mask of frames avoiding amplifier saturation."""
    return np.max(v_window, axis=1) < max_v

def _integrate_window(v_window: np.ndarray, dt: float) -> np.ndarray:
    """Calorimetric step: Returns a 1D array of integrated areas for the given window."""
    return np.trapezoid(v_window, dx=dt, axis=1)

def _check_min_area(areas: np.ndarray, min_area: float) -> np.ndarray:
    """SPE grass cut: Returns a 1D boolean mask of frames passing the minimum charge threshold."""
    return areas > min_area

def _update_stats(stats: dict, pass_clip: np.ndarray, pass_area: np.ndarray, final_mask: np.ndarray) -> dict:
    """Helper to update the stats dictionary with the results from a batch."""
    stats['pass_clip'] += int(pass_clip.sum())
    stats['pass_area'] += int(pass_area.sum())
    stats['accepted'] += int(final_mask.sum())
    return stats

def _print_xray_stats(stats: dict, n_acc: int):
    print(f"    ✓ Classified {stats['total']} frames: {stats['accepted']} accepted")
    print(f"      - Clipping reject:  {stats['total'] - stats['pass_clip']}")
    print(f"      - SPE/Grass reject: {stats['total'] - stats['pass_area']}")
    
def _aggregate_run_stats(run: Run, json_stats: list):
    """Cleanly merges the independent set-level JSON files into a run-level summary."""
    agg_stats = {'total': 0, 'pass_clip': 0, 'pass_area': 0, 'accepted': 0}
    out_root = get_output_root(run) / "xray_areas"
    
    # 1. Accumulate
    for stats_str in json_stats:
        set_stats = json.loads(stats_str)
        for key in agg_stats:
            agg_stats[key] += set_stats.get(key, 0)
            
    # 2. Save JSON
    out_root = get_output_root(run.root_directory) / "xray_areas"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / f"{run.run_id}_xray_stats.json", 'w') as f:
        json.dump(agg_stats, f, indent=4)

# ============================================================================
# 2. PURE MATH & PHYSICS (The Vectorized Engine)
# ============================================================================

def calculate_xray_areas(set_pmt: SetPmt, 
                         t_s1: float, t_s2_start: float,
                         max_files: Optional[int] = None, 
                         config: XRayConfig = XRayConfig()) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-pass 2D vectorized integration of total charge in the drift window."""
    
    print(f"  Searching X-Ray window: [{t_s1:.2f}, {t_s2_start:.2f}] µs")

    out_uids, out_areas = [], []
    stats = {'total': 0, 'pass_clip': 0, 'pass_area': 0, 'accepted': 0}

    for wf in file_ops.iter_waveforms(set_pmt, max_files=max_files, show_progress=True):
        
        # 1. Preprocess
        wf = standard_preprocessing(wf, n_pedestal=int(config.n_pedestal),
                                    ma_window=int(config.ma_window),
                                    threshold=float(config.bs_threshold))
        
        v_2d = wf.v if wf.ff else wf.v[np.newaxis, :]
        
        # 2. Extract Phase Space Window
        win_mask = _get_window_mask(wf.t, t_s1, t_s2_start)
        v_win = v_2d[:, win_mask]

        # 3. Apply Hardware/Physics Cuts
        pass_clip = _check_clipping(v_win, config.max_v_clip)
        areas = _integrate_window(v_win, config.dt)
        pass_area = _check_min_area(areas, config.min_xray_area)

        # 4. Filter and Store
        final_mask = pass_clip & pass_area
        out_uids.append(wf.uids[final_mask])
        out_areas.append(areas[final_mask])

        # 5. Log Chunk Stats
        stats['total'] += len(wf.uids)
        stats = _update_stats(stats, pass_clip=pass_clip, pass_area=pass_area, final_mask=final_mask)
        

    # Compile Final Arrays
    final_uids = np.concatenate(out_uids) if out_uids else np.array([])
    final_areas = np.concatenate(out_areas) if out_areas else np.array([])
    
    _print_xray_stats(stats, n_acc=len(final_uids))
    return final_areas, final_uids, stats
# ============================================================================
# 2. SET-LEVEL ETL (Cached Solver) and RUN-LEVEL AGGREGATOR
# ============================================================================
@allow_force
@load_cached_npz(signal_type='xray_areas')
@write_npz_arrays(signal_type='xray_areas')
@limit_frames
def resolve_set_xrays(set_pmt: SetPmt, 
                      max_files: Optional[int] = None, 
                      config: XRayConfig = XRayConfig()) -> tuple[SetPmt, dict]:
    """
    Executes X-ray classification and formats the arrays for storage.
    """
    t_s1 = XRayConfig().get('t_s1', set_pmt.t_s1)
    t_s2_start = XRayConfig().get('t_s2_start', set_pmt.t_s2_start)

    
    if t_s1 is None or t_s2_start is None:
        raise ValueError(f"Set {set_pmt.source_dir.name} is missing required attributes t_s1 or t_s2_start.")
        
    areas, uids, stats = calculate_xray_areas(set_pmt, t_s1=t_s1, t_s2_start=t_s2_start, max_files=max_files, config=config)
    
    # Dense dictionary naturally saved to {set_name}_xray_areas.npz
    arrays = {
        "s2_areas": areas,
        "uids": uids,
        "stats": np.array([json.dumps(stats)])  # Store stats as a JSON string in a 1-element array for npz compatibility
    }
    
    return set_pmt, arrays

@persist_run_results(signal_type='xray_areas')
def aggregate_run_xrays(run: Run) -> tuple[Run, dict]:
    """Pure reduction function. Extracts all set arrays and combines them."""
    print("  Aggregating run-level X-ray areas...")
    
    all_areas, all_uids, all_stats = [], [], []
    for set_pmt in run.sets:
        arrays = file_ops.load_npz_arrays(set_pmt, 'xray_areas')
        if arrays and 's2_areas' in arrays:
            all_areas.append(arrays['s2_areas'])
            all_uids.append(arrays['uids'])
            print(f"      Loaded {len(arrays['uids'])} X-ray events from set {set_pmt.source_dir.name}")
        
        if 'stats' in arrays:
            all_stats.append(arrays['stats'][0])  # Extract the JSON string from the array

    _aggregate_run_stats(run, all_stats)  # Also aggregate the stats into a run-level JSON file

    combined_arrays = {
        "s2_areas": np.concatenate(all_areas) if all_areas else np.array([]),
        "uids": np.concatenate(all_uids) if all_uids else np.array([])
    }
    
    return run, combined_arrays


# ============================================================================
# 3. RUN-LEVEL ORCHESTRATOR
# ============================================================================

def map_xray_events(run: Run, 
                  max_frames: Optional[int] = None, 
                  config: XRayConfig = XRayConfig(),
                  force: bool = False) -> Run:
    """Entry point: Extracts per-set X-Rays, then reduces them to a run array."""
    
    print("\n" + "="*60 + f"\nIDENTIFYING X-RAY EVENTS: {run.run_id}\n" + "="*60)

    # 1. Map (Extract)
    bound_xrays = lambda s: resolve_set_xrays(s, max_frames=max_frames, config=config, force=force)
    map_over(run.sets, bound_xrays, catch_errors=True)  # This doesn't update the sets in-place, but it populates the cache and data files for the next step
    
    # 2. Reduce (Aggregate)
    final_run = aggregate_run_xrays(run)
    
    return final_run


# ============================================================================
# 4. ANALYTICAL PHASE (Fitting the X-Ray Peak)
# ============================================================================

def fit_xray_events(run: Run, config: FitConfig = FitConfig()) -> Run:
    print("\n" + "="*60 + f"\nFITTING COMBINED X-RAY AREA: {run.run_id}\n" + "="*60)
    
    combined_file = get_output_root(run.root_directory) / "xray_areas" / f"{run.run_id}_xray_areas.npz"
    if not combined_file.exists(): 
        print(f"  ⚠ Combined X-ray areas not found for Run {run.run_id}. Run map_xrays_events first.")
        return run
        
    s2_combined = file_ops.load_s2areas_from_path(combined_file)
    if len(s2_combined.areas) < 10: 
        print(f"  ⚠ Not enough X-ray events for fitting in Run {run.run_id}.")
        return run

    try:
        result = fit_s2_crystalball(s2_combined.areas, bin_cuts=config.bin_cuts, nbins=config.nbins)

        fit_path = get_output_root(run.root_directory) / "fits" / f"{run.run_id}_xray_areas_hist_fit.json"
        file_ops.save_fit_result(result['result'], fit_path)
        print(f"  ✓ Fit: μ={result['peak_position']:.3f} ± {result['ci95']:.3f} mV·µs")

    except Exception as e: print(f"  ⚠ Fit failed: {e}")
        
    return run


# ============================================================================
# 5. VALIDATION PLOTS
# ============================================================================

def _load_xray_plot_data(run: Run) -> tuple[Optional[S2Areas], Optional[Any], Optional[float]]:
    """Helper to safely load both the combined areas and the lmfit model for presentation."""
    combined_file = get_output_root(run.root_directory) / "xray_areas" / f"{run.run_id}_xray_areas.npz"
    fit_path = get_output_root(run.root_directory) / "fits" / f"{run.run_id}_xray_areas_hist_fit.json"
    
    if not combined_file.exists():
        return None, None, None
        
    # 1. Load Data
    s2_combined = file_ops.load_s2areas_from_path(combined_file)
    print(f"  Loaded combined X-ray areas: {len(s2_combined.areas)} events")
    # 2. Load Fit
    fit_model = file_ops.load_fit_result(fit_path, funcdefs={'v_crystalball_right': v_crystalball_right})
    
    # Extract the mean directly from the restored lmfit model parameters
    fit_mean = fit_model.params['sig_x0'].value if fit_model else None
    
    return s2_combined, fit_model, fit_mean

@allow_force
@load_cached_plots(subfolder="xray_areas", expected_suffixes=["histogram", "validation"])
@write_plots(subfolder="xray_areas")
def make_xray_plots(run: Run, force: bool = False) -> tuple[Run, dict]:
    """Generates the combined X-ray area histogram with fit overlay and a validation dashboard of sample waveforms."""
    
    print("\n" + "="*60 + f"\nGENERATING X-RAY QA PLOTS: {run.run_id}\n" + "="*60)

    s2areas_combined, fit_model, fit_mean = _load_xray_plot_data(run)

    if not s2areas_combined or not len(s2areas_combined.areas):
        print("  ⚠ Combined X-ray areas not found. Skipping plots.")
        return run, {}

    figs = {}
    try:
        # 1. Combined Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        plot_s2areas_summary(ax_hist, f"Combined X-Rays - {run.run_id}", s2areas_combined, fit_model, fit_mean)
        figs["combined_histogram"] = fig_hist
        
        # 2. Set-level Validation Dashboards
        for set_pmt in run.sets:
            # Explicit Set-level Constants
            t_s1 = float(set_pmt.t_s1)
            s2_start = float(set_pmt.t_s2_start) if set_pmt.t_s2_start else t_s1 + float(set_pmt.time_drift)
            
            acc_wfs, rej_wfs = sample_validation_waveforms(set_pmt, s2areas_combined.uids, n_samples=4)
            # Pure Plotting
            fig_val = plot_xray_validation(acc_wfs, rej_wfs, t_s1, s2_start, 
                                           title=f"Validation - {set_pmt.source_dir.name}")
            
            # Assign to dictionary (handles multi-set runs safely)
            key = "validation" if len(run.sets) == 1 else f"validation_{set_pmt.source_dir.name}"
            figs[key] = fig_val

        return run, figs
        
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")
        return run, {}


def calculate_xray_calibration(run: Run) -> Run:
    """
    Physical domain phase.
    Converts the abstract statistical fit of the X-ray S2 areas into 
    the absolute physical g_S2 calibration factor and saves a readable JSON.
    """
    print("\n" + "="*60 + f"\nCALCULATING S2 CALIBRATION (g_S2): {run.run_id}\n" + "="*60)
    
    fit_path = get_output_root(run.root_directory) / "fits" / f"{run.run_id}_xray_areas_hist_fit.json"
    fit_model = file_ops.load_fit_result(fit_path, funcdefs={'v_crystalball_right': v_crystalball_right})
        
    # 1. Extract Statistical Parameters 
    # (Using the known bin width of 20/100 = 0.2 from the fit step)
    xrmean = fit_model.params['sig_x0'].value
    xrci95 = compute_fit_ci(fit_model.params['sig_x0'], bin_width=0.2)
    
    # 2. Apply Xenon Physics Constants
    E_XRAY_EV = 12300.0  # Th228 X-ray energy (eV)
    W_I_EV = 22.0        # Work function in 2 bar Xenon (eV/e-)
    
    N_e_gamma = E_XRAY_EV / W_I_EV
    
    gs2_factor = xrmean / N_e_gamma
    gs2_uncert = xrci95 / N_e_gamma
    
    # 3. Format the Declarative Physics JSON
    xray_res = {
        'Wi': W_I_EV,
        'units_Wi': 'eV/e-',
        'A_x_mean': float(xrmean),
        'units_A_x_mean': 'mV us',
        'dA_x_mean': float(xrci95),
        'units_dA_x_mean': 'mV us',
        'gs2': float(gs2_factor),
        'units_gs2': 'mV us / e-',
        'd_gs2': float(gs2_uncert),
        'units_d_gs2': 'mV us / e-'
    }
    
    # 4. Save to Disk
    out_file = get_output_root(run.root_directory) / "xray_areas" / f"{run.run_id}_xray_calibration.json"
    
    with open(out_file, 'w') as f:
        json.dump(xray_res, f, indent=4)
  
    print(f"  ✓ Calibration saved: g_S2 = {gs2_factor:.4f} ± {gs2_uncert:.4f} mV·µs/e-")    
    return run