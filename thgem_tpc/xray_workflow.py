import numpy as np
import sys
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

def _check_min_v(v: np.ndarray, min_v: float) -> np.ndarray:
    """SPE grass cut: Returns a 1D boolean mask of frames passing the minimum charge threshold."""
    return np.max(v, axis=1) > min_v

def _update_stats(stats: dict, pass_clip: np.ndarray, pass_v_min_el: np.ndarray, pass_v_min_drift: np.ndarray, mask_xray: np.ndarray, mask_recoil: np.ndarray) -> dict:
    """Explicitly returns the updated dictionary to prevent NoneType state corruption."""
    stats['pass_clip'] += int(pass_clip.sum())
    stats['pass_v_min_el'] += int(pass_v_min_el.sum())
    stats['pass_v_min_drift'] += int(pass_v_min_drift.sum())
    stats['accepted_xray'] += int(mask_xray.sum())
    stats['accepted_recoil'] += int(mask_recoil.sum())
    return stats

def _print_xray_stats(stats: dict):
    """Updates batch statistics on a single console line."""
    msg = (f"\r    ✓ Batch: {stats['accepted_xray']}/{stats['total']} xray | "
           f"{stats['accepted_recoil']}/{stats['total']} recoil | "
           f"Rejects: {stats['total'] - stats['pass_clip']} clip, "
           f"{stats['total'] - stats['pass_v_min_el']} EL min V, "
           f"{stats['total'] - stats['pass_v_min_drift']} Drift min V")
    
    sys.stdout.write(msg)
    sys.stdout.flush()

def _aggregate_run_stats(run: Run, json_stats: list):
    """Cleanly merges the independent set-level JSON files into a run-level summary."""
    agg_stats = {'total': 0, 'pass_clip': 0, 'pass_v_min_el': 0, 'pass_v_min_drift': 0, 'accepted_xray': 0, 'accepted_recoil': 0}
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
# 2. PURE MATH & PHYSICS (Vectorized Bifurcated Integration on Batches)
# ============================================================================
def calculate_bifurcated_areas(set_pmt: SetPmt, 
                               t_s1: float, t_split: float, 
                               start_file: int, max_files: int,
                               config: XRayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Single-pass 2D vectorized integration mapping both the X-Ray and Recoil phase spaces."""
    
    out_uids_xray, out_xray, out_uids_recoil, out_recoil = [], [], [], []
    stats = {'total': 0, 'pass_clip': 0, 'pass_v_min_el': 0, 'pass_v_min_drift': 0, 'accepted_xray': 0, 'accepted_recoil': 0}

    for wf in file_ops.iter_waveforms(set_pmt, start_file=start_file, max_files=max_files, show_progress=False):
        
        # Guard clause against empty frames/corrupted PyVISA reads
        if wf.v is None or not len(wf.uids):
            continue
            
        wf = standard_preprocessing(wf, n_pedestal=int(config.n_pedestal),
                                    ma_window=int(config.ma_window),
                                    threshold=float(config.bs_threshold))

        v_2d = wf.v if wf.ff else wf.v[np.newaxis, :]
        
        # 1. Extract Geometric Partitions
        mask_xray = _get_window_mask(wf.t, t_s1, t_split)    # X-Ray window
        mask_recoil = _get_window_mask(wf.t, t_split, wf.t[-1])  # Recoil window

        v_xray = v_2d[:, mask_xray]
        v_recoil = v_2d[:, mask_recoil]

        # 2. Apply Hardware/Physics Cuts
        pass_clip = _check_clipping(v_2d, config.max_v_clip)  # Check whole trace for alpha saturation
        pass_v_min_el = _check_min_v(v_recoil, config.min_v_s2)     # Ensure a recoil actually exists
        pass_v_min_drift = _check_min_v(v_xray, config.min_v_xray)  # Ensure a minimum drift signal (X-ray) actually exists

        areas_xray = _integrate_window(v_xray, config.dt)   # Integrate X-Ray window
        areas_recoil = _integrate_window(v_recoil, config.dt)       # Integrate Recoil window

        # 3. Decoupled Topological Masks
        mask_xray = pass_clip & pass_v_min_drift
        mask_recoil = pass_clip & pass_v_min_el
        
        out_uids_xray.append(wf.uids[mask_xray])
        out_xray.append(areas_xray[mask_xray])

        out_uids_recoil.append(wf.uids[mask_recoil])
        out_recoil.append(areas_recoil[mask_recoil])

        # 4. Log Chunk Stats
        stats['total'] += len(wf.uids)
        stats = _update_stats(stats, pass_clip, pass_v_min_el, pass_v_min_drift, mask_xray, mask_recoil)
        
    _print_xray_stats(stats, )

    final_uids_xray = np.concatenate(out_uids_xray) if out_uids_xray else np.array([])
    final_xray = np.concatenate(out_xray) if out_xray else np.array([])
    final_uids_recoil = np.concatenate(out_uids_recoil) if out_uids_recoil else np.array([])
    final_recoil = np.concatenate(out_recoil) if out_recoil else np.array([])

    return final_xray, final_uids_xray, final_recoil, final_uids_recoil, stats

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
    Orchestrates the bifurcated extraction across millions of frames, 
    saving intermediate chunks to prevent RAM/I-O failure.
    """
    
    # Require explicit physics boundaries from config
    t_s1 = config.t_s1 
    t_split = config.t_s2_start

    total_files = len(set_pmt.filenames) if max_files is None else min(len(set_pmt.filenames), max_files)
    
    print(f"  Mapping {total_files} files in geometric partition [{t_s1} - {t_split} | {t_split} - [END] µs")

    # Dedicated checkpoint directories
    out_root = get_output_root(set_pmt.source_dir.parent)
    xray_chk_dir = out_root / "xray_checkpoints"
    recoil_chk_dir = out_root / "recoil_checkpoints"
    xray_chk_dir.mkdir(parents=True, exist_ok=True)
    recoil_chk_dir.mkdir(parents=True, exist_ok=True)

    all_xray, all_uids_xray = [], []
    all_recoil, all_uids_recoil = [], []
    agg_stats = {'total': 0, 'pass_clip': 0, 'pass_v_min_el': 0, 'pass_v_min_drift': 0, 'accepted_xray': 0, 'accepted_recoil': 0}

    for start_idx in range(0, total_files, config.batch_size):
        chunk_size = min(config.batch_size, total_files - start_idx)
        print(f"    -> Processing batch {start_idx} to {start_idx + chunk_size}...")

        xrays, uids_xray, recoils, uids_recoil, batch_stats = calculate_bifurcated_areas(set_pmt, t_s1, t_split,
                                                                                         start_file=start_idx, max_files=chunk_size, config=config)

        batch_xray_data = {"s2_areas": xrays, "uids": uids_xray, "stats": np.array([json.dumps(batch_stats)])}
        batch_recoil_data = {"s2_areas": recoils, "uids": uids_recoil}
        
        np.savez_compressed(xray_chk_dir / f"{set_pmt.source_dir.name}_xray_batch_{start_idx}.npz", **batch_xray_data)
        np.savez_compressed(recoil_chk_dir / f"{set_pmt.source_dir.name}_recoil_batch_{start_idx}.npz", **batch_recoil_data)

        # 2. RAM Accumulation
        all_xray.append(xrays)
        all_uids_xray.append(uids_xray)
        
        all_recoil.append(recoils)
        all_uids_recoil.append(uids_recoil)

        for k in agg_stats:
            agg_stats[k] += batch_stats.get(k, 0)

        
    print()

    ## 3. Final Reduction
    final_xray = np.concatenate(all_xray) if all_xray else np.array([])
    final_uids_xray = np.concatenate(all_uids_xray) if all_uids_xray else np.array([])
    
    final_recoil = np.concatenate(all_recoil) if all_recoil else np.array([])
    final_uids_recoil = np.concatenate(all_uids_recoil) if all_uids_recoil else np.array([])

    print(f"  ✓ Set {set_pmt.source_dir.name} Audit: {len(final_uids_xray)} X-Rays | {len(final_uids_recoil)} Recoils / {agg_stats['total']} total")

    # 4. Save independent Recoil baseline NPZ directly using standard API
    file_ops.save_npz_arrays(set_pmt, 's2_areas', {"s2_areas": final_recoil, "uids": final_uids_recoil})

    # 5. Return X-Ray NPZ to standard decorator for pipeline continuity
    arrays = {
        "s2_areas": final_xray,
        "uids": final_uids_xray,
        "stats": np.array([json.dumps(agg_stats)])
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