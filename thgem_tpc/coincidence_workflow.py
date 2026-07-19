import numpy as np
from pathlib import Path
from dataclasses import replace
from sklearn.mixture import GaussianMixture
import pandas as pd

from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.decorators import *
from RaTag.core.config import TimingConfig, FitConfig
from RaTag.core.paths import get_output_root
from RaTag.core.functional import map_over
from RaTag.io.file_ops import iter_waveforms
from RaTag.waveform.preprocessing import subtract_pedestal


from RaTag.thgem_tpc.timing_workflow import compute_timing_statistics, find_s1, find_s2


@allow_force
@load_cached_metadata(target_attr='n_areas_recoil')
@load_cached_npz(signal_type='s2_areas')
@require_attributes('time_drift')
@write_metadata(target_attr='n_areas_recoil')
@write_npz_arrays(signal_type='s2_areas')
@limit_frames
def resolve_set_coincidence(set_pmt: SetPmt, max_files: int = None,
                             config: TimingConfig = TimingConfig(),
                             force: bool = False) -> tuple[SetPmt, dict]:
    """
    Worker function: Finds S1 and S2, enforces strict coincidence, integrates the Area, 
    and saves ALL timing and area data to a dedicated sandbox namespace.
    """
    
    all_s1, all_s2_starts, all_s2_ends = [], [], []
    all_s1_areas, all_s2_areas, uids_out = [], [], []
    total_frames = 0
    coinc_frames = 0

    for wf in iter_waveforms(set_pmt, max_files=max_files, show_progress=True):
        wf_sub = subtract_pedestal(wf, config.n_pedestal)
        total_frames += wf.nframes
        
        # 1. S1 Search (Safely handle your modified return signature)
        # s1_out = find_s1(wf, config)
        # s1_times = s1_out[0] if isinstance(s1_out, tuple) else s1_out
        # has_s1 = ~np.isnan(s1_times)
        # if not np.any(has_s1):
        #     continue

        # # On the fly S1 integration (for valid frames)
        dt = wf_sub.t[1] - wf_sub.t[0]
        t_2d = wf_sub.t[np.newaxis, :]
        # s1_times_2d = s1_times[:, np.newaxis]
        # local_mask_s1 = (t_2d >= s1_times_2d - 0.05) & (t_2d <= s1_times_2d + 0.05)
        # v_full = wf_sub.v if wf_sub.ff else wf_sub.v[np.newaxis, :]
        # s1_areas = np.sum(v_full * local_mask_s1, axis=1) * dt
        

            
        # 2. S2 Search
        t_min_s2 = -2.5 
        s2_out = find_s2(wf, config, t_min_s2)
        s2_starts, s2_ends = s2_out[0], s2_out[1]
        has_s2 = ~np.isnan(s2_starts) & ~np.isnan(s2_ends)
        
        # 3. Coincidence Veto
        # valid_mask = has_s1 & has_s2
        valid_mask = has_s2
        if not np.any(valid_mask):
            continue
            
        coinc_frames += np.sum(valid_mask)
        
        # 4. On-the-fly S2 Integration (Only for valid frames)

        starts_2d = s2_starts[:, np.newaxis]
        ends_2d = s2_ends[:, np.newaxis]
        
        int_mask = (t_2d >= starts_2d) & (t_2d <= ends_2d)
        v_full = wf_sub.v if wf_sub.ff else wf_sub.v[np.newaxis, :]
        
        # Apply boolean mask to slice out only the valid coincidence events
        v_valid = v_full[valid_mask]
        mask_valid = int_mask[valid_mask]
        areas = np.sum(v_valid * mask_valid, axis=1) * dt
        
        # 5. Accumulate
        # all_s1.append(s1_times[valid_mask])
        all_s2_starts.append(s2_starts[valid_mask])
        all_s2_ends.append(s2_ends[valid_mask])
        # all_s1_areas.append(s1_areas[valid_mask])
        all_s2_areas.append(areas)
        uids_out.append(wf.uids[valid_mask])

    area_arrays = {
        "uids": np.concatenate(uids_out) if uids_out else np.array([]),
        "s2_areas": np.concatenate(all_s2_areas) if all_s2_areas else np.array([]),
        "s1_areas": np.concatenate(all_s1_areas) if all_s1_areas else np.array([])
    }
    timing_arrays = {}
    timing_arrays["uids"] = np.concatenate(uids_out) if uids_out else np.array([])
    stats = {}
    for name, buffer in [("t_s1", all_s1),
                         ("t_s2_start", all_s2_starts),
                        ("t_s2_end", all_s2_ends)]: 
        arr_concat = np.concatenate(buffer) if buffer else np.array([])
        timing_arrays[name] = arr_concat

        stats.update(compute_timing_statistics(arr_concat, name=name))
        
    retention = float((coinc_frames / total_frames * 100) if total_frames > 0 else 0.0)
    print(f"  {set_pmt.source_dir.name}: {coinc_frames}/{total_frames} events ({retention:.1f}%)")
    
    stats['n_areas_recoil'] = int(coinc_frames)
    file_ops.save_npz_arrays(set_pmt, 'timing', timing_arrays) # This is done manually to avoid overwriting the S2 areas npz file
    
    return replace(set_pmt, **stats), area_arrays
    

def map_coincidence_extraction(run: Run, max_frames: int = None,
                                config: TimingConfig = TimingConfig(),
                                force: bool = False) -> Run:
    print("\n" + "="*60 + f"\nEXTRACTING COINCIDENCE DATA: {run.run_id}\n" + "="*60)
    bound_timing = lambda s: resolve_set_coincidence(s, max_frames=max_frames, 
                                                    config=config, force=force)
    
    new_sets = map_over(run.sets, bound_timing, catch_errors=True)
    return replace(run, sets=new_sets)


@allow_force
@load_cached_metadata(target_attr='filtered_drift_time')
@write_metadata(target_attr='filtered_drift_time')
@write_npz_arrays(signal_type='s2_areas')
def resolve_drift_time_filter(set_pmt: SetPmt, 
                              dt_tolerance: float = 0.6, 
                              force: bool = False) -> tuple[SetPmt, dict]:
    """
    Worker function: Loads timing and area arrays, calculates the drift time per frame,
    finds the main physical blob (1D mode), and filters out misclassified outliers.
    Rewrites both .npz files with the purified data.
    """
    # 1. Load arrays directly from disk 
    # (We bypass the @load_cached decorator because we explicitly WANT to read, filter, and overwrite)
    timing_arrays = file_ops.load_npz_arrays(set_pmt, 'timing')
    area_arrays = file_ops.load_npz_arrays(set_pmt, 's2_areas')
    
    if not timing_arrays or not area_arrays:
        print(f"  [Skipping] {set_pmt.source_dir.name}: Missing npz files.")
        return set_pmt, None
        
    if 'uids' not in timing_arrays or 'uids' not in area_arrays:
        print(f"  [Skipping] {set_pmt.source_dir.name}: Missing UIDs for alignment.")
        return set_pmt, None

    # 2. Safely merge and align all arrays
    df_time = pd.DataFrame(timing_arrays).drop_duplicates(subset=['uids'])
    df_area = pd.DataFrame(area_arrays).drop_duplicates(subset=['uids'])
    
    df_merged = pd.merge(df_time, df_area, on='uids', how='inner')
    
    if df_merged.empty:
        return set_pmt, None

    # 3. 1D Histogram Peak Finding
    df_merged['drift_time'] = df_merged['t_s2_start'] - df_merged['t_s1']
    
    # Filter purely unphysical times before searching for the peak
    valid_dt = df_merged[df_merged['drift_time'] > 0]['drift_time']
    
    if len(valid_dt) == 0:
        return set_pmt, None
        
    # Find the central drift time (the Mode) via a 1D histogram
    counts, bins = np.histogram(valid_dt, bins=100)
    mode_idx = np.argmax(counts)
    dt_mode = 0.5 * (bins[mode_idx] + bins[mode_idx + 1])
    
    # Define the acceptable physical window around the blob
    lower_cut = dt_mode - dt_tolerance
    upper_cut = dt_mode + dt_tolerance
    
    # 4. Apply the Cut & Repackage
    mask = (df_merged['drift_time'] >= lower_cut) & (df_merged['drift_time'] <= upper_cut)
    df_filtered = df_merged[mask]
    
    retained = len(df_filtered)
    total = len(df_merged)
    pct = (retained / total) * 100 if total > 0 else 0
    print(f"  {set_pmt.source_dir.name}: Mode = {dt_mode:.2f} µs, range [{lower_cut:.2f}, {upper_cut:.2f}] | Kept {retained}/{total} events ({pct:.1f}%)")
    
    # Reconstruct timing_arrays (grabbing only the keys that originally belonged to timing)
    new_timing_arrays = {k: df_filtered[k].values for k in timing_arrays.keys()}
    
    # Reconstruct area_arrays (grabbing only the keys that originally belonged to areas)
    new_area_arrays = {k: df_filtered[k].values for k in area_arrays.keys()}
    
    file_ops.save_npz_arrays(set_pmt, 'timing', new_timing_arrays)
    
    updated_set = replace(set_pmt, filtered_drift_time=True, n_areas_recoil=retained)
    
    return updated_set, new_area_arrays


def map_filter_drift_time(run: Run, dt_tolerance: float = 0.6, force: bool = False) -> Run:
    """Entry point: Maps the drift time filter across all sets in the Run."""
    print("\n" + "="*60)
    print(f"FILTERING BY DRIFT TIME (Blob Isolation): {run.run_id}")
    print("="*60)
    
    bound_filter = lambda s: resolve_drift_time_filter(s, dt_tolerance=dt_tolerance, force=force)
    
    new_sets = map_over(run.sets, bound_filter, catch_errors=True)
    return replace(run, sets=new_sets)


# ----------------------------------------------------------------
# ---- TODO??: Add a post-processing step to apply the Bayesian GMM filter to the coincidence-filtered S2 areas. This will help isolate the high-purity Hole events for downstream analysis.
# ----------------------------------------------------------------

def apply_bayesian_hole_filter(run: Run, fallback_cut=0.06):
    """
    Iterates through the coincidence-filtered s2_areas .npz files.
    Applies the GMM to find the optimal S1 cut, filters for Hole events,
    and overwrites the arrays so the downstream fitting tools see only the high-purity data.
    """
    print("\n" + "="*60)
    print(f"APPLYING BAYESIAN GMM HOLE FILTER: {run.run_id}")
    print("="*60)
    
    # We need to know where the sandbox stored the files
    base_path = get_output_root(run.root_directory) / 's2_areas'
    
    total_retained = 0
    total_initial = 0
    
    for set_pmt in run.sets:
        npz_path = base_path / f"{set_pmt.source_dir.name}_s2_areas.npz"
        if not npz_path.exists():
            print(f"  [Skipping] {set_pmt.source_dir.name}: No npz file found.")
            continue
            
        arrays = np.load(npz_path)
        s1_areas = arrays.get('s1_areas', np.array([]))
        s2_areas = arrays.get('s2_areas', np.array([]))
        uids = arrays.get('uids', np.array([]))
        
        if len(s1_areas) < 50:
            print(f"  [Skipping] {set_pmt.source_dir.name}: Insufficient statistics.")
            continue
            
        total_initial += len(s1_areas)
        
        # 1. Fit the GMM and find the cut
        clean_s1 = s1_areas[(s1_areas > 0.005) & (s1_areas < 0.15)]
        X = clean_s1.reshape(-1, 1)
        
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(X)
            
            means = gmm.means_.flatten()
            order = np.argsort(means)
            web_idx, hole_idx = order[0], order[1]
            
            x_smooth = np.linspace(0.005, 0.15, 1000).reshape(-1, 1)
            probs = gmm.predict_proba(x_smooth)
            
            crossings = np.where(probs[:, hole_idx] > probs[:, web_idx])[0]
            optimal_cut = x_smooth[crossings[0]][0] if len(crossings) > 0 else fallback_cut
        except Exception as e:
            print(f"  [Warning] GMM failed for {set_pmt.source_dir.name}. Using fallback. ({e})")
            optimal_cut = fallback_cut

        # 2. Apply the Cut (Select the Hole Tail)
        mask = s1_areas > optimal_cut
        
        filtered_s1 = s1_areas[mask]
        filtered_s2 = s2_areas[mask]
        filtered_uids = uids[mask]
        
        retained = len(filtered_s2)
        total_retained += retained
        pct = (retained / len(s1_areas)) * 100
        print(f"  {set_pmt.source_dir.name}: Cut > {optimal_cut:.3f} | Retained {retained}/{len(s1_areas)} ({pct:.1f}%)")
        
        # 3. Overwrite the main npz file so the @load_cached_npz decorators pick it up
        # We save it as the standard 's2_areas' suffix that your fit decorators expect
        out_path = base_path / f"{set_pmt.source_dir.name}_s2_areas.npz"
        np.savez(out_path, 
                 s1_areas=filtered_s1, 
                 s2_areas=filtered_s2, 
                 uids=filtered_uids)

    print(f"\nFinal Hole Population: {total_retained} events across the run.")