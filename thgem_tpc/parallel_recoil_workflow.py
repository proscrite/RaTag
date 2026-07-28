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
from RaTag.thgem_tpc.recoil_workflow import find_s2

# ============================================================================
# 2. SET-LEVEL ETL (Chunked Map-Reduce)
# ============================================================================
import concurrent.futures
from RaTag.core.dataIO import load_wfm

def integrate_s2_chunk(source_dir, file_chunk: list[str], start_seq: int, config: TimingConfig) -> dict:
    """
    Pure worker function for the Map-Reduce pipeline.
    Loads waveforms directly from disk to avoid IPC overhead and memory swapping.
    """
    accum_areas, accum_uids, accum_starts, accum_ends, accum_peaks = [], [], [], [], []
    accepted_frames = 0
    total_frames = 0

    for i, fn in enumerate(file_chunk):
        # 1. Load data within the worker
        wf = load_wfm(source_dir / fn)
        wf.file_seq = start_seq + i  # Enforce global sequence for UID integrity

        # 2. Execute vectorized physics logic
        res = find_s2(wf, config=config)

        # 3. Local reduction
        total_frames += wf.nframes
        accepted_frames += res['n_accepted']

        if res['n_accepted'] > 0:
            accum_areas.append(res['s2_areas'])
            accum_uids.append(res['uids'])
            accum_starts.append(res['start_times'])
            accum_ends.append(res['end_times'])
            accum_peaks.append(res['peak_times'])

    return {
        'total_frames': total_frames,
        'accepted_frames': accepted_frames,
        's2_areas': np.concatenate(accum_areas) if accum_areas else np.array([]),
        'uids': np.concatenate(accum_uids) if accum_uids else np.array([]),
        'start_times': np.concatenate(accum_starts) if accum_starts else np.array([]),
        'end_times': np.concatenate(accum_ends) if accum_ends else np.array([]),
        'peak_times': np.concatenate(accum_peaks) if accum_peaks else np.array([])
    }


@allow_force
@load_cached_metadata(target_attr='n_areas_recoil')
@load_cached_npz(signal_type='s2_areas')
@write_metadata(target_attr='n_areas_recoil')
@write_npz_arrays(signal_type='s2_areas')
@limit_frames
def resolve_set_recoils(set_pmt: SetPmt, 
                        max_files: Optional[int] = None, 
                        config: TimingConfig = TimingConfig(),
                        chunk_size: int = 50) -> tuple[SetPmt, dict]:
    """
    Executes parallel S2 detection using a chunked ProcessPool.
    Replaces the serial iter_waveforms generator for high-throughput scaling.
    """
    files_to_process = set_pmt.filenames[:max_files] if max_files else set_pmt.filenames
    total_files = len(files_to_process)

    if total_files == 0:
        return set_pmt, {}

    # 1. Prepare Chunks (Map Setup)
    chunks = []
    for i in range(0, total_files, chunk_size):
        chunks.append((
            set_pmt.source_dir,
            files_to_process[i:i + chunk_size],
            i,  
            config
        ))

    # 2. Parallel Execution
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(integrate_s2_chunk, *chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 3. Global Reduction
    total_frames = sum(r['total_frames'] for r in results)
    accepted_frames = sum(r['accepted_frames'] for r in results)

    retention = float((accepted_frames / total_frames * 100) if total_frames > 0 else 0.0)
    print(f"  {set_pmt.source_dir.name}: {accepted_frames}/{total_frames} events ({retention:.1f}%)")
    
    if accepted_frames == 0:
        return set_pmt, {}

    uids_concat = np.concatenate([r['uids'] for r in results if r['uids'].size > 0])
    areas_concat = np.concatenate([r['s2_areas'] for r in results if r['s2_areas'].size > 0])
    start_concat = np.concatenate([r['start_times'] for r in results if r['start_times'].size > 0])
    end_concat = np.concatenate([r['end_times'] for r in results if r['end_times'].size > 0])
    peaks_concat = np.concatenate([r['peak_times'] for r in results if r['peak_times'].size > 0])

    # 4. Strict Temporal Sorting
    sort_idx = np.argsort(uids_concat)
    uids_concat = uids_concat[sort_idx]
    areas_concat = areas_concat[sort_idx]
    start_concat = start_concat[sort_idx]
    end_concat = end_concat[sort_idx]
    peaks_concat = peaks_concat[sort_idx]

    area_arrays = {
        "s2_areas": areas_concat,
        "uids": uids_concat
    }

    timing_arrays = {
        "uids": uids_concat,
        "t_s2_start": start_concat,
        "t_s2_end": end_concat,
        "t_s2_peak": peaks_concat
    }

    stats = {}
    stats.update(compute_timing_statistics(start_concat, name='t_s2_start'))
    stats.update(compute_timing_statistics(end_concat, name='t_s2_end'))
    stats['n_areas_recoil'] = len(uids_concat)

    # Update Set Metadata
    file_ops.save_npz_arrays(set_pmt, 'timing', timing_arrays)
    updated_set = replace(set_pmt, **stats)
    
    return updated_set, area_arrays