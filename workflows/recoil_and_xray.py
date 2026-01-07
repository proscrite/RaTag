"""
Unified X-ray and S2 recoil integration workflow.

This workflow processes all frames in each set, performing both X-ray event classification and S2 area integration in a single pass.
- Mirrors the structure of the previous xray_only and recoil_only workflows.
- Designed for efficiency: only one loop over all waveforms per set.
- Results are stored in the same format as the individual workflows (S2Areas, .npz files, set metadata).

Usage:
    from RaTag.workflows.recoil_and_xray import integrate_unified_in_run
    run = integrate_unified_in_run(run, ...)
"""

from typing import Optional
from pathlib import Path
from dataclasses import replace
import numpy as np

from RaTag.core.datatypes import SetPmt, Run, S2Areas
from RaTag.core.config import IntegrationConfig, XRayConfig
from RaTag.core.dataIO import iter_frameproxies, store_s2area, save_set_metadata
from RaTag.core.functional import apply_workflow_to_run, compute_max_files
from RaTag.core.uid_utils import make_uid
from RaTag.waveform.integration import integrate_s2_in_frame
from RaTag.waveform.xray_classification import classify_xray_in_frame
from RaTag.workflows.xray_integration import _RejectionTracker


def _setup_output_directories(set_pmt: SetPmt) -> tuple[Path, Path, Path]:
    """
    Ensure plot and data directories exist for the given set.
    Returns (plot_dir, data_dir) as Path objects.
    """
    plot_dir_recoil = Path(set_pmt.source_dir) / "plots" / "all" /"s2_areas"
    plot_dir_xray = Path(set_pmt.source_dir) / "plots" / "all" / "xrays"
    data_dir = Path(set_pmt.source_dir) / "processed_data"
    plot_dir_recoil.mkdir(parents=True, exist_ok=True)
    plot_dir_xray.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir_recoil, plot_dir_xray, data_dir


def _get_time_windows(set_pmt: SetPmt) -> tuple[float, float, float]:
    """

    """
    # Check that t_s1 is available
    if 't_s1' not in set_pmt.metadata:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing t_s1 - run timing estimation first")
    
    # Extract and ensure proper float type (metadata might store as string)
    t_s1 = float(set_pmt.metadata['t_s1'])
    
    # Determine S2 start: prefer t_s2_start, fallback to t_s1 + time_drift
    if 't_s2_start' in set_pmt.metadata or 't_s2_end' not in set_pmt.metadata:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing S2 window metadata")
    
    s2_start = float(set_pmt.metadata['t_s2_start'])
    s2_end = float(set_pmt.metadata['t_s2_end'])
    
    return t_s1, s2_start, s2_end

def _integrate_unified_in_set(set_pmt: SetPmt,
                              t_s1: float, s2_start: float, s2_end: float,
                              max_frames: Optional[int],
                              xray_config: XRayConfig = XRayConfig(),
                              integration_config: IntegrationConfig = IntegrationConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Complete unified X-ray and S2 integration workflow for a single set.
    1. For each frame: classify X-ray, integrate S2.
    2. Save S2 and X-ray areas to disk.
    3. Update set metadata.
    """
    
    accepted_xray_areas, accepted_xray_uids, s2_areas, s2_uids = [], [], [], []
    tracker = _RejectionTracker()

    max_files, actual_frames = compute_max_files(max_frames, set_pmt.nframes)
    if max_frames is not None:
        print(f"  Processing {max_files} files (~{actual_frames} frames)")

    for frame_wf in iter_frameproxies(set_pmt, max_files=max_files):    # type: ignore[arg-type]
        try:
            uid = make_uid(frame_wf.file_seq, frame_wf.frame_idx)
            frame_pmt = frame_wf.load_pmt_frame() # type: ignore[arg-type]
            frame_pmt = subtract_pedestal(frame_pmt, n_points=xray_config.n_pedestal)  # type: ignore[arg-type]

            is_xray, rejection_reason, xray_area = classify_xray_in_frame(frame_pmt, t_s1, s2_start, xray_config)
            
            if is_xray:
                 accepted_xray_areas.append(xray_area)
                 accepted_xray_uids.append(uid)
            
            tracker.record(is_xray, rejection_reason)

            s2_area = integrate_s2_in_frame(frame_pmt, s2_start, s2_end, integration_config)
            s2_areas.append(s2_area)
            s2_uids.append(uid)
        except Exception as e:
            print(f"  Warning: Frame processing failed: {e}"); continue
    
    if len(accepted_xray_uids) == 0:
        print(f"    ⚠ No frames accepted (all {tracker.n_total} rejected)")
    
    tracker.print_summary()
    
    accepted_xray_uids = np.array(accepted_xray_uids)
    accepted_xray_areas = np.array(accepted_xray_areas)
    s2_uids = np.array(s2_uids)
    s2_areas = np.array(s2_areas)
    return accepted_xray_areas, accepted_xray_uids, s2_areas, s2_uids, tracker.get_stats()


def workflow_unified_integration(set_pmt: SetPmt,
                                max_frames: Optional[int] = None,
                                integration_config: IntegrationConfig = IntegrationConfig(),
                                xray_config: XRayConfig = XRayConfig()) -> SetPmt:
    """
    Complete unified X-ray and S2 integration workflow for a single set.
    1. For each frame: classify X-ray, integrate S2.
    2. Save S2 and X-ray areas to disk.
    3. Update set metadata.
    """
    # Setup directories if not provided
    plot_recoil_dir, plot_xray_dir, data_dir = _setup_output_directories(set_pmt)
    
    # Extract time windows
    t_s1, s2_start, s2_end = _get_time_windows(set_pmt)
    
    print(f"\nProcessing set: {set_pmt.source_dir}")
    # Integrate X-ray and S2 areas
    accepted_xray_areas, accepted_xray_uids, s2_areas, s2_uid, rejection_stats = _integrate_unified_in_set(set_pmt, t_s1, s2_start, s2_end, max_frames, xray_config, integration_config)


    s2_obj = S2Areas(source_dir=set_pmt.source_dir,
                      areas=s2_areas, uids=s2_uids, method="unified_integration", 
                      params={"integration_config": integration_config, "xray_config": xray_config})
    store_s2area(s2_obj, set_pmt=set_pmt, suffix="s2_areas")

    xray_obj = S2Areas(source_dir=set_pmt.source_dir, 
                       areas=accepted_xray_areas, uids=accepted_xray_uids, method="xray_classification",
                       params={"xray_config": xray_config})
    store_s2area(xray_obj, set_pmt=set_pmt, suffix="xray_areas")
    
    new_metadata = {**set_pmt.metadata,
                    **rejection_stats,
                    'xray_area_mean': float(np.mean(accepted_xray_areas)) if len(accepted_xray_areas) > 0 else np.nan,
                    'xray_area_std': float(np.std(accepted_xray_areas)) if len(accepted_xray_areas) > 0 else np.nan,
                    }
    updated_set = replace(set_pmt, metadata=new_metadata)
    save_set_metadata(updated_set)
    return updated_set


def integrate_unified_in_run(run: Run,
                            range_sets: Optional[slice] = None,
                            max_frames: Optional[int] = None,
                            integration_config: IntegrationConfig = IntegrationConfig(),
                            xray_config: XRayConfig = XRayConfig()) -> Run:
    """
    Integrate X-ray and S2 areas for all sets (unified workflow).
    Uses apply_workflow_to_run for set-level orchestration and caching.
    """
    
    # Allow processing a subset of sets via range_sets (consistent with other run-level wrappers)
    if range_sets is not None:
        filtered_run = replace(run, sets=run.sets[range_sets])
    else:
        filtered_run = run

    return apply_workflow_to_run(filtered_run,
                                 workflow_func=workflow_unified_integration,
                                 workflow_name="Unified X-ray + S2 integration",
                                 cache_key="area_s2_mean",
                                 data_file_suffix="s2_areas.npz",
                                 max_frames=max_frames,
                                 integration_config=integration_config,
                                 xray_config=xray_config)
