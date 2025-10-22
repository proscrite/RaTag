"""
Unified integration module for simultaneous X-ray classification and S2 area integration.

Version 2: Optimized to avoid redundant preprocessing operations.

This module combines two previously separate workflows into a single pass over all waveforms:
1. X-ray event classification (for calibration)
2. Ion S2 area integration (for recombination analysis)

Key optimization: Moving average and threshold clipping are performed ONCE per waveform,
then different windows are extracted for X-ray vs S2 analysis.
"""

from typing import Dict, Tuple, Optional
from dataclasses import replace
from pathlib import Path
import numpy as np

from .datatypes import PMTWaveform, SetPmt, S2Areas, XRayEvent, XRayResults, Run
from .dataIO import iter_waveforms, store_s2area, store_xrayset
from .transformations import (
    t_in_us, v_in_mV, subtract_pedestal, extract_window, 
    moving_average, threshold_clip, integrate_trapz
)
from .config import IntegrationConfig
from .units import s_to_us


def process_waveform_unified(
    wf: PMTWaveform,
    t_s1: float,
    s2_start: float,
    s2_end: float,
    xray_config: dict,
    s2_config: dict
) -> Tuple[list[XRayEvent], np.ndarray]:
    """
    Process a single waveform for both X-ray classification and S2 integration.
    
    OPTIMIZATION: Preprocessing (moving average, threshold clip) done ONCE,
    then different windows extracted for each analysis.
    
    Args:
        wf: Raw waveform
        t_s1: S1 peak time [µs]
        s2_start: S2 window start [µs]
        s2_end: S2 window end [µs]
        xray_config: Configuration for X-ray classification
        s2_config: Configuration for S2 integration
        
    Returns:
        Tuple of (xray_events, s2_areas_array)
        - xray_events: List of XRayEvent objects (one per frame if FastFrame)
        - s2_areas_array: NumPy array of S2 areas (one per frame if FastFrame)
    """
    wf_id = Path(wf.source).stem
    xray_events = []
    s2_areas = []
    
    # Convert to standard units (ONCE per waveform)
    wf = t_in_us(wf)
    wf = v_in_mV(wf)
    wf = subtract_pedestal(wf, n_points=xray_config['n_pedestal'])
    
    # Handle FastFrame vs single frame
    frames = enumerate(wf.v) if wf.ff else [(0, wf.v)]
    
    for frame_idx, v_frame in frames:
        # Create single-frame view
        frame_wf = replace(wf, v=v_frame, ff=False)
        
        # ============================================
        # SHARED PREPROCESSING (ONCE per frame)
        # ============================================
        # Apply moving average and threshold clipping to FULL waveform
        processed_wf = moving_average(frame_wf, window=xray_config['ma_window'])
        processed_wf = threshold_clip(processed_wf, threshold=xray_config['bs_threshold'])
        
        # ============================================
        # 1. X-RAY CLASSIFICATION
        # ============================================
        xray_event = classify_xray_frame(
            processed_wf, wf_id, frame_idx, t_s1, s2_start, xray_config
        )
        xray_events.append(xray_event)
        
        # ============================================
        # 2. S2 AREA INTEGRATION
        # ============================================
        s2_area = integrate_s2_frame(
            processed_wf, s2_start, s2_end, s2_config
        )
        s2_areas.append(s2_area)
    
    return xray_events, np.array(s2_areas)


def classify_xray_frame(
    processed_wf: PMTWaveform,
    wf_id: str,
    frame_idx: int,
    t_s1: float,
    s2_start: float,
    config: dict
) -> XRayEvent:
    """
    Classify a single frame as X-ray or not.
    
    Args:
        processed_wf: Waveform that has ALREADY been processed with moving average 
                      and threshold clipping
        wf_id: Waveform identifier
        frame_idx: Frame index (for FastFrame)
        t_s1: S1 peak time [µs]
        s2_start: S2 window start [µs]
        config: X-ray configuration parameters
        
    Returns:
        XRayEvent with acceptance decision and area (if accepted)
    """
    frame_id = f"{wf_id}_ff{frame_idx}" if frame_idx > 0 else wf_id
    
    # Check 1: Excessive S2 signal
    mask = (processed_wf.t >= s2_start)
    Vs2 = processed_wf.v[mask]
    if Vs2[Vs2 > config['bs_threshold']].sum() > config['max_area_s2']:
        return XRayEvent(
            wf_id=frame_id,
            accepted=False,
            reason="excessive S2 signal above baseline"
        )
    
    # Extract drift window for X-ray integration (already processed!)
    xray_wf = extract_window(processed_wf, t_s1, s2_start)
    
    # Check 2: Sufficient separation
    t_thresh = xray_wf.t[xray_wf.v > 0] if np.any(xray_wf.v > 0) else None
    if t_thresh is None or len(t_thresh) == 0:
        return XRayEvent(
            wf_id=frame_id,
            accepted=False,
            reason="no signal above baseline"
        )
    
    t_pre_s2 = s2_start - t_thresh[-1]
    t_post_s1 = t_thresh[0] - t_s1
    
    if not (t_pre_s2 > config['min_s2_sep'] and t_post_s1 > config['min_s1_sep']):
        return XRayEvent(
            wf_id=frame_id,
            accepted=False,
            reason="insufficient separation"  # Simplified - no timing details
        )
    
    # Integrate X-ray signal (no further preprocessing needed!)
    integrator = config.get('integrator', integrate_trapz)
    area = float(integrator(xray_wf, dt=config['dt']))
    
    return XRayEvent(
        wf_id=frame_id,
        accepted=True,
        reason="ok",
        area=area
    )


def integrate_s2_frame(
    processed_wf: PMTWaveform,
    s2_start: float,
    s2_end: float,
    config: dict
) -> float:
    """
    Integrate S2 signal for a single frame.
    
    Args:
        processed_wf: Waveform that has ALREADY been processed with moving average 
                      and threshold clipping
        s2_start: S2 window start [µs]
        s2_end: S2 window end [µs]
        config: S2 integration configuration
        
    Returns:
        S2 area [mV·µs]
    """
    # Extract S2 window (already processed!)
    s2_wf = extract_window(processed_wf, s2_start, s2_end)
    
    # Integrate (no further preprocessing needed!)
    integrator = config.get('integrator', integrate_trapz)
    area = float(integrator(s2_wf, dt=config['dt']))
    
    return area


def integrate_set_unified(set_pmt: SetPmt,
                        t_s1: float,
                        s2_start: float,
                        s2_end: float,
                        xray_config: dict,
                        s2_config: dict
                                        ) -> Tuple[XRayResults, S2Areas]:
    """
    Process all waveforms in a set simultaneously for X-ray classification and S2 integration.
    
    This is the set-level function that replaces:
    - classify_xrays_set() from xray_integration.py
    - integrate_set_s2() from analysis.py
    
    Args:
        set_pmt: Measurement set
        t_s1: S1 peak time [µs]
        s2_start: S2 window start [µs]
        s2_end: S2 window end [µs]
        xray_config: X-ray classification parameters
        s2_config: S2 integration parameters
        
    Returns:
        Tuple of (XRayResults, S2Areas)
    """
    all_xray_events = []
    all_s2_areas = []
    
    for idx, wf in enumerate(iter_waveforms(set_pmt)):
        try:
            xray_events, s2_areas = process_waveform_unified(
                wf, t_s1, s2_start, s2_end, xray_config, s2_config
            )
            all_xray_events.extend(xray_events)
            all_s2_areas.extend(s2_areas)
        except Exception as e:
            # Handle failed waveforms gracefully
            wf_id = Path(wf.source).stem
            all_xray_events.append(XRayEvent(
                wf_id=wf_id,
                accepted=False,
                reason=f"processing_error: {str(e)}"
            ))
            all_s2_areas.append(np.nan)
    
    # Package results
    # Note: Remove 'integrator' function from configs before storing (not JSON serializable)
    xray_params_serializable = {k: v for k, v in xray_config.items() if k != 'integrator'}
    s2_params_serializable = {k: v for k, v in s2_config.items() if k != 'integrator'}
    
    xray_results = XRayResults(
        set_id=set_pmt.source_dir,
        events=all_xray_events,
        params={
            't_s1': t_s1,
            's2_start': s2_start,
            **xray_params_serializable
        }
    )
    
    s2_areas_obj = S2Areas(
        source_dir=set_pmt.source_dir,
        areas=np.array(all_s2_areas),
        method="unified_integration",
        params={
            't_window': (s2_start, s2_end),
            **s2_params_serializable,
            'set_metadata': {
                't_s1': set_pmt.metadata.get('t_s1'),
                't_s1_std': set_pmt.metadata.get('t_s1_std'),
                't_s2_start_mean': set_pmt.metadata.get('t_s2_start_mean'),
                't_s2_start_std': set_pmt.metadata.get('t_s2_start_std'),
                't_s2_end_mean': set_pmt.metadata.get('t_s2_end_mean'),
                't_s2_end_std': set_pmt.metadata.get('t_s2_end_std'),
                's2_duration_mean': set_pmt.metadata.get('s2_duration_mean'),
                's2_duration_std': set_pmt.metadata.get('s2_duration_std'),
            }
        }
    )
    
    return xray_results, s2_areas_obj


def integrate_run_unified(
    run: Run,
    ts2_tol: float = -2.7,
    range_sets: Optional[slice] = None,
    xray_config: Optional[IntegrationConfig] = None,
    ion_config: Optional[IntegrationConfig] = None,
    use_estimated_s2_windows: bool = True
) -> Tuple[Dict[str, XRayResults], Dict[str, S2Areas]]:
    """
    Process all sets in a run with unified X-ray classification and S2 integration.
    
    This is the run-level function that replaces:
    - run_xray_classification() from pipeline.py
    - run_ion_integration() from pipeline.py (integration part only, not fitting)
    
    Args:
        run: Run object with prepared sets
        ts2_tol: Time tolerance before S2 window (µs), used if not using estimated windows
        range_sets: Optional slice to process subset of sets
        xray_config: X-ray classification configuration
        ion_config: Ion S2 integration configuration
        use_estimated_s2_windows: If True, use estimated S2 windows from metadata
        
    Returns:
        Tuple of (xray_results_dict, s2_areas_dict)
        - xray_results_dict: {set_name: XRayResults}
        - s2_areas_dict: {set_name: S2Areas}
    """
    # Default configurations
    if xray_config is None:
        xray_config = IntegrationConfig()
    if ion_config is None:
        ion_config = IntegrationConfig()
    
    # Convert to parameter dictionaries
    xray_params = {
        'bs_threshold': xray_config.bs_threshold,
        'max_area_s2': 1e5,  # X-ray specific
        'min_s2_sep': 1.0,   # X-ray specific
        'min_s1_sep': 0.5,   # X-ray specific
        'n_pedestal': xray_config.n_pedestal,
        'ma_window': xray_config.ma_window,
        'dt': xray_config.dt,
        'integrator': integrate_trapz
    }
    
    s2_params = {
        'bs_threshold': ion_config.bs_threshold,
        'n_pedestal': ion_config.n_pedestal,
        'ma_window': ion_config.ma_window,
        'dt': ion_config.dt,
        'integrator': integrate_trapz
    }
    
    # Select sets to process
    sets_to_process = run.sets[range_sets] if range_sets is not None else run.sets
    
    xray_results_dict = {}
    s2_areas_dict = {}
    
    print("=" * 60)
    print("UNIFIED INTEGRATION: X-RAY CLASSIFICATION + S2 AREAS")
    print("=" * 60)
    print(f"Processing {len(sets_to_process)} sets with SINGLE pass over waveforms")
    print()
    
    for i, set_pmt in enumerate(sets_to_process, 1):
        set_name = set_pmt.source_dir.name
        print(f"[{i}/{len(sets_to_process)}] Processing: {set_name}")
        
        # Validate preconditions
        if "t_s1" not in set_pmt.metadata or set_pmt.time_drift is None:
            raise ValueError(f"Set {set_name} missing t_s1 or time_drift. Run prepare_run() first.")
        
        t_s1 = set_pmt.metadata['t_s1']
        dt_s1 = set_pmt.metadata.get('t_s1_std', 0.0)
        print(f"  → Using tS1: [{t_s1:.2f} ± {dt_s1:.2f}] µs")
        # Determine S2 integration window
        if use_estimated_s2_windows and 't_s2_start_mean' in set_pmt.metadata:
            s2_start = set_pmt.metadata['t_s2_start_mean']
            s2_end = set_pmt.metadata['t_s2_end_mean']
            print(f"  → Using estimated S2 window: [{s2_start:.2f}, {s2_end:.2f}] µs")
        else:
            s2_start = t_s1 + set_pmt.time_drift + ts2_tol
            s2_end = s2_start + 50.0  # Default 50 µs window
            print(f"  → Using offset-based S2 window: [{s2_start:.2f}, {s2_end:.2f}] µs")
        
        # Process set (SINGLE PASS)
        xray_results, s2_areas = integrate_set_unified(
            set_pmt, t_s1, s2_start, s2_end, xray_params, s2_params
        )
        
        # Store results (both .npy and .json files)
        store_xrayset(xray_results)  # Saves xray_areas.npy + xray_results.json
        store_s2area(s2_areas, set_pmt=set_pmt)  # Saves s2_areas.npy + s2_results.json
        
        # Count accepted X-rays
        n_accepted = sum(1 for e in xray_results.events if e.accepted)
        n_total = len(xray_results.events)
        print(f"  → X-rays: {n_accepted}/{n_total} accepted ({100*n_accepted/n_total:.1f}%)")
        print(f"  → S2 areas: {len(s2_areas.areas)} integrated")
        print(f"  → Saved: xray_results.json, xray_areas.npy, s2_areas.npy, s2_results.json")
        print()
        
        xray_results_dict[set_name] = xray_results
        s2_areas_dict[set_name] = s2_areas
    
    print("=" * 60)
    print("UNIFIED INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"✓ Processed {len(sets_to_process)} sets in SINGLE pass")
    print(f"✓ X-ray results saved for calibration workflow")
    print(f"✓ S2 areas saved for fitting workflow")
    print()
    
    return xray_results_dict, s2_areas_dict
