from RaTag.el_tpc.xray_workflow import (
    map_xray_events,
    fit_xray_events,
    make_xray_plots,
    calculate_xray_calibration,
)
from RaTag.core.datatypes import Run
from RaTag.core.config import XRayConfig

def pipeline_xray_calibration(run: Run, config: dict = None) -> Run:
    """
    High-level orchestration of the X-ray Calibration pipeline.
    
    Executes the 4-stage pipeline:
    1. Event Extraction & Aggregation
    2. Statistical Fitting
    3. QA Plot Generation
    4. g_S2 Physics Calibration
    """
    if config is None: 
        config = {}
        
    # Safely extract parameters
    xray_config = config.get('xray_config', XRayConfig())
    force = config.get('force', False)
    max_frames = config.get('max_frames', None)
    
    # 1. Extraction & Aggregation (Creates _combined.npz)
    run = map_xray_events(run, max_frames=max_frames, config=xray_config, force=force)
    
    # 2. Pure Math (Saves fit JSON)
    run = fit_xray_events(run)
    
    # 3. Presentation (Saves plots)
    run, figs = make_xray_plots(run, force=force)
    
    # 4. Physical Interpretation (Saves g_S2 calibration JSON)
    run = calculate_xray_calibration(run)
    
    return run