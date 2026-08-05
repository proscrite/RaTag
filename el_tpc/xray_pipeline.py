from RaTag.el_tpc.xray_workflow import (
    map_xray_events,
    fit_xray_events,
    make_xray_plots,
    calculate_xray_calibration,
)
from RaTag.core.datatypes import Run
from RaTag.core.config import FitConfig, XRayConfig

def pipeline_xray_calibration(run: Run, config: dict = None) -> Run:
    """
    High-level orchestration of the X-ray Calibration pipeline.
    
    Executes the 4-stage pipeline:
    1. Event Extraction & Aggregation
    2. Statistical Fitting
    3. QA Plot Generation
    4. g_S2 Physics Calibration
    """
    exec_cfg = config.get('execution', {})
        
    # Safely extract parameters
    xray_config = config.get('xray_config', XRayConfig())
    xray_state = exec_cfg.get('run_xrays', False)
    force_xrays = (xray_state == 'force')

    fit_config = config.get('fit_config', FitConfig())
    fit_state = exec_cfg.get('run_fit', False)
    force_fit = (fit_state == 'force')

    # 1. Extraction & Aggregation (Creates _combined.npz)
    run = map_xray_events(run, max_frames=xray_config.max_frames, config=xray_config, force=force_xrays)
    
    # 2. Pure Math (Saves fit JSON)
    run = fit_xray_events(run)
    
    # 3. Presentation (Saves plots)
    run, figs = make_xray_plots(run, force=force_fit)
    
    # 4. Physical Interpretation (Saves g_S2 calibration JSON)
    run = calculate_xray_calibration(run)
    
    return run