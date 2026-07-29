from RaTag.thgem_tpc.xray_workflow import (
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
    if config is None: 
        config = {}
        
    # Safely extract parameters
    xray_config_dict = config.get('xray_config', XRayConfig())
    xray_config = XRayConfig(**{k: v for k, v in xray_config_dict.items() if hasattr(XRayConfig, k)})

    fit_config_dict = config.get('fit_config', FitConfig())
    fit_config = FitConfig(**{k: v for k, v in fit_config_dict.items() if hasattr(FitConfig, k)})

    # 1. Extraction & Aggregation (Creates _combined.npz)
    run = map_xray_events(run, max_frames=xray_config.max_frames, config=xray_config, force=xray_config.force)
    
    # 2. Pure Math (Saves fit JSON)
    run = fit_xray_events(run, config=fit_config)
    
    # 3. Presentation (Saves plots)
    run, figs = make_xray_plots(run, force=fit_config.force)
    
    # 4. Physical Interpretation (Saves g_S2 calibration JSON)
    run = calculate_xray_calibration(run)
    
    return run