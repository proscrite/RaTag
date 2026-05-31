from RaTag.core.datatypes import Run
from RaTag.io.bootstrap import bootstrap_from_config
from RaTag.el_tpc.drift_workflow import map_drift_physics
from RaTag.el_tpc.baseline_workflow import map_run_baseline
from RaTag.el_tpc.timing_pipeline import pipeline_timing_estimation

def pipeline_preprocessing(run: Run, config: dict) -> Run:
    """Preprocessing meta-Pipeline: Raw Dir -> Ready for Integration."""
    
    # 1. Bootstrapping (From your bootstrap.py)
    run = bootstrap_from_config(config)
    
    # 2. Drift Physics Layer
    run = map_drift_physics(run)
    run = map_run_baseline(run, max_frames=480, n_points=200)  # Baseline calculation added here for timing optimization
    
    # 3. S1/S2 Timing Layer
    run = pipeline_timing_estimation(run)
    
        
    return run  