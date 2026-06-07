from RaTag.core.datatypes import Run
from .timing_workflow import map_time_windows, map_multiiso_timing, map_timing_plots

def pipeline_timing_estimation(run: Run, config: dict) -> Run:
    """
    High-level orchestration of the timing pipeline.
    
    This is the entry point for the timing domain. It sequences 
    the operations and branches based on the Run's isotope configuration.
    """
    
    # 1. Mainstream S1/S2 Estimation
    run = map_time_windows(run, **config.get('preparation', {}))
    
    # 2. Plotting and Validation
    run = map_timing_plots(run)

    # 3. Conditional Multi-Isotope Branching
    # Explicit logic is safer than hidden magic
    if run.sets[0].multiiso:
        run = map_multiiso_timing(run, **config.get('multiiso', {}))
    
    return run