# Import libraries
from pathlib import Path

from RaTag.core.config import TimingConfig, IntegrationConfig, FitConfig

from RaTag.io.bootstrap import bootstrap_from_config
from RaTag.io.file_ops import load_yaml
from RaTag.thgem_tpc.drift_workflow import map_drift_physics
from RaTag.thgem_tpc.timing_workflow import map_time_windows, map_timing_plots
from RaTag.thgem_tpc.recoil_workflow import map_recoil_integration, map_recoil_fits, map_recoil_plots
from RaTag.core.datatypes import Run


def pipeline_recoil_analysis(run: Run, config: dict = None) -> Run:
    """
    High-level orchestration of the Recoil Analysis pipeline.
    
    Executes the 5-stage pipeline:
    1. Drift Physics Mapping
    2. Timing Window Calculation and QA Plot Generation
    3. Recoil S2 Integration
    4. Recoil S2 Fitting & Plotting
    """
    if config is None: 
        config = {}

    timing_config = config.get('preparation', TimingConfig())
    integ_config = config.get('integration', IntegrationConfig())
    fit_config = config.get('fit_config', FitConfig())

    timing_config = TimingConfig(**{k: v for k, v in timing_config.items() if hasattr(TimingConfig, k)})
    integ_config = IntegrationConfig(**{k: v for k, v in integ_config.items() if hasattr(IntegrationConfig, k)})
    fit_config = FitConfig(**{k: v for k, v in fit_config.items() if hasattr(FitConfig, k)})
    
    
    # 1. Drift Physics Mapping (Saves drift physics JSON)
    run = map_drift_physics(run, force=False)

    # 2. Timing Window Calculation & QA Plots (Saves timing JSON & plots)
    # run = map_time_windows(run, max_frames=timing_config.max_frames_s1, config=TimingConfig(), force=force)

    # 3. Recoil S2 Integration (Saves {set_name}_s2_areas.npz)
    run = map_recoil_integration(run, max_frames=timing_config.max_frames,
                                  config=timing_config, force=timing_config.force)
    run = map_timing_plots(run, force=timing_config.force)

    # 4. Recoil S2 Fitting & Plotting (Saves fit JSON & plots)
    run = map_recoil_fits(run, config=fit_config, force=fit_config.force)
    run = map_recoil_plots(run, config=fit_config, force=fit_config.force)

    return run