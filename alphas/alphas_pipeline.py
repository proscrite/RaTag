# RaTag/alphas/alpha_pipeline.py
from typing import Optional
from RaTag.core.datatypes import Run
from RaTag.core.config import AlphaCalibrationConfig, FitConfig


from RaTag.alphas.alphas_workflow import map_alpha_events

from RaTag.alphas.alphas_calibration import (
    map_alpha_calibrations,
    map_alpha_plots
)

def pipeline_alpha_calibration(run: Run, config: dict = None) -> Run:
    """
    High-level orchestration of the Alpha Calibration pipeline.
    
    Executes the 3-stage pipeline:
    1. Energy Map Generation (Waveforms -> NPZ)
    2. Peak Fitting & Range Derivation (SCA -> MeV)
    3. QA Dashboard Generation
    """
    if config is None: 
        config = {}
        
    # Extract parameters from the YAML dictionary
    exec_cfg = config.get('execution', {})
    alpha_config = config.get('energy_mapping', AlphaCalibrationConfig())
    alpha_config = AlphaCalibrationConfig(**{k: v for k, v in alpha_config.items() if hasattr(AlphaCalibrationConfig, k)})
    
    max_frames = alpha_config.max_frames
    fit_config = config.get('fit_config', FitConfig())
    fit_config = FitConfig(**{k: v for k, v in fit_config.items() if hasattr(FitConfig, k)})

    alphas_state = exec_cfg.get('run_alphas', False)
    force_alphas = (alphas_state == 'force')

    fit_state = exec_cfg.get('run_fit', False)
    force_fit = (fit_state == 'force')

    # 1. Extraction
    run = map_alpha_events(run, max_frames=max_frames, config=alpha_config, force=force_alphas)
    
    # 2. Pure Math & Calibration (Saves flat metadata and fits.json)
<<<<<<< HEAD
    run = map_alpha_calibrations(run, config=alpha_config, force=fit_config.force)
=======
    run = map_alpha_calibrations(run, config=alpha_config, force=force_fit)
>>>>>>> 11cc7ccb7d488de26d22fa50d618806c45237683
    
    # 3. Presentation (Saves plots)
    run = map_alpha_plots(run, force=force_fit)
    
    return run