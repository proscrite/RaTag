"""
Unified X-ray and S2 recoil analysis pipeline.

This pipeline executes the unified workflow using functional composition:
1. Unified X-ray classification and S2 area integration (set-level workflows)
2. (Future) Quality assurance plots
3. (Future) Multi-isotope distribution analysis

Structure mirrors xray_only.py and recoil_only.py for consistency.
All stages are explicit and composable - no hidden flags or conditionals.
"""

from functools import partial
from typing import Optional
from RaTag.core.datatypes import Run
from RaTag.core.config import IntegrationConfig, XRayConfig, FitConfig
from RaTag.core.functional import pipe_run

from RaTag.workflows.recoil_and_xray import integrate_unified_in_run
from RaTag.pipelines.recoil_only import recoil_downstream_pipeline
from RaTag.pipelines.xray_only import xray_downstream_pipeline

# =========================================================================
# FUNCTIONAL PIPELINE COMPOSITION
# =========================================================================

def unified_pipeline(run: Run,
                    range_sets: Optional[slice] = None,
                    max_frames: Optional[int] = None,
                    integration_config: IntegrationConfig = IntegrationConfig(),
                    xray_config: XRayConfig = XRayConfig(),
                    fit_config: FitConfig = FitConfig(),
                    isotope_ranges: Optional[dict] = None,
                    n_validation_frames: int = 5):
    """
    Execute complete unified X-ray + S2 analysis pipeline.
    
    Prerequisites:
    - Run must be prepared (see run_preparation.prepare_run)
    - Sets must have S2 window metadata (t_s2_start, t_s2_end)
    - Sets must have t_s1 in metadata
    
    Pipeline stages:
    1. integrate_unified_in_run: Complete ETL for all sets
       - X-ray classification + S2 integration + storage
       - Uses caching for efficiency
    2. (Future) QA validation plots
    3. (Future) Multi-isotope distribution analysis
    
    Args:
        run: Prepared Run object
        range_sets: Optional slice to process subset of sets
        max_frames: Optional limit on frames per set (testing)
        integration_config: S2 integration parameters
        xray_config: X-ray classification parameters
        fit_config: Gaussian fitting parameters
    
    Returns:
        Run with S2 and X-ray statistics in set metadata
    """
    print("\n" + "="*60)
    print(f"UNIFIED X-RAY + S2 ANALYSIS PIPELINE: {run.run_id}")
    print("="*60)
    
    # Step 1: Unified integration (X-ray + S2 in one pass)
    run = integrate_unified_in_run(run, 
                                   range_sets=range_sets,
                                   max_frames=max_frames,
                                   integration_config=integration_config,
                                   xray_config=xray_config)

    # Step 2: Downstream S2 (recoil) operations
    run = recoil_downstream_pipeline(run,
                                     fit_config=fit_config,
                                     isotope_ranges=isotope_ranges)

    # Step 3: Downstream X-ray operations
    run = xray_downstream_pipeline(run,
                                   xray_config=xray_config,
                                   isotope_ranges=isotope_ranges,
                                   n_validation_frames=n_validation_frames)

    print("\n" + "="*60)
    print("UNIFIED ANALYSIS COMPLETE")
    print("="*60)
    return run
