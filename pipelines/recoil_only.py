"""
Complete ion recoil analysis pipeline.

This pipeline executes the full workflow using functional composition:
1. S2 area integration, fitting, and storage (set-level workflows)
2. Field-dependent summary plot (run-level)

Structure mirrors run_preparation.py for consistency.
All stages are explicit and composable - no hidden flags or conditionals.
"""

from functools import partial
from typing import Optional
from dataclasses import replace

from RaTag.core.datatypes import Run
from RaTag.core.config import IntegrationConfig, FitConfig
from RaTag.core.functional import pipe_run
from RaTag.workflows.recoil_integration import integrate_s2_in_run, summarize_s2_vs_field


# ============================================================================
# FUNCTIONAL PIPELINE COMPOSITION
# ============================================================================

def recoil_pipeline(run: Run,
                    range_sets: slice = None,
                    max_files: Optional[int] = None,
                    integration_config: IntegrationConfig = IntegrationConfig(),
                    fit_config: FitConfig = FitConfig()) -> Run:
    """
    Execute complete ion recoil S2 analysis pipeline.
    
    Prerequisites:
    - Run must be prepared (see run_preparation.prepare_run)
    - Sets must have S2 window metadata (t_s2_start, t_s2_end)
    
    Pipeline stages:
    1. integrate_s2_in_run: Complete ETL for all sets
       - Integration → Fitting → Storage → Histogram plots
       - Uses caching for efficiency
    2. plot_s2_vs_field: Summary plot of S2 vs drift field
    
    Each stage is explicit and can be customized/reordered.
    
    Args:
        run: Prepared Run object
        range_sets: Optional slice to process subset of sets
        max_files: Optional limit on files per set (testing)
        integration_config: S2 integration parameters
        fit_config: Gaussian fitting parameters
        
    Returns:
        Run with _s2_results attribute containing fitted S2Areas
        
    Example:
        >>> from RaTag.pipelines import prepare_run, recoil_pipeline
        >>> run = prepare_run(my_run)
        >>> run = recoil_pipeline(run, max_files=100)
    """
    print("\n" + "="*60)
    print(f"ION RECOIL ANALYSIS PIPELINE: {run.run_id}")
    print("="*60)
    
    # Build pipeline stages
    steps = [
        # Stage 1: Complete set-level ETL (integration + fitting + storage + plots)
        partial(integrate_s2_in_run,
                range_sets=range_sets,
                max_files=max_files,
                integration_config=integration_config,
                fit_config=fit_config),
        
        # Stage 2: Run-level summary plot
        summarize_s2_vs_field
    ]
    
    # Execute pipeline
    result = pipe_run(run, *steps)
    
    print("\n" + "="*60)
    print("RECOIL ANALYSIS COMPLETE")
    print("="*60)
    
    return result


# ============================================================================
# EXAMPLE: Custom Pipeline Variations
# ============================================================================

def recoil_pipeline_quick(run: Run, max_files: int = 50) -> Run:
    """
    Quick test version - limited files, skip field plot.
    
    Useful for testing integration settings on a small sample.
    """
    return integrate_s2_in_run(run, max_files=max_files)


def recoil_pipeline_replot(run: Run) -> Run:
    """
    Regenerate summary plots from cached S2 integration results.
    
    Useful when you want to adjust plotting parameters without
    recomputing integrations.
    
    Prerequisites:
    - Sets must have area_s2_mean in metadata (from previous integration)
    
    Args:
        run: Run with sets containing cached S2 results in metadata
        
    Returns:
        Run (unchanged, just regenerates plots)
    """
    
    print("\n" + "="*60)
    print("REPLOT FROM CACHE")
    print("="*60)
    
    # Check that we have cached results
    sets_with_results = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    
    if len(sets_with_results) == 0:
        print("  ⚠ No cached results found - run integration first")
        return run
    
    print(f"  Found {len(sets_with_results)} sets with cached results")
    
    # Just regenerate the summary plot
    run = summarize_s2_vs_field(run)
    
    return run