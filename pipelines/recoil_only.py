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
from RaTag.workflows.recoil_integration import (integrate_s2_in_run,
                                                fit_s2_in_run,
                                                summarize_s2_vs_field,
                                                run_s2_area_multiiso)


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
        # Stage 1: Complete set-level ETL (integration + storage + plots)
        partial(integrate_s2_in_run,
                range_sets=range_sets,
                max_files=max_files,
                integration_config=integration_config),
        
        # Stage 2: Fit S2 area distributions in each set
        partial(fit_s2_in_run,
                fit_config=fit_config),
        
        # Stage 3: Run-level summary plot
        summarize_s2_vs_field
    ]
    
    # Execute pipeline
    result = pipe_run(run, *steps)
    
    print("\n" + "="*60)
    print("RECOIL ANALYSIS COMPLETE")
    print("="*60)
    
    return result



# ============================================================================
# MULTI-ISOTOPE PIPELINE
# ============================================================================

def recoil_pipeline_multiiso(run: Run,
                             isotope_ranges: dict,
                             range_sets: slice = None,
                             max_files: Optional[int] = None,
                             integration_config: IntegrationConfig = IntegrationConfig(),
                             fit_config: FitConfig = FitConfig()) -> Run:
    """
    Complete ion recoil S2 analysis pipeline (multi-isotope).
    
    Pipeline stages:
    1. Integrate S2 areas for all sets (aggregated)
    2. Map S2 areas to isotopes by energy
    3. Fit Gaussian to distributions (per isotope)
    4. Summarize S2 vs drift field
    """
    print("\n" + "="*60)
    print(f"MULTI-ISOTOPE RECOIL PIPELINE: {run.run_id}")
    print("="*60)
    
    steps = [
        # Standard integration (all events together)
        partial(integrate_s2_in_run,
                range_sets=range_sets,
                max_files=max_files,
                integration_config=integration_config),
        
        # Standard fitting (aggregated)
        partial(fit_s2_in_run,
                fit_config=fit_config),
        
        # Map to isotopes
        partial(run_s2_area_multiiso,
                isotope_ranges=isotope_ranges),
        
        # Summary plot
        summarize_s2_vs_field
    ]
    
    return pipe_run(run, *steps)

# ============================================================================
# EXAMPLE: Custom Pipeline Variations
# ============================================================================

def recoil_pipeline_quick(run: Run, max_files: int = 50) -> Run:
    """
    Quick test version - limited files, skip field plot.
    
    Useful for testing integration settings on a small sample.
    """
    return integrate_s2_in_run(run, max_files=max_files)


def recoil_pipeline_replot(run: Run,
                           fit_config: FitConfig = FitConfig()) -> Run:
    """
    Refit and regenerate plots from cached S2 integration results.
    
    Useful when you want to:
    - Refit with different bin_cuts or other FitConfig parameters
    - Regenerate plots after patching data units
    - Adjust plotting parameters without recomputing integrations
    
    Prerequisites:
    - Sets must have s2_areas.npz files (from previous integration)
    
    Args:
        run: Run with sets containing cached S2 area NPZ files
        fit_config: Gaussian fitting parameters
        
    Returns:
        Run with updated fit results in metadata
    """
    
    print("\n" + "="*60)
    print("REFIT & REPLOT FROM CACHE")
    print("="*60)
    
    # Check that we have cached NPZ files
    sets_with_data = []
    for s in run.sets:
        data_file = s.source_dir.parent / "processed_data" / "all" / f"{s.source_dir.name}_s2_areas.npz"
        if data_file.exists():
            sets_with_data.append(s)
    
    if len(sets_with_data) == 0:
        print("  ⚠ No cached s2_areas.npz files found - run integration first")
        return run
    
    print(f"  Found {len(sets_with_data)} sets with cached S2 areas")
    
    # Build refit pipeline
    steps = [
        # Refit S2 area distributions
        partial(fit_s2_in_run, fit_config=fit_config),
        
        # Regenerate summary plot
        summarize_s2_vs_field
    ]
    
    return pipe_run(run, *steps)