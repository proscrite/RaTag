"""
Complete X-ray classification pipeline.

This pipeline executes the X-ray event classification workflow using functional composition:
1. X-ray classification and integration (set-level workflows)
2. (Future) Quality assurance plots
3. (Future) Multi-isotope distribution analysis

Structure mirrors run_preparation.py and recoil_only.py for consistency.
All stages are explicit and composable - no hidden flags or conditionals.
"""

from functools import partial
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.config import XRayConfig
from RaTag.core.functional import pipe_run
from RaTag.core.dataIO import aggregate_xray_areas
from RaTag.workflows.xray_integration import classify_xrays_in_run, validate_xray_classification, fit_xray_areas, run_xray_multiiso
from RaTag.alphas.energy_join import generic_multiiso_workflow


# ============================================================================
# FUNCTIONAL PIPELINE COMPOSITION
# ============================================================================

def _map_combined_xray_to_isotopes(run: Run, isotope_ranges: dict) -> Run:
    """
    Helper to map combined X-ray areas to isotopes using dummy SetPmt.
    
    Creates a dummy SetPmt with run.root_directory as source_dir to reuse
    generic_multiiso_workflow infrastructure.
    """
    # Create dummy SetPmt instance
    dummy_set = SetPmt(
        source_dir=run.root_directory,
        filenames=[],
        nframes=0,
        metadata={}
    )
    
    # Call generic workflow with combined data
    generic_multiiso_workflow(
        set_pmt=dummy_set,
        data_filename=f"{run.run_id}_xray_areas_combined.npz",
        value_keys=["xray_areas"],
        isotope_ranges=isotope_ranges,
        output_suffix="xray_areas_combined_multi",
        plot_columns=["xray_areas"],
        bins=40
    )
    
    return run  # Unchanged


def xray_pipeline(run: Run,
                  range_sets: Optional[slice] = None,
                  max_frames: Optional[int] = None,
                  xray_config: XRayConfig = XRayConfig(),
                  isotope_ranges: Optional[dict] = None) -> Run:
    """
    Execute complete X-ray classification pipeline.
    
    Prerequisites:
    - Run must be prepared (see run_preparation.prepare_run)
    - Sets must have t_s1 in metadata
    - Sets should have t_s2_start in metadata (or time_drift as fallback)
    
    Pipeline stages:
    1. classify_xrays_in_run: Complete ETL for all sets
       - Classification → Integration → Storage
       - Uses caching for efficiency
    2. (Future) Quality assurance plots
    3. (Future) Multi-isotope distribution analysis
    
    Each stage is explicit and can be customized/reordered.
    
    Args:
        run: Prepared Run object
        range_sets: Optional slice to process subset of sets
        max_frames: Optional limit on frames per set (None = process all)
        xray_config: X-ray classification parameters
        
    Returns:
        Run with X-ray statistics in set metadata
        
    Example:
        >>> from RaTag.pipelines import prepare_run, xray_pipeline
        >>> run = prepare_run(my_run)
        >>> run = xray_pipeline(run, max_frames=10000)
    """
    print("\n" + "="*60)
    print(f"X-RAY CLASSIFICATION PIPELINE: {run.run_id}")
    print("="*60)
    
    # Build pipeline stages
    steps = [
        # Stage 1: Complete set-level ETL (classification + integration + storage)
        partial(classify_xrays_in_run,
                range_sets=range_sets,
                max_frames=max_frames,
                xray_config=xray_config),
        
        # Stage 2: QA validation plots (accepted vs rejected examples)
        partial(validate_xray_classification, n_waveforms=5),
        
        # Stage 3: Aggregation over all sets
        aggregate_xray_areas,

        # Stage 4: Fit Gaussian to combined distribution
        fit_xray_areas,
    ]
    
    # Stage 5: Multi-isotope mapping (optional)
    if isotope_ranges is not None:
        # 5a: Per-set isotope mapping
        steps.append(partial(run_xray_multiiso, isotope_ranges=isotope_ranges))
        # 5b: Combined isotope mapping
        steps.append(partial(_map_combined_xray_to_isotopes, isotope_ranges=isotope_ranges))
    
    # Execute pipeline
    result = pipe_run(run, *steps)
    
    print("\n" + "="*60)
    print("X-RAY CLASSIFICATION COMPLETE")
    print("="*60)
    
    return result


# ============================================================================
# PIPELINE VARIATIONS
# ============================================================================

def xray_pipeline_multiiso(run: Run,
                           isotope_ranges: dict,
                           range_sets: Optional[slice] = None,
                           max_frames: Optional[int] = None,
                           xray_config: XRayConfig = XRayConfig()) -> Run:
    """
    Complete X-ray classification pipeline with multi-isotope mapping.
    
    Executes full pipeline including per-set and combined isotope mapping.
    This is equivalent to calling xray_pipeline(..., isotope_ranges=isotope_ranges).
    
    Args:
        run: Prepared Run object
        isotope_ranges: {isotope: (Emin, Emax)} in keV
        range_sets: Optional slice to process subset of sets
        max_frames: Optional limit on frames per set
        xray_config: X-ray classification parameters
        
    Returns:
        Run with X-ray statistics and isotope mappings
    """
    return xray_pipeline(run,
                        range_sets=range_sets,
                        max_frames=max_frames,
                        xray_config=xray_config,
                        isotope_ranges=isotope_ranges)


def xray_pipeline_quick(run: Run, max_frames: int = 5000) -> Run:
    """
    Quick test version - limited frames, basic classification only.
    
    Useful for testing classification settings on a small sample.
    """
    return classify_xrays_in_run(run, max_frames=max_frames)



# =========================================================================
# DOWNSTREAM PIPELINE (post-classification only)
# =========================================================================

def xray_downstream_pipeline(run: Run,
                            xray_config: Optional[XRayConfig] = None,
                            isotope_ranges: Optional[Dict] = None,
                            n_validation_frames: int = 5) -> Run:
    """
    Perform all downstream X-ray operations after classification.
    Includes QA plots, aggregation, fitting, and multi-isotope analysis if requested.
    """
    if xray_config is None:
        xray_config = XRayConfig()
    steps = [
        partial(validate_xray_classification, n_waveforms=n_validation_frames),
        aggregate_xray_areas,
        fit_xray_areas
    ]
    if isotope_ranges is not None:
        steps += [
            partial(run_xray_multiiso, isotope_ranges=isotope_ranges),
            partial(_map_combined_xray_to_isotopes, isotope_ranges=isotope_ranges)
        ]
    return pipe_run(run, *steps)


# =========================================================================
# RE-PLOTTING PIPELINE
# =========================================================================

def xray_pipeline_replot(run: Run, 
                         n_validation_frames: int = 5,
                         isotope_ranges: Optional[dict] = None) -> Run:
    """
    Regenerate plots from cached X-ray classification results.
    
    Useful when you want to adjust plotting parameters without
    recomputing classifications. Regenerates validation plots and
    optionally isotope mapping plots.
    
    Prerequisites:
    - Sets must have n_accepted in metadata (from previous classification)
    - Combined X-ray areas file must exist
    
    Args:
        run: Run with sets containing cached X-ray results
        n_validation_frames: Number of frames per validation plot
        isotope_ranges: Optional isotope ranges for multi-isotope plots
        
    Returns:
        Run (unchanged, just regenerates plots)
    """
    print("\n" + "="*60)
    print("REGENERATE X-RAY PLOTS FROM CACHE")
    print("="*60)
    
    # Check that we have cached results
    sets_with_results = [s for s in run.sets if 'n_accepted' in s.metadata]
    
    if len(sets_with_results) == 0:
        print("  ⚠ No cached classification results found - run classification first")
        return run
    
    print(f"  Found {len(sets_with_results)} sets with cached results")
    
    # Build replot pipeline
    steps = [
        # Validation plots
        partial(validate_xray_classification, n_waveforms=n_validation_frames),
        
        # Refit and replot histogram
        fit_xray_areas,
    ]
    
    # Add isotope mapping plots if requested
    if isotope_ranges is not None:
        steps.append(partial(run_xray_multiiso, isotope_ranges=isotope_ranges))
        steps.append(partial(_map_combined_xray_to_isotopes, isotope_ranges=isotope_ranges))
    
    return pipe_run(run, *steps)
