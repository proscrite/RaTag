"""
Pipeline module for orchestrating complete RaTag analysis workflows.

This module provides high-level functions that coordinate:
1. X-ray event classification and area extraction
2. Ion (Ra recoil) S2 area integration and fitting
3. Energy calibration and recombination analysis

All functions follow functional programming principles with immutable data structures.
"""

from typing import Dict, Optional
from dataclasses import replace

from .constructors import populate_run, set_fields, set_transport_properties, estimate_s1_from_batches, set_from_dir
from .analysis import integrate_run_s2, fit_run_s2
from .xray_integration import classify_xrays_run
from .xray_calibration import calibrate_and_analyze
from .dataIO import store_s2area, load_s2area
from .datatypes import Run, SetPmt, S2Areas, CalibrationResults
from .config import IntegrationConfig, FitConfig
from .transport import with_gas_density
import RaTag.plotting as plotting


def prepare_set(s: SetPmt, run: Run, n_batches: int = 5, batch_size: int = 20, flag_plot: bool = False) -> SetPmt:
    """
    Prepare a single set for analysis by:
    1. Estimating S1 time from batch analysis
    2. Setting drift and EL field values
    3. Computing transport properties (drift time, velocity, etc.)

    Args:
        s: SetPmt to prepare
        run: Run object with experiment parameters
        n_batches: Number of batches for S1 estimation
        batch_size: Files per batch for S1 estimation
        flag_plot: If True, show S1 distribution plot

    Returns:
        New SetPmt with all fields populated
    """
    s1 = estimate_s1_from_batches(s, n_batches=n_batches, batch_size=batch_size, flag_plot=flag_plot)
    s1 = set_fields(s1, drift_gap_cm=run.drift_gap, el_gap_cm=run.el_gap, gas_density=run.gas_density)
    s1 = set_transport_properties(s1, drift_gap_cm=run.drift_gap, transport=None)
    return s1


def prepare_run(
    run: Run,
    n_batches: int = 5,
    batch_size: int = 20,
    flag_plot: bool = False,
    nfiles: Optional[int] = None
) -> Run:
    """
    Complete run preparation pipeline:
    1. Add gas density
    2. Populate sets from directory structure
    3. Prepare each set (S1 estimation, fields, transport)

    Args:
        run: Run object with root_directory and experiment parameters
        n_batches: Number of batches for S1 estimation
        batch_size: Files per batch for S1 estimation
        flag_plot: If True, show diagnostic plots during preparation
        nfiles: If provided, limit each set to this many files (for testing)

    Returns:
        New Run with all sets prepared
    """
    print("=" * 60)
    print(f"PREPARING RUN: {run.run_id}")
    print("=" * 60)
    
    # Add gas density
    run = with_gas_density(run)
    print(f"\n[1/3] Gas density: {run.gas_density:.3e} cm⁻³")
    
    # Populate sets
    run = populate_run(run)
    
    # If nfiles is specified, limit files per set
    if nfiles is not None:
        print(f"\n[2/3] Limiting sets to {nfiles} files each (test mode)")
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=nfiles)
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    else:
        print(f"\n[2/3] Loaded {len(run.sets)} sets")
    
    # Prepare all sets
    print(f"\n[3/3] Preparing sets (S1 estimation, fields, transport)...")
    prepared_sets = []
    for i, s in enumerate(run.sets):
        print(f"  → Processing set {i+1}/{len(run.sets)}: {s.source_dir.name}")
        prepared_sets.append(prepare_set(s, run, n_batches=n_batches, batch_size=batch_size, flag_plot=flag_plot))
    
    run = replace(run, sets=prepared_sets)
    
    print("\n" + "=" * 60)
    print("RUN PREPARATION COMPLETE")
    print("=" * 60)
    
    return run


def run_xray_classification(
    run: Run,
    ts2_tol: float = -2.7,
    range_sets: Optional[slice] = None,
    config: IntegrationConfig = IntegrationConfig(),
    nfiles: Optional[int] = None
) -> Dict[str, any]:
    """
    Execute X-ray event classification across all sets in a run.

    This identifies X-ray-like signals in the drift region between S1 and S2,
    classifying events as accepted/rejected based on signal quality criteria.

    Args:
        run: Prepared Run object with sets populated
        ts2_tol: Time tolerance before S2 window start (µs)
        range_sets: Optional slice to process subset of sets
        config: IntegrationConfig with analysis parameters
        nfiles: If provided, limit each set to this many files (for testing)

    Returns:
        Dictionary mapping set_id -> XRayResults
    """
    print("\n" + "=" * 60)
    print("X-RAY CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # If nfiles is specified, create limited version of run
    if nfiles is not None:
        print(f"\nLimiting sets to {nfiles} files each (test mode)")
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=nfiles)
            # Copy metadata from prepared set
            limited_set.metadata = s.metadata
            limited_set.drift_field = s.drift_field
            limited_set.EL_field = s.EL_field
            limited_set.time_drift = s.time_drift
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    
    results = classify_xrays_run(run, ts2_tol=ts2_tol, range_sets=range_sets, config=config)
    
    print("\n" + "=" * 60)
    print("X-RAY CLASSIFICATION COMPLETE")
    print("=" * 60)
    
    return results


def run_ion_integration(
    run: Run,
    ts2_tol: float = -2.7,
    range_sets: Optional[slice] = None,
    integration_config: IntegrationConfig = IntegrationConfig(),
    fit_config: FitConfig = FitConfig(),
    nfiles: Optional[int] = None,
    flag_plot: bool = False
) -> Dict[str, S2Areas]:
    """
    Execute ion (Ra recoil) S2 area integration and fitting across a run.

    This integrates S2 signals in the expected time window and fits
    Gaussian distributions to extract mean areas per field setting.

    Args:
        run: Prepared Run object with sets populated
        ts2_tol: Time tolerance before S2 window start (µs)
        range_sets: Optional slice to process subset of sets
        integration_config: IntegrationConfig with integration parameters
        fit_config: FitConfig with fitting parameters
        nfiles: If provided, limit each set to this many files (for testing)
        flag_plot: If True, plot histograms with fits and S2 vs drift field

    Returns:
        Dictionary mapping set_id -> S2Areas (with fit results)
    """
    print("\n" + "=" * 60)
    print("ION S2 INTEGRATION PIPELINE")
    print("=" * 60)
    
    # If nfiles is specified, create limited version of run
    if nfiles is not None:
        print(f"\nLimiting sets to {nfiles} files each (test mode)")
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=nfiles)
            # Copy metadata from prepared set
            limited_set.metadata = s.metadata
            limited_set.drift_field = s.drift_field
            limited_set.EL_field = s.EL_field
            limited_set.time_drift = s.time_drift
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    
    # Integrate S2 areas
    print("\n[1/3] Integrating S2 areas...")
    areas = integrate_run_s2(run, ts2_tol=ts2_tol, range_sets=range_sets, integration_config=integration_config)
    
    # Fit Gaussian distributions
    print("\n[2/3] Fitting Gaussian distributions...")
    fitted = fit_run_s2(areas, fit_config=fit_config, flag_plot=flag_plot)
    
    # Store results
    print("\n[3/3] Storing results to disk...")
    for s2 in fitted.values():
        store_s2area(s2)
    
    # Generate summary plot if requested
    if flag_plot:
        print("\nGenerating S2 vs drift field plot...")
        plotting.plot_s2_vs_drift(run, fitted)
    
    print("\n" + "=" * 60)
    print("ION S2 INTEGRATION COMPLETE")
    print("=" * 60)
    
    return fitted


def run_calibration_analysis(
    run: Run,
    ion_fitted_areas: Optional[Dict[str, S2Areas]] = None,
    xray_bin_cuts: tuple = (0.6, 20),
    xray_nbins: int = 100,
    flag_plot: bool = True
) -> tuple:
    """
    Execute complete calibration and recombination analysis.

    This combines X-ray calibration data with ion S2 measurements to:
    1. Extract gain factor (g_S2) from X-ray energy calibration
    2. Normalize ion S2 areas using X-ray reference
    3. Compute electron recombination fractions vs drift field

    Args:
        run: Run object with X-ray and ion data
        ion_fitted_areas: Dictionary of fitted ion S2 areas per set.
                         If None, will attempt to load from disk using load_s2area().
        xray_bin_cuts: Range for X-ray histogram fitting
        xray_nbins: Number of bins for X-ray histogram
        flag_plot: If True, generate diagnostic plots

    Returns:
        Tuple of (CalibrationResults, recombination_dict)
        
    Note:
        If ion_fitted_areas is not provided, the function will load stored
        S2 results from each set's directory. This makes the pipeline modular
        and allows running calibration separately from ion integration.
    """
    return calibrate_and_analyze(
        run,
        ion_fitted_areas=ion_fitted_areas,
        xray_bin_cuts=xray_bin_cuts,
        xray_nbins=xray_nbins,
        flag_plot=flag_plot
    )
