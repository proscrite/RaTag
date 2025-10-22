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

from .constructors import (populate_run, set_fields, set_transport_properties, 
                          estimate_s1_from_frames, set_from_dir, s2_variance_run)
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
    s1 = estimate_s1_from_frames(s, max_waveforms=n_batches*batch_size, threshold_s1=0.1, flag_plot=flag_plot)
    s1 = set_fields(s1, drift_gap_cm=run.drift_gap, el_gap_cm=run.el_gap, gas_density=run.gas_density)
    s1 = set_transport_properties(s1, drift_gap_cm=run.drift_gap, transport=None)
    return s1


def prepare_run(
    run: Run,
    n_batches: int = 5,
    batch_size: int = 20,
    flag_plot: bool = False,
    nfiles: Optional[int] = None,
    estimate_s2_windows: bool = True,
    s2_duration_cuts: tuple = (5, 25),
    threshold_s2: float = 0.4,
    max_waveforms_s2: Optional[int] = None
) -> Run:
    """
    Complete run preparation pipeline:
    1. Add gas density
    2. Populate sets from directory structure
    3. Prepare each set (S1 estimation, fields, transport)
    4. Optionally estimate S2 windows for refined integration

    Args:
        run: Run object with root_directory and experiment parameters
        n_batches: Number of batches for S1 estimation
        batch_size: Files per batch for S1 estimation
        flag_plot: If True, show diagnostic plots during preparation
        nfiles: If provided, limit each set to this many files (for testing)
        estimate_s2_windows: If True, estimate S2 timing windows from data
        s2_duration_cuts: (min, max) duration cuts in µs for S2 variance estimation
        threshold_s2: S2 detection threshold in mV
        max_waveforms_s2: Max waveforms for S2 window estimation (None = all, or use nfiles)

    Returns:
        New Run with all sets prepared and S2 windows estimated
    """
    print("=" * 60)
    print(f"PREPARING RUN: {run.run_id}")
    print("=" * 60)
    
    # Add gas density
    run = with_gas_density(run)
    print(f"\n[1/4] Gas density: {run.gas_density:.3e} cm⁻³")
    
    # Populate sets
    run = populate_run(run)
    
    # If nfiles is specified, limit files per set
    if nfiles is not None:
        print(f"\n[2/4] Limiting sets to {nfiles} files each (test mode)")
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=nfiles)
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    else:
        print(f"\n[2/4] Loaded {len(run.sets)} sets")
    
    # Prepare all sets (S1, fields, transport)
    print(f"\n[3/4] Preparing sets (S1 estimation, fields, transport)...")
    prepared_sets = []
    for i, s in enumerate(run.sets):
        print(f"  → Processing set {i+1}/{len(run.sets)}: {s.source_dir.name}")
        prepared_sets.append(prepare_set(s, run, n_batches=n_batches, batch_size=batch_size, flag_plot=flag_plot))
    
    run = replace(run, sets=prepared_sets)
    
    # Estimate S2 windows if requested
    if estimate_s2_windows:
        print(f"\n[4/4] Estimating S2 timing windows...")
        # Use nfiles as max_waveforms if specified, otherwise use max_waveforms_s2
        max_wf = nfiles if nfiles is not None else max_waveforms_s2
        
        run = s2_variance_run(
            run,
            s2_duration_cuts=s2_duration_cuts,
            threshold_s2=threshold_s2,
            max_waveforms=max_wf,
            method='percentile'
        )
        print("  → S2 timing statistics stored in set metadata")
    else:
        print(f"\n[4/4] Skipping S2 window estimation (estimate_s2_windows=False)")
    
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
    flag_plot: bool = False,
    use_estimated_s2_windows: bool = True
) -> Dict[str, S2Areas]:
    """
    Execute ion (Ra recoil) S2 area integration and fitting across a run.

    This integrates S2 signals in the expected time window and fits
    Gaussian distributions to extract mean areas per field setting.

    Args:
        run: Prepared Run object with sets populated
        ts2_tol: Time tolerance before S2 window start (µs). Ignored if use_estimated_s2_windows=True
                 and S2 windows were estimated in prepare_run().
        range_sets: Optional slice to process subset of sets
        integration_config: IntegrationConfig with integration parameters
        fit_config: FitConfig with fitting parameters
        nfiles: If provided, limit each set to this many files (for testing)
        flag_plot: If True, plot histograms with fits and S2 vs drift field
        use_estimated_s2_windows: If True, use S2 window statistics from set metadata
                                  (from s2_variance_run in prepare_run). If False, use ts2_tol.

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
            # Copy metadata from prepared set (including S2 window estimates if present)
            limited_set.metadata = s.metadata
            limited_set.drift_field = s.drift_field
            limited_set.EL_field = s.EL_field
            limited_set.time_drift = s.time_drift
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    
    # Integrate S2 areas
    if use_estimated_s2_windows:
        print("\n[1/3] Integrating S2 areas using estimated windows from metadata...")
    else:
        print("\n[1/3] Integrating S2 areas using ts2_tol offset...")
    
    areas = integrate_run_s2(run, ts2_tol=ts2_tol, range_sets=range_sets, 
                            integration_config=integration_config,
                            use_estimated_s2_windows=use_estimated_s2_windows)
    
    # Fit Gaussian distributions
    print("\n[2/3] Fitting Gaussian distributions...")
    fitted = fit_run_s2(areas, fit_config=fit_config, flag_plot=flag_plot)
    
    # Store results and generate plots
    print("\n[3/3] Storing results and generating plots...")
    
    # Create plots directory
    plots_dir = run.root_directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for set_pmt in run.sets:
        set_name = set_pmt.source_dir.name
        s2_result = fitted[set_name]
        
        # Store S2 areas with complete metadata
        from .dataIO import store_s2area, save_figure
        store_s2area(s2_result, set_pmt=set_pmt)
        
        # Generate and save histogram plot
        fig_hist, _ = plotting.plot_hist_fit(s2_result, nbins=fit_config.nbins, 
                                             bin_cuts=fit_config.bin_cuts)
        save_figure(fig_hist, plots_dir / f"{set_name}_s2_histogram.png")
        
        # Generate and save waveform validation plot (10 random waveforms)
        if len(set_pmt.filenames) > 0:
            n_waveforms = min(10, len(set_pmt.filenames))
            fig_wf, _ = plotting.plot_waveforms_with_s1_s2(
                set_pmt,
                n_waveforms=n_waveforms,
                t_s1_mean=set_pmt.metadata.get("t_s1"),
                t_s1_std=set_pmt.metadata.get("t_s1_std"),
                t_s2_start_mean=set_pmt.metadata.get("t_s2_start_mean"),
                t_s2_start_std=set_pmt.metadata.get("t_s2_start_std"),
                t_s2_end_mean=set_pmt.metadata.get("t_s2_end_mean"),
                t_s2_end_std=set_pmt.metadata.get("t_s2_end_std"),
                figsize=(10, 4 * n_waveforms)
            )
            save_figure(fig_wf, plots_dir / f"{set_name}_waveform_validation.png")
    
    # Generate summary plots
    if flag_plot:
        print("\nGenerating summary plots...")
        
        # S2 vs drift field (unnormalized)
        fig_drift, _ = plotting.plot_s2_vs_drift(run, fitted, normalized=False)
        save_figure(fig_drift, plots_dir / f"{run.run_id}_s2_vs_drift.png")
    
    print("\n" + "=" * 60)
    print("ION S2 INTEGRATION COMPLETE")
    print("=" * 60)
    
    return fitted


def run_calibration_analysis(
    run: Run,
    ion_fitted_areas: Optional[Dict[str, S2Areas]] = None,
    xray_bin_cuts: tuple = (0.6, 20),
    xray_nbins: int = 100,
    flag_plot: bool = True,
    save_plots: bool = True
) -> tuple:
    """
    Execute complete calibration and recombination analysis with comprehensive plotting.

    This combines X-ray calibration data with ion S2 measurements to:
    1. Extract gain factor (g_S2) from X-ray energy calibration
    2. Normalize ion S2 areas using X-ray reference
    3. Compute electron recombination fractions vs drift field
    4. Generate and save all diagnostic plots

    Args:
        run: Run object with X-ray and ion data
        ion_fitted_areas: Dictionary of fitted ion S2 areas per set.
                         If None, will attempt to load from disk using load_s2area().
        xray_bin_cuts: Range for X-ray histogram fitting
        xray_nbins: Number of bins for X-ray histogram
        flag_plot: If True, generate diagnostic plots
        save_plots: If True, save plots to disk

    Returns:
        Tuple of (CalibrationResults, recombination_dict)
        
    Note:
        If ion_fitted_areas is not provided, the function will load stored
        S2 results from each set's directory. This makes the pipeline modular
        and allows running calibration separately from ion integration.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION & RECOMBINATION ANALYSIS")
    print("=" * 60)
    
    # Run calibration
    calib_results, recomb_dict = calibrate_and_analyze(
        run,
        ion_fitted_areas=ion_fitted_areas,
        xray_bin_cuts=xray_bin_cuts,
        xray_nbins=xray_nbins,
        flag_plot=flag_plot
    )
    
    if save_plots:
        print("\nGenerating and saving comprehensive plots...")
        from .dataIO import save_figure, load_xray_results, store_xray_areas_combined
        
        # Create plots directory
        plots_dir = run.root_directory / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and save combined X-ray areas
        try:
            xray_areas = load_xray_results(run)
            store_xray_areas_combined(xray_areas, run, plots_dir)
            
            # Plot combined X-ray histogram (this should be done in calibrate_and_analyze)
            # but we can regenerate it here for consistency
            print("  → Combined X-ray histogram...")
            
        except Exception as e:
            print(f"Warning: Could not process X-ray results: {e}")
        
        # Generate normalized S2 vs drift plot
        if ion_fitted_areas is not None:
            print("  → S2 vs drift (normalized)...")
            fig_norm, _ = plotting.plot_s2_vs_drift(run, ion_fitted_areas, normalized=True)
            save_figure(fig_norm, plots_dir / f"{run.run_id}_s2_vs_drift_normalized.png")
        
        # Generate diffusion analysis plots if S2 variance data is available
        print("  → Diffusion analysis...")
        try:
            drift_times = []
            sigma_obs_squared = []
            speeds_drift = []
            drift_fields = []
            
            for set_pmt in run.sets:
                if "s2_duration_std" in set_pmt.metadata:
                    drift_times.append(set_pmt.time_drift)
                    sigma_obs_squared.append(set_pmt.metadata["s2_duration_std"] ** 2)
                    speeds_drift.append(set_pmt.speed_drift)
                    drift_fields.append(set_pmt.drift_field)
            
            if len(drift_times) > 0:
                import numpy as np
                fig_diff, _ = plotting.plot_s2_diffusion_analysis(
                    np.array(drift_times),
                    np.array(sigma_obs_squared),
                    np.array(speeds_drift),
                    np.array(drift_fields),
                    run.pressure
                )
                save_figure(fig_diff, plots_dir / f"{run.run_id}_diffusion_analysis.png")
            else:
                print("  ⚠ No S2 variance data available for diffusion analysis")
                
        except Exception as e:
            print(f"Warning: Could not generate diffusion analysis plots: {e}")
    
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 60)
    
    return calib_results, recomb_dict
