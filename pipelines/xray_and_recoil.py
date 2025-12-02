"""
Optimized pipeline module using unified integration workflow.

This module provides an optimized version of the analysis pipeline that:
1. Uses unified integration (X-ray + S2 in single pass)
2. Keeps S2 window estimation separate (only needs ~2k waveforms)
3. Keeps calibration as post-processing (loads from disk)

Performance improvement: ~50% reduction in processing time for large datasets.

Usage:
    from RaTag.pipeline_optimized import (
        prepare_run_optimized,
        run_unified_integration,
        run_s2_fitting,
        run_calibration_analysis_optimized
    )
    
    # Step 1: Prepare run (includes S2 window estimation with limited waveforms)
    run = prepare_run_optimized(run, estimate_s2_windows=True, max_waveforms_s2=2000)
    
    # Step 2: Unified integration (X-ray classification + S2 integration in ONE pass)
    xray_results, s2_areas = run_unified_integration(run)
    
    # Step 3: Fit S2 distributions (fast, no waveform loading)
    s2_fitted = run_s2_fitting(run, s2_areas)
    
    # Step 4: Calibration analysis (loads X-ray data from disk)
    calib, recomb = run_calibration_analysis_optimized(run, s2_fitted)
"""

from typing import Dict, Optional
from dataclasses import replace
from pathlib import Path
import numpy as np

from core.constructors import (estimate_s1_from_frames, set_from_dir, set_fields, set_transport_properties,
                                s2_variance_run, populate_run)
from core.datatypes import Run, SetPmt, S2Areas, XRayResults
from core.config import IntegrationConfig, FitConfig
from core.physics import with_gas_density
from core.dataIO import save_figure, store_s2area
from core.fitting import fit_run_s2
from workflows.unified_processing import integrate_run_unified
from workflows.xrays_calibration import calibrate_and_analyze
from .. import plotting as plotting


def prepare_run_unified(run: Run,
                        flag_plot: bool = False,
                        skip_s1: bool = False,
                        max_frames_s1: int = 1000,
                        threshold_s1: float = 1.0,
                        estimate_s2_windows: bool = True,
                        max_frames_s2: int = 10000,
                        s2_duration_cuts: tuple = (5, 25),
                        threshold_s2: float = 0.8) -> Run:
    """
    Complete run preparation pipeline (OPTIMIZED).
    
    This prepares the FULL run (all files) but uses limited frames for estimation:
    - S1 estimation: max_frames_s1 frames per set (default: 1000)
    - S2 window estimation: max_frames_s2 frames per set (default: 10000)
    
    The returned Run object contains ALL files and is ready for full integration.
    
    Steps:
    1. Add gas density
    2. Populate sets from directory structure (ALL files)
    3. Prepare each set (S1 estimation with limited frames, fields, transport)
    4. Optionally estimate S2 windows (with limited frames)
    
    Args:
        run: Run object with root_directory and experiment parameters
        flag_plot: If True, show diagnostic plots during preparation
        max_frames_s1: Target frames per set for S1 estimation (default: 1000)
                       Actual frames = ceil(max_frames_s1/nframes) × nframes
        estimate_s2_windows: If True, estimate S2 timing windows from data
        max_frames_s2: Target frames per set for S2 window estimation (default: 10000)
                       Actual frames = ceil(max_frames_s2/nframes) × nframes
        s2_duration_cuts: (min, max) duration cuts in µs for S2 variance estimation
        threshold_s2: S2 detection threshold in mV
    
    Returns:
        New Run with ALL files, prepared with S1 times and S2 windows estimated
    """
    print("=" * 60)
    print(f"PREPARING RUN (OPTIMIZED): {run.run_id}")
    print("=" * 60)
    
    # Add gas density
    run = with_gas_density(run)
    print(f"\n[1/4] Gas density: {run.gas_density:.3e} cm⁻³")
    
    # Populate sets with ALL files
    run = populate_run(run)
    print(f"\n[2/4] Loaded {len(run.sets)} sets (all files)")
    
    # Prepare all sets (S1, fields, transport)
    print(f"\n[3/4] Preparing sets (using ~{max_frames_s1} frames for S1 estimation)...")

    prepared_sets = []
    for i, s in enumerate(run.sets):
        print(f"\n  Set {i+1}/{len(run.sets)}: {s.source_dir.name}")
        
        if not skip_s1:
            s1 = estimate_s1_from_frames(s, max_frames=max_frames_s1, threshold_s1=threshold_s1, flag_plot=False)

        # Set fields and transport (uses metadata, not files)
        s1 = set_fields(s1, drift_gap_cm=run.drift_gap, el_gap_cm=run.el_gap, gas_density=run.gas_density)
        prepared_set = set_transport_properties(s1, drift_gap_cm=run.drift_gap, transport=None)
        
        prepared_sets.append(prepared_set)
    
    run = replace(run, sets=prepared_sets)
    
    # Estimate S2 windows if requested
    if estimate_s2_windows:
        print(f"\n[4/4] Estimating S2 timing windows (using ~{max_frames_s2} frames per set)...\n")
        
        run = s2_variance_run(
            run,
            s2_duration_cuts=s2_duration_cuts,
            threshold_s2=threshold_s2,
            max_frames=max_frames_s2,
            flag_plot=flag_plot,
            method='percentile'
        )
        print("  → S2 timing statistics stored in set metadata")
    else:
        print(f"\n[4/4] Skipping S2 window estimation (estimate_s2_windows=False)")
    
    print("\n" + "=" * 60)
    print("RUN PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Run contains ALL files and is ready for full integration")
    print(f"Use max_frames parameter in run_unified_integration() for testing with limited frames")
    return run


def run_unified_integration(
    run: Run,
    ts2_tol: float = -2.7,
    range_sets: Optional[slice] = None,
    xray_config: Optional[IntegrationConfig] = None,
    ion_config: Optional[IntegrationConfig] = None,
    max_frames: Optional[int] = None,
    use_estimated_s2_windows: bool = True
) -> tuple[Dict[str, XRayResults], Dict[str, S2Areas]]:
    """
    UNIFIED integration: X-ray classification + S2 integration in SINGLE pass.
    Args:
        run: Prepared Run object with sets populated (should contain ALL files)
        ts2_tol: Time tolerance before S2 window start (µs)
        range_sets: Optional slice to process subset of sets
        xray_config: X-ray classification configuration
        ion_config: Ion S2 integration configuration
        max_frames: TESTING ONLY - If provided, target number of frames per set
                    Actual frames = ceil(max_frames/nframes) × nframes
                    For production runs, leave as None to process all files
        use_estimated_s2_windows: If True, use S2 windows from metadata
    
    Returns:
        Tuple of (xray_results_dict, s2_areas_dict)
        - xray_results_dict: {set_name: XRayResults} - for calibration
        - s2_areas_dict: {set_name: S2Areas} - for fitting
    """
    # If max_frames specified for TESTING, create limited sets
    if max_frames is not None:
        # Compute files needed based on first set's nframes (assume uniform)
        typical_nframes = run.sets[0].nframes if run.sets else 1
        max_files = int(np.ceil(max_frames / typical_nframes))
        actual_frames = max_files * typical_nframes
        
        print(f"⚠ TEST MODE: limiting to ~{max_frames} frames per set ({max_files} files, ~{actual_frames} actual frames)")
        print(f"   For production, run without max_frames parameter to process all files\n")
        
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=max_files)
            # Copy metadata and physics properties from prepared set
            limited_set = replace(limited_set,
                                metadata=s.metadata,
                                drift_field=s.drift_field,
                                EL_field=s.EL_field,
                                time_drift=s.time_drift,
                                speed_drift=s.speed_drift)
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    else:
        total_files = sum(len(s) for s in run.sets)
        total_frames = sum(s.n_waveforms for s in run.sets)
        print(f"Production mode: processing ALL files ({total_files} files, {total_frames} frames)")
    
    # Run unified integration
    xray_results, s2_areas = integrate_run_unified(
        run,
        ts2_tol=ts2_tol,
        range_sets=range_sets,
        xray_config=xray_config,
        ion_config=ion_config,
        use_estimated_s2_windows=use_estimated_s2_windows
    )
    
    return xray_results, s2_areas


def run_s2_fitting(run: Run,
                    s2_areas_dict: Dict[str, S2Areas],
                    fit_config: Optional[FitConfig] = None,
                    flag_plot: bool = False,
                    save_plots: bool = True ) -> Dict[str, S2Areas]:
    """
    Fit Gaussian distributions to S2 area histograms.
    
    This is a FAST post-processing step (no waveform loading).
    Fits are done on the already-integrated S2 areas.
    
    Args:
        run: Run object
        s2_areas_dict: Dictionary of S2Areas from unified integration
        fit_config: Fitting configuration
        flag_plot: If True, generate plots
        save_plots: If True, save plots to disk
    
    Returns:
        Dictionary of S2Areas with fit results
    """
    if fit_config is None:
        fit_config = FitConfig()
    
    print("=" * 60)
    print("FITTING S2 DISTRIBUTIONS")
    print("=" * 60)
    
    # Fit all S2 distributions
    fitted = fit_run_s2(s2_areas_dict, fit_config=fit_config, flag_plot=flag_plot)
    
    # Generate and save plots if requested
    if save_plots:
        print("\nGenerating and saving plots...")
        plots_dir = run.root_directory / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-set plots
        for set_pmt in run.sets:
            set_name = set_pmt.source_dir.name
            if set_name not in fitted:
                continue
            
            s2_result = fitted[set_name]
            
            # Histogram plot
            fig_hist, _ = plotting.plot_hist_fit(
                s2_result, nbins=fit_config.nbins, bin_cuts=fit_config.bin_cuts
            )
            save_figure(fig_hist, plots_dir / f"{set_name}_s2_histogram.png")
            
            # Waveform validation plot (if we have S2 timing metadata)
            if 't_s2_start_mean' in set_pmt.metadata and len(set_pmt.filenames) > 0:
                n_waveforms = min(3, len(set_pmt.filenames))
                fig_wf = plotting.plot_set_windows(
                    set_pmt,
                )
                # save_figure(fig_wf, plots_dir / f"{set_name}_waveform_validation.png")
        
        # Run-level plot: S2 vs drift field
        if flag_plot:
            fig_drift, _ = plotting.plot_s2_vs_drift(run, fitted, normalized=False)
            save_figure(fig_drift, plots_dir / f"{run.run_id}_s2_vs_drift.png")
    
    print("\n" + "=" * 60)
    print("S2 FITTING COMPLETE")
    print("=" * 60)
    
    return fitted


def run_calibration_analysis_optimized(
    run: Run,
    ion_fitted_areas: Optional[Dict[str, S2Areas]] = None,
    xray_bin_cuts: tuple = (0.6, 20),
    xray_nbins: int = 100,
    flag_plot: bool = True,
    save_plots: bool = True
) -> tuple:
    """
    Execute calibration and recombination analysis (OPTIMIZED).
    
    This is identical to run_calibration_analysis() but loads X-ray data
    from disk (saved by unified integration).
    
    Args:
        run: Run object
        ion_fitted_areas: Optional S2Areas dict (loads from disk if None)
        xray_bin_cuts: Range for X-ray histogram fitting
        xray_nbins: Number of bins for X-ray histogram
        flag_plot: If True, generate diagnostic plots
        save_plots: If True, save plots to disk
    
    Returns:
        Tuple of (CalibrationResults, recombination_dict)
    """
    print("\n" + "=" * 60)
    print("CALIBRATION & RECOMBINATION ANALYSIS")
    print("=" * 60)
    print("Loading X-ray data from disk (saved by unified integration)...")
    
    # Run calibration (automatically loads X-ray and S2 data from disk)
    calib_results, recomb_dict = calibrate_and_analyze(
        run,
        ion_fitted_areas=ion_fitted_areas,
        xray_bin_cuts=xray_bin_cuts,
        xray_nbins=xray_nbins,
        flag_plot=flag_plot
    )
    
    if save_plots:
        print("\nGenerating and saving comprehensive plots...")
        from .core.dataIO import load_xray_results, store_xray_areas_combined
        
        # Create plots directory
        plots_dir = run.root_directory / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and save combined X-ray areas
        try:
            xray_areas = load_xray_results(run)
            store_xray_areas_combined(xray_areas, run, plots_dir)
            print("  → Combined X-ray data saved")
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
