
def run_calibration_analysis(run: Run,
                            ion_fitted_areas: Optional[Dict[str, S2Areas]] = None,
                            xray_bin_cuts: tuple = (0.6, 20),
                            xray_nbins: int = 100,
                            flag_plot: bool = True,
                            save_plots: bool = True) -> tuple:
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
        from .core.dataIO import save_figure, load_xray_results, store_xray_areas_combined
        
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
