"""
Alpha spectrum calibration pipeline.

This pipeline handles the complete alpha energy calibration workflow:
1. Energy map generation (UID → energy mapping files)
2. Alpha energy spectrum plotting for QC
3. Preliminary peak fitting and energy calibration
4. Isotope energy range derivation
5. Calibration validation plotting
6. Hierarchical 9-peak fitting with satellites

Structure mirrors run_preparation.py and recoil_only.py:
- Declarative composition using functools.partial
- No iteration logic (handled by workflow functions)
- Single-purpose, composable stages
"""

from functools import partial

from RaTag.core.datatypes import Run
from RaTag.core.config import AlphaCalibrationConfig
from RaTag.core.functional import pipe_run
from dataclasses import replace
from RaTag.core.constructors import populate_alpha_sets
from RaTag.workflows.energy_mapping import create_energy_maps_in_run, plot_energy_spectra_in_run
from RaTag.workflows.spectrum_calibration import (
    fit_and_calibrate_spectrum,
    derive_isotope_ranges_from_calibration,
    plot_calibration_validation,
    fit_hierarchical_alpha_spectrum,
    create_alpha_overlay,
)


# ============================================================================
# MAIN PIPELINE - FUNCTIONAL COMPOSITION
# ============================================================================

def alpha_calibration(run: Run,
                     savgol_window: int = 501,
                     energy_range: tuple = (4, 8.2),
                     calibration_config: AlphaCalibrationConfig = AlphaCalibrationConfig(),
                     force_refit: bool = False) -> Run:
    """
    Complete alpha spectrum calibration pipeline.
    
    This pipeline executes all steps for alpha energy calibration:
    1. Create energy maps (UID → energy binary files) - **CACHED: only if .bin files missing**
    2. Plot alpha energy spectra (QC)
    3. Fit preliminary peaks and derive calibration
    4. Derive isotope energy ranges (including overlap resolution)
    5. Generate calibration validation plot
    6. Fit hierarchical model (9 peaks with satellites)
    
    Prerequisites:
    - Run must be initialized (sets populated)
    - Raw alpha channel waveform files must exist

    Args:
        run: Initialized Run object
        savgol_window: Savitzky-Golay window size (default: 501 samples ≈ 100 ns)
                       Must be odd. Larger = more smoothing (21-5001 range)
        energy_range: (min, max) energy range for fitting [mV in SCA scale]
        calibration_config: AlphaCalibrationConfig with low-level parameters
        force_refit: If True, recompute fitting/calibration/ranges/plots (default: False)
                     **IMPORTANT**: Energy maps (.bin files) are NEVER recomputed by this flag.
                     To regenerate energy maps, delete the .bin files manually.
        
    Returns:
        Run with .alpha_calibration and .isotope_ranges attributes
        
    Example:
        >>> from RaTag.pipelines import alpha_calibration
        >>> from RaTag.core.config import AlphaCalibrationConfig
        >>> run = initialize_run(my_run)
        >>> config = AlphaCalibrationConfig(n_sigma=2.0, use_quadratic=True)
        >>> # First run: generates everything
        >>> run = alpha_calibration(run, savgol_window=501, config=config)
        >>> # Update n_sigma and refit (skips energy maps, recomputes calibration)
        >>> config = AlphaCalibrationConfig(n_sigma=1.5, use_quadratic=True)
        >>> run = alpha_calibration(run, config=config, force_refit=True)
    """
    print("\n" + "="*70)
    print(f"ALPHA SPECTRUM CALIBRATION PIPELINE: {run.run_id}")
    print("="*70)
    if force_refit:
        print("⚙️  FORCE REFIT MODE: Recomputing calibration/ranges/plots")
        print("   (Energy maps cached - delete .bin files to regenerate)")
        print("="*70)
    
    # Pipeline stages
    steps = [
        # Stage 1: Create energy maps (ALWAYS CACHED - checks for .bin files)
        # To regenerate, delete energy_maps/**/*.bin files
        partial(create_energy_maps_in_run,
                files_per_chunk=calibration_config.files_per_chunk,
                fmt=calibration_config.fmt,
                scale=calibration_config.scale,
                pattern=calibration_config.pattern,
                savgol_window=savgol_window),
        
        # Stage 2: Plot alpha spectra (QC)
        partial(plot_energy_spectra_in_run,
                nbins=calibration_config.nbins,
                energy_range=energy_range),
        
        # Stage 3: Fit and calibrate
        partial(fit_and_calibrate_spectrum,
                energy_range=energy_range,
                aggregate=True,
                force_refit=force_refit),
        
        # Stage 4: Derive isotope ranges (includes overlap resolution)
        partial(derive_isotope_ranges_from_calibration,
                n_sigma=calibration_config.n_sigma,
                use_quadratic=calibration_config.use_quadratic,
                force_refit=force_refit),
        
        # Stage 5: Validation plot (includes overlap resolution diagnostic)
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=force_refit),
        
        # Stage 6: Hierarchical fit
        partial(fit_hierarchical_alpha_spectrum,
                force_refit=force_refit),
    ]
    
    result = pipe_run(run, *steps)
    
    print("\n" + "="*70)
    print("ALPHA CALIBRATION PIPELINE COMPLETE")
    print("="*70)
    
    return result


# ============================================================================
# SECONDARY PIPELINES - REPLOTTING AND REFITTING
# ============================================================================

def alpha_calibration_replot(run: Run,
                            calibration_config: AlphaCalibrationConfig = AlphaCalibrationConfig()) -> Run:
    """
    Regenerate plots from cached calibration data.
    
    Useful when you want to:
    - Regenerate plots with different parameters
    - Update plots after adjusting n_sigma for ranges
    - Refresh plots without recomputing expensive fits
    
    Prerequisites:
    - Calibration NPZ files must exist (from previous alpha_calibration run)
    
    Skips:
    - Energy map generation (expensive I/O)
    - Peak fitting and calibration (expensive computation)
    
    Args:
        run: Run with existing calibration files
        calibration_config: AlphaCalibrationConfig (uses n_sigma and use_quadratic)
        
    Returns:
        Run with updated plots
    """
    print("\n" + "="*70)
    print("ALPHA CALIBRATION REPLOT")
    print("="*70)
    
    steps = [
        # Recompute ranges (cheap)
        partial(derive_isotope_ranges_from_calibration,
                n_sigma=calibration_config.n_sigma,
                use_quadratic=calibration_config.use_quadratic,
                force_refit=True),
        
        # Regenerate validation plot
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=True),
        
        # Regenerate hierarchical fit plot
        partial(fit_hierarchical_alpha_spectrum,
                force_refit=True),
    ]
    
    return pipe_run(run, *steps)


def alpha_calibration_quick(run: Run,
                           energy_range: tuple = (4, 8.2),
                           calibration_config: AlphaCalibrationConfig = AlphaCalibrationConfig()) -> Run:
    """
    Quick calibration without hierarchical fit.
    
    Useful for:
    - Testing calibration parameters quickly
    - Getting isotope ranges without full hierarchical fit
    - Iterating on preliminary fit settings
    
    Args:
        run: Run with energy maps
        energy_range: Energy range for fitting
        calibration_config: AlphaCalibrationConfig (uses n_sigma and use_quadratic)
        
    Returns:
        Run with calibration and ranges (no hierarchical fit)
    """
    print("\n" + "="*70)
    print("ALPHA CALIBRATION (QUICK MODE)")
    print("="*70)
    
    steps = [
        partial(fit_and_calibrate_spectrum,
                energy_range=energy_range,
                aggregate=True,
                force_refit=False),
        
        partial(derive_isotope_ranges_from_calibration,
                n_sigma=calibration_config.n_sigma,
                use_quadratic=calibration_config.use_quadratic,
                force_refit=False),
        
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=False),
    ]
    
    return pipe_run(run, *steps)


def alpha_calibration_singleiso(run: Run,
                                savgol_window: int = 501,
                                energy_range: tuple = (4, 8.2),
                                calibration_config: AlphaCalibrationConfig = AlphaCalibrationConfig(),
                                force_refit: bool = False) -> Run:
    """
    Alpha spectrum calibration for single-isotope runs with monitoring sets.
    
    This pipeline handles alpha monitoring calibration for single-isotope runs
    where alpha data is stored in separate Ch4_* directories. Automatically
    detects alpha monitoring sets and performs calibration.
    
    Prerequisites:
    - Run must be initialized (sets populated)
    - Raw alpha waveform files must exist in Ch4_* directories
    
    Args:
        run: Base Run object (alpha sets will be auto-detected)
        savgol_window: Savitzky-Golay window size (default: 501 samples)
        energy_range: (min, max) energy range for fitting [mV in SCA scale]
        calibration_config: AlphaCalibrationConfig with parameters
        force_refit: If True, recompute calibration/ranges/plots
        
    Returns:
        Run object (note: ephemeral, contains only alpha sets)
        
    Example:
        >>> alpha_calibration_singleiso(run, force_refit=True)

    This variant of the alpha calibration pipeline performs the usual
    calibration stages but modifies the plotting step for single-isotope
    monitoring runs:
      - Energy maps are generated (cached)
      - The existing per-set plots are created by plot_energy_spectra_in_run()
      - The aggregated plot created by the generic function is removed
        (not physically meaningful for mixed SCA/noSCA sets)
      - A new normalized overlay plot is created that overlays the
        per-set spectra after peak-normalization
      - The remaining calibration steps (fit, derive ranges, validation,
        hierarchical fit) are run as in the standard pipeline

    Returns an ephemeral run containing only the alpha monitoring sets.
    """

    # Auto-detect alpha monitoring sets and construct ephemeral runs
    alpha_run, alpha_run_noSCA = populate_alpha_sets(run)

    if not alpha_run.sets:
        print("No alpha monitoring sets found - skipping alpha calibration")
        return run

    print(f"\n{'='*70}")
    print("SINGLE-ISOTOPE ALPHA MONITORING CALIBRATION")
    print(f"Found {len(alpha_run.sets)} alpha monitoring sets: {[s.source_dir.name for s in alpha_run.sets]}")
    print(f"{'='*70}")

    # Stage 1: Create energy maps (cached check inside)
    create_energy_maps_in_run(alpha_run,
                              files_per_chunk=calibration_config.files_per_chunk,
                              fmt=calibration_config.fmt,
                              scale=calibration_config.scale,
                              pattern=calibration_config.pattern,
                              savgol_window=savgol_window)

    # Stage 2: Plot per-set spectra (this will also create an aggregated plot)
    plot_energy_spectra_in_run(alpha_run,
                              nbins=calibration_config.nbins,
                              energy_range=energy_range)

    # Post-process plots: delete aggregated and create normalized overlay via workflow
    create_alpha_overlay(alpha_run,
                         nbins=calibration_config.nbins,
                         energy_range=energy_range,
                         normalize='peak')

    # Continue with remaining calibration steps (fit, derive ranges, validation, hierarchical fit)
    # Calibration requires a noSCA set. If none was found, skip the calibration steps.
    if alpha_run_noSCA is None:
        print("  ⚠ No 'Ch4_noSCA' calibration set found; calibration requires a noSCA set. Skipping calibration pipeline.")
        return run

    calib_run = fit_and_calibrate_spectrum(alpha_run_noSCA,
                                           energy_range=energy_range,
                                           aggregate=True,
                                           force_refit=force_refit)

    calib_run = derive_isotope_ranges_from_calibration(calib_run,
                                                       n_sigma=calibration_config.n_sigma,
                                                       use_quadratic=calibration_config.use_quadratic,
                                                       force_refit=force_refit)

    calib_run = plot_calibration_validation(calib_run,
                                           use_quadratic=calibration_config.use_quadratic,
                                           force_replot=force_refit)

    calib_run = fit_hierarchical_alpha_spectrum(calib_run,
                                                force_refit=force_refit)

    print(f"\n✓ Alpha monitoring calibration complete")
    return calib_run