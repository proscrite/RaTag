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
from RaTag.workflows.energy_mapping import create_energy_maps_in_run, plot_energy_spectra_in_run
from RaTag.workflows.spectrum_calibration import (
    fit_and_calibrate_spectrum,
    derive_isotope_ranges_from_calibration,
    plot_calibration_validation,
    fit_hierarchical_alpha_spectrum,
)


# ============================================================================
# MAIN PIPELINE - FUNCTIONAL COMPOSITION
# ============================================================================

def alpha_calibration(run: Run,
                     savgol_window: int = 501,
                     energy_range: tuple = (4, 8.2),
                     calibration_config: AlphaCalibrationConfig = AlphaCalibrationConfig()) -> Run:
    """
    Complete alpha spectrum calibration pipeline.
    
    This pipeline executes all steps for alpha energy calibration:
    1. Create energy maps (UID → energy binary files)
    2. Plot alpha energy spectra (QC)
    3. Fit preliminary peaks and derive calibration
    4. Derive isotope energy ranges
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
        
    Returns:
        Run with .alpha_calibration and .isotope_ranges attributes
        
    Example:
        >>> from RaTag.pipelines import alpha_calibration
        >>> from RaTag.core.config import AlphaCalibrationConfig
        >>> run = initialize_run(my_run)
        >>> config = AlphaCalibrationConfig(n_sigma=1.5, use_quadratic=True)
        >>> run = alpha_calibration(run, savgol_window=501, config=config)
    """
    print("\n" + "="*70)
    print(f"ALPHA SPECTRUM CALIBRATION PIPELINE: {run.run_id}")
    print("="*70)
    
    # Pipeline stages
    steps = [
        # Stage 1: Create energy maps (prerequisite)
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
                force_recompute=False),
        
        # Stage 4: Derive isotope ranges
        partial(derive_isotope_ranges_from_calibration,
                n_sigma=calibration_config.n_sigma,
                use_quadratic=calibration_config.use_quadratic,
                force_recompute=False),
        
        # Stage 5: Validation plot
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=False),
        
        # Stage 6: Hierarchical fit
        partial(fit_hierarchical_alpha_spectrum,
                force_recompute=False),
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
                force_recompute=True),
        
        # Regenerate validation plot
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=True),
        
        # Regenerate hierarchical fit plot
        partial(fit_hierarchical_alpha_spectrum,
                force_recompute=True),
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
                force_recompute=False),
        
        partial(derive_isotope_ranges_from_calibration,
                n_sigma=calibration_config.n_sigma,
                use_quadratic=calibration_config.use_quadratic,
                force_recompute=False),
        
        partial(plot_calibration_validation,
                use_quadratic=calibration_config.use_quadratic,
                force_replot=False),
    ]
    
    return pipe_run(run, *steps)