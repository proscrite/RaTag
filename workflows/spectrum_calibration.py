"""
Alpha spectrum calibration workflows.

This module provides individual workflow steps for alpha spectrum calibration:
1. fit_and_calibrate_spectrum: Load spectrum, fit peaks, derive calibrations
2. derive_isotope_ranges_from_calibration: Compute isotope energy ranges
3. plot_calibration_validation: Generate calibration summary plot
4. fit_hierarchical_alpha_spectrum: Full 9-peak hierarchical fit

Each workflow is composable and can be chained in pipelines.
Workflows handle variable flows between low-level functions.
"""

from pathlib import Path
from typing import Optional, Dict
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

from RaTag.core.datatypes import Run
from RaTag.core.config import ALPHA_PEAK_DEFINITIONS, ALPHA_SATELLITE_DEFINITIONS
from RaTag.core.dataIO import save_figure
from RaTag.alphas.spectrum_fitting import (
    load_spectrum_from_run,
    fit_multi_crystalball_progressive,
    derive_energy_calibration,
    derive_isotope_ranges,
    prepare_hierarchical_fit,
    fit_full_spectrum_hierarchical,
    ranges_to_dict,
)
from RaTag.alphas.spectrum_plotting import (
    plot_calibration_summary,
    plot_hierarchical_fit,
    plot_overlap_resolution,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_computed_ranges(run_id: str, root_directory: Path) -> dict:
    """
    Load overlap-resolved isotope ranges from calibration output.
    
    Reads the .npz file created by derive_isotope_ranges_from_calibration()
    and extracts the resolved ranges (without '_windowed' suffix).
    
    Parameters
    ----------
    run_id : str
        Run identifier (e.g., 'RUN18')
    root_directory : Path
        Root directory of the run (to locate processed_data)
        
    Returns
    -------
    dict
        Isotope ranges as {isotope: (E_min, E_max)}
        
    Raises
    ------
    FileNotFoundError
        If calibration ranges file not found
    """
    ranges_file = root_directory / 'processed_data' / 'spectrum_calibration' / f'{run_id}_isotope_ranges.npz'
    
    if not ranges_file.exists():
        raise FileNotFoundError(f"Computed ranges not found: {ranges_file}\n"
                               f"Run with --alphas-only first to generate calibration.")
    
    # Load .npz file
    data = np.load(ranges_file)
    
    # Extract resolved ranges (keys without '_windowed' suffix)
    isotope_ranges = {}
    for key in data.keys():
        if key.endswith('_range') and not key.endswith('_range_windowed'):
            isotope = key.replace('_range', '')
            isotope_ranges[isotope] = tuple(data[key])
    
    return isotope_ranges


def _setup_output_directories(run: Run) -> tuple[Path, Path]:
    """Setup output directories for spectrum calibration."""
    plots_dir = run.root_directory / "plots" / "spectrum_calibration"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = run.root_directory / "processed_data" / "spectrum_calibration"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir

def _store_calibration(calib_file: Path, calibration_linear, calibration_quad) -> None:
    np.savez(calib_file,
            # Linear calibration
            linear_a=calibration_linear.a,
            linear_b=calibration_linear.b,
            linear_residuals=calibration_linear.residuals,
            # Quadratic calibration
            quad_a=calibration_quad.a,
            quad_b=calibration_quad.b,
            quad_c=calibration_quad.c,
            quad_residuals=calibration_quad.residuals,
            # Anchor points
            anchor_names=list(calibration_quad.anchors.keys()),
            anchor_E_SCA=[v[0] for v in calibration_quad.anchors.values()],
            anchor_E_true=[v[1] for v in calibration_quad.anchors.values()])

# ============================================================================
# WORKFLOW STEP 1: FIT AND CALIBRATE
# ============================================================================

def fit_and_calibrate_spectrum(run: Run,
                               energy_range: tuple[float, float] = (4, 8.2),
                               aggregate: bool = True,
                               force_refit: bool = False) -> Run:
    """
    Fit preliminary peaks and derive energy calibration.
    
    Workflow:
    1. Load aggregated spectrum from energy maps
    2. Fit main peaks (5 peaks) in SCA scale using Crystal Ball function
    3. Derive linear and quadratic energy calibrations
    4. Save calibration parameters to disk
    5. Attach results to run._alpha_calibration
    
    Args:
        run: Run with energy maps
        energy_range: (E_min, E_max) ROI for fitting [mV in SCA scale]
        aggregate: If True, combine all sets for better statistics
        force_refit: Force recomputation even if cached
        
    Returns:
        Run with ._alpha_calibration attribute containing:
            - fit_results: Dict of fitted peak parameters
            - calibration_linear: Linear EnergyCalibration
            - calibration_quad: Quadratic EnergyCalibration
            - spectrum: Original AlphaSpectrum
            - spectrum_calibrated: Calibrated energy array
            
    Side Effects:
        Creates: processed_data/spectrum_calibration/{run_id}_calibration.npz
    """
    
    plots_dir, data_dir = _setup_output_directories(run)
    calib_file = data_dir / f"{run.run_id}_calibration.npz"
    
    # Check cache
    if not force_refit and calib_file.exists():
        print(f"\n📂 Calibration already exists (skipping fit)")
        print(f"   Use force_refit=True to regenerate")
        # TODO: Load calibration from disk and attach to run
        return run
    
    print("\n[1/4] Fitting and calibrating alpha spectrum...")
    
    # Load spectrum
    spectrum = load_spectrum_from_run(run, energy_range=energy_range, aggregate=aggregate)
    print(f"  ✓ Loaded {len(spectrum.energies):,} alpha events")
    
    # Fit main peaks
    energies, counts = spectrum.select_roi()
    fit_results = fit_multi_crystalball_progressive(energies, counts,
                                                    peak_definitions=ALPHA_PEAK_DEFINITIONS,
                                                    global_beta_m=True)
    print(f"  ✓ Fitted {len(fit_results)} main peaks")
    
    # Derive calibrations
    literature_energies = {p['name']: p['ref_energy'] for p in ALPHA_PEAK_DEFINITIONS}
    
    calibration_linear = derive_energy_calibration(fit_results, literature_energies,
                                                   use_peaks=list(literature_energies.keys()),
                                                   order=1)
    
    calibration_quad = derive_energy_calibration(fit_results, literature_energies,
                                                 use_peaks=list(literature_energies.keys()),
                                                 order=2)
    
    rms_linear = np.sqrt(np.mean(calibration_linear.residuals**2))
    rms_quad = np.sqrt(np.mean(calibration_quad.residuals**2))
    print(f"  ✓ Linear calibration: RMS = {rms_linear*1000:.1f} keV")
    print(f"  ✓ Quadratic calibration: RMS = {rms_quad*1000:.1f} keV")
    print(f"    Improvement: {(1 - rms_quad/rms_linear)*100:.1f}%")
    
    # Apply quadratic calibration (recommended for Po212)
    spectrum_calibrated = calibration_quad.apply(spectrum.energies)
    
    # Save calibration to disk
    _store_calibration(calib_file, calibration_linear, calibration_quad)
    print(f"  💾 Saved calibration to {calib_file.name}")
    
    # TODO: Save fit_results (requires serialization strategy for lmfit.ModelResult)
    
    # Attach results to run (immutable - return new Run)
    calibration_data = {
        'fit_results': fit_results,
        'calibration_linear': calibration_linear,
        'calibration_quad': calibration_quad,
        'spectrum': spectrum,
        'spectrum_calibrated': spectrum_calibrated,
    }
    
    return replace(run, alpha_calibration=calibration_data)


# ============================================================================
# WORKFLOW STEP 2: DERIVE ISOTOPE RANGES
# ============================================================================

def derive_isotope_ranges_from_calibration(run: Run,
                                          n_sigma: float = 1.0,
                                          use_quadratic: bool = True,
                                          force_refit: bool = False) -> Run:
    """
    Derive isotope energy ranges from calibration.
    
    Workflow:
    1. Load calibration from run.alpha_calibration (or disk if needed)
    2. Compute isotope ranges based on fitted peaks
    3. Save ranges to disk
    4. Attach to run.isotope_ranges
    
    Args:
        run: Run with .alpha_calibration attribute
        n_sigma: Number of sigmas for range definition
        use_quadratic: Use quadratic (vs linear) calibration
        force_refit: Force recomputation even if cached
        
    Returns:
        Run with ._isotope_ranges attribute (dict of {isotope: (E_min, E_max)})
        
    Side Effects:
        Creates: processed_data/spectrum_calibration/{run_id}_isotope_ranges.npz
    """
    
    plots_dir, data_dir = _setup_output_directories(run)
    ranges_file = data_dir / f"{run.run_id}_isotope_ranges.npz"
    
    # Check cache
    if not force_refit and ranges_file.exists():
        print(f"\n📂 Isotope ranges already exist (skipping)")
        print(f"   Use force_refit=True to regenerate")
        # TODO: Load ranges from disk and attach to run
        return run
    
    print("\n[2/4] Deriving isotope energy ranges...")
    
    # Get calibration from run
    if run.alpha_calibration is None:
        raise RuntimeError("Missing run.alpha_calibration - run fit_and_calibrate_spectrum first")
    
    calib_data = run.alpha_calibration
    fit_results = calib_data['fit_results']
    calibration = calib_data['calibration_quad'] if use_quadratic else calib_data['calibration_linear']
    
    # Derive ranges
    literature_energies = {p['name']: p['ref_energy'] for p in ALPHA_PEAK_DEFINITIONS}
    
    isotope_ranges = derive_isotope_ranges(fit_results=fit_results,
                                          calibration=calibration,
                                          literature_energies=literature_energies,
                                          n_sigma=n_sigma)
    
    print(f"  ✓ Derived ranges for {len(isotope_ranges)} isotopes (windowed method)")
    
    # Resolve overlaps using likelihood crossover (Bayes-optimal boundaries)
    from RaTag.alphas.spectrum_fitting import resolve_overlapping_ranges
    
    isotope_ranges_resolved = resolve_overlapping_ranges(isotope_ranges=isotope_ranges,
                                                         fit_results=fit_results,
                                                         calibration=calibration,
                                                         overlap_pairs=[('Th228', 'Ra224')])  # Default: resolve Th228/Ra224 overlap
    
    print(f"  ✓ Resolved overlaps using likelihood crossover")
    
    # Save to disk (save both windowed and resolved ranges for diagnostics)
    ranges_dict_resolved = ranges_to_dict(isotope_ranges_resolved)
    ranges_dict_windowed = ranges_to_dict(isotope_ranges)
    
    np.savez(ranges_file, 
             **{f"{iso}_range": np.array(rng) for iso, rng in ranges_dict_resolved.items()},
             **{f"{iso}_range_windowed": np.array(rng) for iso, rng in ranges_dict_windowed.items()})
    print(f"  💾 Saved isotope ranges to {ranges_file.name}")
    
    # Store windowed ranges in calibration data for visualization
    calib_data_updated = {**calib_data, 'isotope_ranges_windowed': isotope_ranges}
    
    # Attach to run (immutable - return new Run with RESOLVED ranges and windowed in calibration)
    return replace(run, 
                  isotope_ranges=isotope_ranges_resolved,
                  alpha_calibration=calib_data_updated)


# ============================================================================
# WORKFLOW STEP 3: VALIDATION PLOT
# ============================================================================

def plot_calibration_validation(run: Run,
                                use_quadratic: bool = True,
                                force_replot: bool = False) -> Run:
    """
    Generate calibration summary validation plot.
    
    Workflow:
    1. Load calibration and ranges from run attributes
    2. Generate 4-panel summary plot (fits + calibration + ranges)
    3. Save plot
    
    Args:
        run: Run with ._alpha_calibration and ._isotope_ranges
        use_quadratic: Use quadratic calibration for spectrum
        force_replot: Force regeneration even if plot exists
        
    Returns:
        Unchanged run
        
    Side Effects:
        Creates: plots/spectrum_calibration/{run_id}_calibration_summary.png
    """
    
    plots_dir, data_dir = _setup_output_directories(run)
    summary_plot = plots_dir / f"{run.run_id}_calibration_summary.png"
    
    # Check cache
    if not force_replot and summary_plot.exists():
        print(f"\n📂 Calibration summary plot already exists (skipping)")
        print(f"   Use force_replot=True to regenerate")
        return run
    
    print("\n[3/4] Generating calibration validation plot...")
    
    # Get data from run
    if run.alpha_calibration is None or run.isotope_ranges is None:
        raise RuntimeError("Missing calibration data - run previous workflow steps first")
    
    calib_data = run.alpha_calibration
    spectrum = calib_data['spectrum']
    fit_results = calib_data['fit_results']
    calibration_linear = calib_data['calibration_linear']
    calibration_quad = calib_data['calibration_quad']
    spectrum_calibrated = calib_data['spectrum_calibrated']
    isotope_ranges = run.isotope_ranges
    
    # Generate plot
    peak_names = [p['name'] for p in ALPHA_PEAK_DEFINITIONS]
    
    fig, axes = plot_calibration_summary(spectrum=spectrum,
                                         fit_results=fit_results,
                                         peak_names=peak_names,
                                         calibration_linear=calibration_linear,
                                         calibration_quad=calibration_quad,
                                         spectrum_calibrated=spectrum_calibrated,
                                         isotope_ranges=isotope_ranges,
                                         figsize=(16, 12))
    
    save_figure(fig, summary_plot)
    plt.close(fig)
    print(f"  📊 Saved calibration summary to {summary_plot.name}")
    
    # Generate overlap resolution diagnostic plot (if windowed ranges available)
    if 'isotope_ranges_windowed' in calib_data:
        overlap_plot = plots_dir / f"{run.run_id}_overlap_resolution.png"
        
        if force_replot or not overlap_plot.exists():
            isotope_ranges_windowed = calib_data['isotope_ranges_windowed']
            calibration = calibration_quad  # Use quadratic for overlap resolution
            
            fig_overlap, axes_overlap = plot_overlap_resolution(spectrum_calibrated=spectrum_calibrated,
                                                                fit_results=fit_results,
                                                                calibration=calibration,
                                                                isotope_ranges_windowed=isotope_ranges_windowed,
                                                                isotope_ranges_resolved=isotope_ranges,
                                                                overlap_pairs=[('Th228', 'Ra224')],
                                                                figsize=(18, 6))
            
            save_figure(fig_overlap, overlap_plot)
            plt.close(fig_overlap)
            print(f"  📊 Saved overlap resolution diagnostic to {overlap_plot.name}")
    
    return run


# ============================================================================
# WORKFLOW STEP 4: HIERARCHICAL FIT
# ============================================================================

def fit_hierarchical_alpha_spectrum(run: Run,
                                   force_refit: bool = False) -> Run:
    """
    Fit full spectrum with hierarchical 9-peak model (including satellites).
    
    Workflow:
    1. Load calibration from run attributes
    2. Prepare hierarchical fit (average shape parameters)
    3. Fit 9 peaks (5 main + 4 satellites)
    4. Plot hierarchical fit with components
    5. Save plot and fit results
    
    Args:
        run: Run with .alpha_calibration attribute
        force_refit: Force recomputation even if cached
        
    Returns:
        Unchanged run
        
    Side Effects:
        Creates: plots/spectrum_calibration/{run_id}_hierarchical_fit.png
        TODO: Save fit results to disk
    """
    
    plots_dir, data_dir = _setup_output_directories(run)
    hierarchical_plot = plots_dir / f"{run.run_id}_hierarchical_fit.png"
    
    # Check cache
    if not force_refit and hierarchical_plot.exists():
        print(f"\n📂 Hierarchical fit plot already exists (skipping)")
        print(f"   Use force_refit=True to regenerate")
        return run
    
    print("\n[4/4] Hierarchical fitting (9 peaks with satellites)...")
    
    # Get calibration from run
    if run.alpha_calibration is None:
        raise RuntimeError("Missing run.alpha_calibration - run fit_and_calibrate_spectrum first")
    
    calib_data = run.alpha_calibration
    fit_results = calib_data['fit_results']
    calibration_linear = calib_data['calibration_linear']
    spectrum_calibrated = calib_data['spectrum_calibrated']
    
    # Prepare hierarchical fit
    calibrated_sigmas, beta_avg, m_avg, energies_cal, counts_cal = prepare_hierarchical_fit(
        fit_results=fit_results,
        calibration=calibration_linear,  # Use linear for shape transformation
        spectrum_calibrated=spectrum_calibrated,
        exclude_from_shape=['Po212']
    )
    
    # Bi212 uses Rn220 shape (wasn't in preliminary fits)
    calibrated_sigmas['Bi212'] = calibrated_sigmas['Rn220']
    
    # Fit full spectrum
    result_hierarchical = fit_full_spectrum_hierarchical(energies_cal, counts_cal,
                                                         peak_definitions=ALPHA_PEAK_DEFINITIONS + ALPHA_SATELLITE_DEFINITIONS,
                                                         calibrated_sigmas=calibrated_sigmas,
                                                         beta_avg=beta_avg, m_avg=m_avg,
                                                         normalize=True, x0_tolerance=0.04)
    
    print(f"  ✓ Hierarchical fit complete (χ²_red = {result_hierarchical.redchi:.3f})")
    
    # Plot
    fig, (ax1, ax2) = plot_hierarchical_fit(energies_cal, counts_cal, result_hierarchical,
                                           figsize=(14, 10), height_ratios=(2, 1))
    save_figure(fig, hierarchical_plot)
    plt.close(fig)
    print(f"  📊 Saved hierarchical fit to {hierarchical_plot.name}")
    
    # TODO: Save fit results to disk (requires serialization strategy)
    
    return run
