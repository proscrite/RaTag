# RaTag/alphas/alpha_calibration.py
from re import A

import numpy as np
from dataclasses import replace

from RaTag.core.datatypes import Run, SetAlpha
from RaTag.core.config import ALPHA_PEAK_DEFINITIONS, AlphaCalibrationConfig
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.io import file_ops
from RaTag.core.fitting import v_crystalball_left
from RaTag.core.paths import get_output_root
from RaTag.alphas.alphas_fitting import (
    select_roi,
    beta_continuum,
    fit_multi_crystalball_progressive,
    derive_energy_calibration,
    refine_overlapping_pair,
    resolve_likelihood_crossover,
    derive_isotope_ranges,
)

from RaTag.alphas.alphas_plotting import plot_calibration_summary

# ============================================================================
# 1. I/O Fit helper
# ============================================================================

def _load_alpha_fits(set_alpha: SetAlpha) -> dict:
    """Helper to cleanly load all standard alpha fits from disk for a single set."""
    fit_results = {}
    fits_dir = get_output_root(set_alpha.source_dir.parent) / "fits" / "alpha_fits"
    
    for peak in ALPHA_PEAK_DEFINITIONS:
        name = peak['name']
        fit_path = fits_dir / f"{set_alpha.source_dir.name}_{name}_alpha_fit.json"
        if fit_path.exists():
            fit_results[name] = file_ops.load_fit_result(
                fit_path, 
                funcdefs={'v_crystalball_left': v_crystalball_left, 'beta_continuum': beta_continuum}
            )
    return fit_results

# ============================================================================
# 1. SET-LEVEL ETL (Calibration & Range Derivation)
# ============================================================================
@allow_force
@load_cached_metadata(target_attr='calib_a')
@load_cached_alpha_fits()
@require_attributes('n_alpha_energies') 
@write_metadata(target_attr='calib_a')
@write_alpha_fits()
def resolve_set_calibration(set_alpha: SetAlpha, 
                            config: AlphaCalibrationConfig = AlphaCalibrationConfig(),
                            force: bool = False) -> tuple[SetAlpha, dict]:
                            
    print(f"  Calibrating alpha spectrum for {set_alpha.source_dir.name}...")
    
    arrays = file_ops.load_npz_arrays(set_alpha, 'alpha_energies')
    bin_centers, counts = select_roi(arrays['energies'], *config.energy_range)
    
    # Preliminary Fits
    fit_results = fit_multi_crystalball_progressive(bin_centers, counts, ALPHA_PEAK_DEFINITIONS)
    
    calibration = derive_energy_calibration(fit_results, ALPHA_PEAK_DEFINITIONS, 
                                            order=2 if config.use_quadratic else 1)
    
    crossovers = {}
    if 'Th228' in fit_results and 'Ra224' in fit_results:
        res_th, res_ra = refine_overlapping_pair(bin_centers, counts,
                                                 fit_results['Th228'], fit_results['Ra224'])
        
        fit_results['Th228'], fit_results['Ra224'] = res_th, res_ra  # Update fits with refined versions
        
        crossovers[('Th228', 'Ra224')] = resolve_likelihood_crossover(res_th, res_ra)
        print(f"    ✓ Bayesian overlap for Th228-Ra224 resolved at {crossovers[('Th228', 'Ra224')]:.3f} mV")

    # Commented out because it doesn't yield better results than the individual fits, and it can introduce instability in the calibration.
    # if 'Rn220' in fit_results and 'Bi212' in fit_results:  
    #     res_rn, res_bi = refine_overlapping_pair(bin_centers, counts,
    #                                                 fit_results['Rn220'], fit_results['Bi212'])
        
    #     fit_results['Rn220'], fit_results['Bi212'] = res_rn, res_bi  # Update fits with refined versions
        
    #     crossovers[('Rn220', 'Bi212')] = resolve_likelihood_crossover(res_rn, res_bi)
    #     print(f"    ✓ Bayesian overlap for Rn220-Bi212 resolved at {crossovers[('Rn220', 'Bi212')]:.3f} mV")

    ranges_V, ranges_E, mean_res = derive_isotope_ranges(fit_results, calibration,
                                                         n_sigma=config.n_sigma, crossovers=crossovers)
    
    print(f"    ✓ Calibration successful. Mean Resolution: {mean_res*100:.2f}%")

    # 6. State Update
    updated_set = replace(set_alpha, 
                          calib_a=calibration.a, 
                          calib_b=calibration.b, 
                          calib_c=calibration.c if calibration.order == 2 else None, 
                          calib_order=calibration.order,
                          mean_energy_resolution=mean_res,
                          isotope_ranges_V=ranges_V,
                          isotope_ranges_E=ranges_E)
                          
    return updated_set, fit_results

# ============================================================================
# 2. FITTING ORCHESTRATOR
# ============================================================================

def map_alpha_calibrations(run: Run, 
                           config: AlphaCalibrationConfig = AlphaCalibrationConfig(),
                           force: bool = False) -> Run:
    """Entry point: Calibrates each alpha set independently."""
    print("\n" + "="*60 + f"\nCALIBRATING ALPHA ENERGIES: {run.run_id}\n" + "="*60)
    
    bound_calibration = lambda s: resolve_set_calibration(s, config=config, force=force)
    
    # Map over sets independently
    updated_alpha_sets = map_over(run.alpha_sets, bound_calibration, catch_errors=True)
    
    return replace(run, alpha_sets=updated_alpha_sets)

# ============================================================================
# 3. QA PLOTTING
# ============================================================================
@allow_force
@load_cached_plots(subfolder="alpha_qa", expected_suffixes=["calibration_summary"])
@require_attributes('calib_a', 'isotope_ranges_V')
@write_plots(subfolder="alpha_qa")
def resolve_calibration_plot(set_alpha: SetAlpha, force: bool = False) -> tuple[SetAlpha, dict]:
    """Generates the QA plot for a single set using declarative requirements."""
    
    arrays = file_ops.load_npz_arrays(set_alpha, 'alpha_energies')
    energies_SCA = arrays['energies']
    
    fit_results = _load_alpha_fits(set_alpha)
    
    fig_summary = plot_calibration_summary(
        energies_SCA=energies_SCA,
        fit_results=fit_results,
        calib_coeffs=(set_alpha.calib_a, set_alpha.calib_b, set_alpha.calib_c),
        ranges_V=set_alpha.isotope_ranges_V,
        peak_definitions=ALPHA_PEAK_DEFINITIONS
    )
    
    # The @write_plots decorator automatically prefixes this key with the set_name!
    return set_alpha, {"calibration_summary": fig_summary}

def map_alpha_plots(run: Run, force: bool = False) -> Run:
    """Entry point: Maps the QA plotting workflow across all sets."""
    print("\n" + "="*60 + f"\nGENERATING ALPHA CALIBRATION PLOTS: {run.run_id}\n" + "="*60)
    
    bound_plot = lambda s: resolve_calibration_plot(s, force=force)
    map_over(run.alpha_sets, bound_plot, catch_errors=True)
    
    return run