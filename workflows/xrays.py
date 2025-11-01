"""
Calibration module for X-ray based energy calibration and recombination analysis.

This module provides functions to:
1. Fit X-ray S2 area distributions to extract calibration constants
2. Compute gain factors (g_S2) from known X-ray energies
3. Calculate electron recombination fractions from ion S2 areas
"""

from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
from dataclasses import replace

from .datatypes import Run, S2Areas, CalibrationResults, XRayResults
from .core.dataIO import load_s2area, load_xray_results


def compute_calibration_constants(
    A_x_mean: float,
    E_gamma: float,
    W_value: float
) -> Tuple[float, float]:
    """
    Compute calibration constants from X-ray measurements.

    Args:
        A_x_mean: Mean X-ray S2 area (mV·µs)
        E_gamma: X-ray photon energy (eV)
        W_value: Mean energy per electron-ion pair (eV)

    Returns:
        Tuple of (N_e_exp, g_S2) where:
            - N_e_exp: Expected number of electrons
            - g_S2: Gain factor (mV·µs per electron)
    """
    N_e_exp = E_gamma / W_value
    g_S2 = A_x_mean / N_e_exp
    return N_e_exp, g_S2


def compute_recombination(
    ion_areas: np.ndarray,
    ion_uncertainties: np.ndarray,
    drift_fields: np.ndarray,
    g_S2: float,
    N_e_exp: float,
    A_x_mean: float
) -> Dict[str, np.ndarray]:
    """
    Compute recombination fractions from ion S2 areas.

    Args:
        ion_areas: Array of mean ion S2 areas per field (mV·µs)
        ion_uncertainties: Array of uncertainties (mV·µs)
        drift_fields: Array of drift field values (V/cm)
        g_S2: Gain factor (mV·µs per electron)
        N_e_exp: Expected electrons from X-ray
        A_x_mean: Mean X-ray S2 area (mV·µs)

    Returns:
        Dictionary with:
            - 'N_e_meas': Measured electron numbers
            - 'dN_e_meas': Uncertainties on N_e_meas
            - 'r': Recombination fractions
            - 'dr': Uncertainties on r
            - 'norm_A_ion': Normalized ion areas (A_ion / A_x)
            - 'dnorm_A_ion': Uncertainties on normalized areas
    """
    # Measured electron numbers
    N_e_meas = ion_areas / g_S2
    dN_e_meas = ion_uncertainties / g_S2

    # Recombination fraction: r = 1 - N_e_meas / N_e_exp
    r = 1 - N_e_meas / N_e_exp
    dr = dN_e_meas / N_e_exp

    # Normalized areas
    norm_A_ion = ion_areas / A_x_mean
    dnorm_A_ion = ion_uncertainties / A_x_mean

    return {
        'N_e_meas': N_e_meas,
        'dN_e_meas': dN_e_meas,
        'r': r,
        'dr': dr,
        'norm_A_ion': norm_A_ion,
        'dnorm_A_ion': dnorm_A_ion,
        'drift_fields': drift_fields
    }


def calibrate_and_analyze(
    run: Run,
    ion_fitted_areas: Optional[Dict[str, S2Areas]] = None,
    xray_bin_cuts: Tuple[float, float] = (0.6, 20),
    xray_nbins: int = 100,
    flag_plot: bool = True
) -> Tuple[CalibrationResults, Dict[str, np.ndarray]]:
    """
    Complete calibration and recombination analysis pipeline.

    Args:
        run: Run object with X-ray and ion data
        ion_fitted_areas: Dictionary of fitted ion S2 areas per set.
                         If None, will attempt to load from disk.
        xray_bin_cuts: Range for X-ray histogram fitting
        xray_nbins: Number of bins for X-ray histogram
        flag_plot: If True, generate diagnostic plots

    Returns:
        Tuple of (CalibrationResults, recombination_dict)
    """
    print("=" * 60)
    print("CALIBRATION AND RECOMBINATION ANALYSIS")
    print("=" * 60)
    
    # 1. Load X-ray data and extract accepted events
    print("\n[1/5] Loading X-ray results...")
    xray_results_dict = load_xray_results(run)
    
    # Extract accepted X-ray areas from all sets
    xray_areas = []
    for set_name, xray_result in xray_results_dict.items():
        # accepted_events = [e.area for e in xray_result.events if e.accepted and e.area is not None]
        xray_areas.extend(xray_result.events)
        print(f"  → {set_name}: {len(xray_result.events)} accepted events")
    
    xray_areas = np.array(xray_areas)
    print(f"  → Total: {len(xray_areas)} accepted X-ray events")
    
    print("\n[2/5] Fitting X-ray histogram...")
    A_x_mean, sigma_x, ci95_x = fit_xray_histogram(
        xray_areas, 
        bin_cuts=xray_bin_cuts,
        nbins=xray_nbins,
        flag_plot=flag_plot
    )
    print(f"  → X-ray mean: {A_x_mean:.3f} ± {ci95_x:.3f} mV·µs")
    print(f"  → X-ray sigma: {sigma_x:.3f} mV·µs")
    
    # Save X-ray histogram plot
    if flag_plot:
        from .core.dataIO import save_figure
        plots_dir = run.root_directory / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save X-ray histogram plot
        mean_fit, sigma_fit, ci95_fit, cbins, n, fit_result = _fit_gaussian_to_histogram(
            xray_areas, xray_bin_cuts, xray_nbins
        )
        
        import RaTag.plotting as plotting
        fig_xray, _ = plotting.plot_xray_histogram(
            xray_areas, run.run_id, xray_nbins, xray_bin_cuts,
            fit_result=fit_result, mean=mean_fit, ci95=ci95_fit
        )
        save_figure(fig_xray, plots_dir / f"{run.run_id}_xray_histogram.png")
    
    # 2. Compute calibration constants
    print("\n[3/5] Computing calibration constants...")
    N_e_exp, g_S2 = compute_calibration_constants(
        A_x_mean,
        run.E_gamma_xray,
        run.W_value
    )
    print(f"  → Expected electrons (N_e_exp): {N_e_exp:.1f}")
    print(f"  → Gain factor (g_S2): {g_S2:.4f} mV·µs/electron")
    
    # 3. Load ion fitted areas if not provided
    if ion_fitted_areas is None:
        print("\n[4/5] Loading ion S2 fitted areas from disk...")
        ion_fitted_areas = {}
        for s in run.sets:
            try:
                s2_data = load_s2area(s)
                if s2_data.fit_success and s2_data.mean is not None:
                    ion_fitted_areas[s.source_dir.name] = s2_data
                else:
                    print(f"  ⚠ Warning: No fit results for {s.source_dir.name}")
            except Exception as e:
                print(f"  ⚠ Error loading S2 data for {s.source_dir.name}: {e}")
        
        if not ion_fitted_areas:
            raise ValueError("No ion S2 fitted areas found. Run ion integration first.")
        
        print(f"  → Loaded fit results for {len(ion_fitted_areas)} sets")
    else:
        print(f"\n[4/5] Using provided ion fitted areas ({len(ion_fitted_areas)} sets)")
    
    # 4. Extract ion S2 areas and compute recombination
    print("\n[5/5] Computing recombination fractions...")
    drift_fields = np.array([s.drift_field for s in run.sets])
    ion_means = np.array([ion_fitted_areas[s.source_dir.name].mean for s in run.sets])
    ion_ci95s = np.array([ion_fitted_areas[s.source_dir.name].ci95 for s in run.sets])
    
    recomb_results = compute_recombination(
        ion_means,
        ion_ci95s,
        drift_fields,
        g_S2,
        N_e_exp,
        A_x_mean
    )
    
    print(f"  → Recombination fractions computed for {len(drift_fields)} field points")
    print(f"  → r range: [{recomb_results['r'].min():.3f}, {recomb_results['r'].max():.3f}]")
    
    # 5. Generate plots if requested
    if flag_plot:
        _plot_calibration_results(run, recomb_results, A_x_mean)
    
    # Create CalibrationResults object
    calib_results = CalibrationResults(
        run_id=run.run_id,
        A_x_mean=A_x_mean,
        N_e_exp=N_e_exp,
        g_S2=g_S2
    )
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    
    return calib_results, recomb_results


def _plot_calibration_results(
    run: Run,
    recomb_results: Dict[str, np.ndarray],
    A_x_mean: float
):
    """Generate diagnostic plots for calibration results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Normalized ion S2 areas
    ax1.errorbar(
        recomb_results['drift_fields'],
        recomb_results['norm_A_ion'],
        yerr=recomb_results['dnorm_A_ion'],
        fmt='o',
        markersize=8,
        capsize=5,
        label='$A_{ion} / A_x$'
    )
    ax1.set_xlabel('Drift Field (V/cm)', fontsize=12)
    ax1.set_ylabel('Normalized Ion S2 Area', fontsize=12)
    ax1.set_title(f'{run.run_id}: Ion S2 Area vs Drift Field', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Recombination fraction
    ax2.errorbar(
        recomb_results['drift_fields'],
        recomb_results['r'],
        yerr=recomb_results['dr'],
        fmt='s',
        markersize=8,
        capsize=5,
        color='crimson',
        label='Recombination Fraction $r$'
    )
    ax2.set_xlabel('Drift Field (V/cm)', fontsize=12)
    ax2.set_ylabel('Recombination Fraction $r$', fontsize=12)
    ax2.set_title(f'{run.run_id}: Electron Recombination vs Drift Field', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
