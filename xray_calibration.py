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
from .dataIO import load_s2area, load_xray_results


def _fit_gaussian_to_histogram(
    data: np.ndarray,
    bin_cuts: Tuple[float, float],
    nbins: int = 100,
    exclude_index: int = 0
):
    """
    Helper function to fit Gaussian to histogram data using lmfit.
    
    Args:
        data: Array of values to fit
        bin_cuts: (min, max) range for histogram
        nbins: Number of histogram bins
        exclude_index: Number of initial bins to exclude from fit (for pedestal removal)
        
    Returns:
        Tuple of (mean, sigma, ci95, bin_centers, bin_counts, fitted_model_result)
    """
    # Filter data within range
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    
    if len(filtered) == 0:
        raise ValueError(f"No data found in range {bin_cuts}")

    # Create histogram
    n, bins = np.histogram(filtered, bins=nbins, range=bin_cuts)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    
    # Exclude first bins if requested (for pedestal removal)
    if exclude_index > 0:
        n = n[exclude_index:]
        cbins = cbins[exclude_index:]

    # Fit Gaussian model using lmfit
    model = GaussianModel()
    params = model.make_params(
        amplitude=n.max(),
        center=np.mean(cbins),  # Use bin centers after exclusion
        sigma=np.std(cbins)
    )
    result = model.fit(n, params, x=cbins)

    # Extract parameters
    mean = result.params["center"].value
    sigma = result.params["sigma"].value
    stderr = result.params["center"].stderr
    ci95 = 1.96 * stderr if stderr else None

    return mean, sigma, ci95, cbins, n, result


def _plot_gaussian_fit(
    data: np.ndarray,
    bin_cuts: Tuple[float, float],
    nbins: int,
    fit_result,
    **plot_kwargs
):
    """
    Helper function to plot histogram with Gaussian fit.
    
    Args:
        data: Original data array
        bin_cuts: (min, max) histogram range
        nbins: Number of bins for histogram
        fit_result: lmfit fit result object
        **plot_kwargs: Plotting options (xlabel, title, data_label, color)
    """
    # Extract plot settings with defaults
    xlabel = plot_kwargs.get('xlabel', 'Value')
    title = plot_kwargs.get('title', 'Distribution')
    data_label = plot_kwargs.get('data_label', 'Data')
    color = plot_kwargs.get('color', 'blue')
    
    # Extract fit parameters
    mean = fit_result.params["center"].value
    sigma = fit_result.params["sigma"].value
    stderr = fit_result.params["center"].stderr
    ci95 = 1.96 * stderr if stderr is not None else None
    ci95_str = f"{ci95:.2f}" if ci95 is not None else "N/A"
    
    # Extract unit from xlabel if present
    unit = xlabel.split("(")[1].split(")")[0] if "(" in xlabel else ""
    
    plt.figure(figsize=(8, 6))
    
    # Plot histogram
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    plt.hist(filtered, bins=nbins, alpha=0.6, color=color, label=data_label)
    
    # Plot fit curve using the SAME number of bins for smooth curve
    cbins = np.linspace(bin_cuts[0], bin_cuts[1], nbins * 3)
    model = GaussianModel()
    plt.plot(cbins, model.eval(x=cbins, params=fit_result.params), 
            'r--', linewidth=2, 
            label=f'Gaussian fit\n$\\mu={mean:.2f}$ ± ${ci95_str}$ {unit}')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def fit_xray_histogram(
    xray_areas: np.ndarray,
    bin_cuts: Tuple[float, float] = (0.6, 20),
    nbins: int = 100,
    flag_plot: bool = False
) -> Tuple[float, float, float]:
    """
    Fit Gaussian to X-ray S2 area distribution.

    Args:
        xray_areas: Array of X-ray S2 areas (mV·µs)
        bin_cuts: (min, max) range for histogram
        nbins: Number of histogram bins
        flag_plot: If True, display fit result

    Returns:
        Tuple of (mean, sigma, ci95) where ci95 is 95% confidence interval
    """
    mean, sigma, ci95, cbins, n, fit_result = _fit_gaussian_to_histogram(
        xray_areas, bin_cuts, nbins
    )
    
    if flag_plot:
        _plot_gaussian_fit(
            xray_areas, bin_cuts, nbins, fit_result,
            xlabel='S2 Area (mV·µs)',
            title='X-ray S2 Area Distribution',
            data_label='X-ray events',
            color='green'
        )

    return mean, sigma, ci95


def fit_s2_timing_histogram(
    data: np.ndarray,
    bin_cuts: Tuple[float, float],
    nbins: int = 100,
    flag_plot: bool = False,
    timing_type: str = "duration"
) -> Tuple[float, float, float]:
    """
    Fit Gaussian to S2 timing histogram (start, end, or duration).
    
    Args:
        data: Array of times (µs)
        bin_cuts: (t_min, t_max) time range
        nbins: Number of histogram bins
        flag_plot: Whether to plot fit
        timing_type: "start", "end", or "duration"
        
    Returns:
        Tuple of (mean_time, std_time, ci95) in µs
    """
    labels = {
        "start": {
            'xlabel': "S2 Start Time (µs)",
            'title': "S2 Start Time Distribution",
            'data_label': "S2 Start Times",
            'color': 'blue'
        },
        "end": {
            'xlabel': "S2 End Time (µs)",
            'title': "S2 End Time Distribution",
            'data_label': "S2 End Times",
            'color': 'purple'
        },
        "duration": {
            'xlabel': "S2 Duration (µs)",
            'title': "S2 Duration Distribution",
            'data_label': "S2 Durations",
            'color': 'orange'
        }
    }
    
    if timing_type not in labels:
        raise ValueError(f"timing_type must be one of {list(labels.keys())}")
    
    mean, sigma, ci95, cbins, n, fit_result = _fit_gaussian_to_histogram(
        data, bin_cuts, nbins
    )
    
    if flag_plot:
        _plot_gaussian_fit(
            data, bin_cuts, nbins, fit_result,
            **labels[timing_type]
        )
    
    return mean, sigma, ci95


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
        accepted_events = [e.area for e in xray_result.events if e.accepted and e.area is not None]
        xray_areas.extend(accepted_events)
        print(f"  → {set_name}: {len(accepted_events)} accepted events")
    
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
        from .dataIO import save_figure
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
