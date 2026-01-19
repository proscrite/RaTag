from typing import Dict, Tuple, Any
from dataclasses import replace
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from lmfit.models import GaussianModel # type: ignore
import lmfit # type: ignore
import pandas as pd # type: ignore

from .datatypes import S2Areas
from .config import FitConfig


def compute_hist_variance(hist_array: np.ndarray,
                       duration_cuts: tuple[float, float] = None,
                       method: str = 'percentile') -> tuple[float, float]:
    """
    Compute robust variance estimate from S2 duration distribution.
    
    Args:
        hist_array: input array to histogram
        duration_cuts: Optional (min, max) cuts to remove outliers
        method: 'percentile' (16-84), 'mad', or 'std'
        
    Returns:
        (central_value, spread) in µs
    """
    if duration_cuts is not None:
        mask = (hist_array >= duration_cuts[0]) & (hist_array <= duration_cuts[1])
        array_clean = hist_array[mask]
    else:
        array_clean = hist_array
    
    if len(array_clean) < 10:
        raise ValueError(f"Too few events after cuts: {len(array_clean)}")

    if method == 'percentile':
        p16, p50, p84 = np.percentile(array_clean, [16, 50, 84])
        sigma_lower = p50 - p16
        sigma_upper = p84 - p50
        spread = (sigma_upper + sigma_lower) / 2
        return p50, spread
    
    elif method == 'mad':
        median = np.median(array_clean)
        mad = 1.4826 * np.median(np.abs(array_clean - median))
        return median, mad
    
    elif method == 'std':
        return np.mean(array_clean), np.std(array_clean)

    else:
        raise ValueError(f"Unknown method: {method}")


def fit_gaussian_to_histogram(data: np.ndarray,
                              bin_cuts: Tuple[float, float],
                              nbins: int = 100,
                              exclude_index: int = 0) -> Tuple[float, float, float, np.ndarray, np.ndarray, Any]:
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
    data = np.array(data)

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
    ci95 = 1.96 * stderr if stderr else mean
    
    return mean, sigma, ci95, cbins, n, result


def plot_gaussian_fit(data: np.ndarray,
                      bin_cuts: Tuple[float, float],
                      nbins: int,
                      fit_result,
                      **plot_kwargs):
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
    
    plt.gca().set(xlabel=xlabel, ylabel='Counts', title=title)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig = plt.gcf()  # Get current figure
    return fig

# -------------------------------------------------
#  S1/S2 timing histogram fit
# -------------------------------------------------

def _create_histogram(data: np.ndarray, 
                      bin_cuts: Tuple[float, float], 
                      nbins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to create histogram from data.
    
    Parameters
    ----------
    data : array-like
        Raw data values
    bin_cuts : tuple
        (min, max) range for histogram
    nbins : int
        Number of histogram bins
        
    Returns
    -------
    tuple of (counts, bins, bin_centers)
        counts : np.ndarray
            Histogram counts
        bins : np.ndarray
            Bin edges
        bin_centers : np.ndarray
            Center of each bin
    """
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    counts, bins = np.histogram(filtered, bins=nbins, range=bin_cuts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    return counts, bins, bin_centers

def fit_s2_timing_histogram(data: np.ndarray,
                            bin_cuts: Tuple[float, float],
                            nbins: int = 100, 
                            flag_plot: bool = False,
                            timing_type: str = "duration") -> Tuple[float, float, float]:
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
    
    mean, sigma, ci95, cbins, n, fit_result = fit_gaussian_to_histogram(data=data,
                                                                         bin_cuts=bin_cuts,
                                                                         nbins=nbins, 
                                                                         exclude_index=0)
    
    if flag_plot:
        plot_gaussian_fit(data, bin_cuts, nbins, fit_result,
                          **labels[timing_type])
    
    return mean, sigma, ci95

# -------------------------------------------------
#  ION RECOIL S2 AREA FITTING
# -------------------------------------------------

# --------------------------#
# Advanced fitting functions
# --------------------------#
def v_crystalball_right(x, N, beta, m, x0, sigma):
    """
    Crystal Ball function with RIGHT tail for ionization signals.
    
    Parameters
    ----------
    x : array-like
        Input variable (S2 area values)
    N : float
        Normalization (amplitude)
    beta : float
        Tail steepness parameter (transition point)
    m : float
        Power-law exponent for tail (typically > 1)
    x0 : float
        Peak location
    sigma : float
        Width parameter (Gaussian core width)
        
    Returns
    -------
    array-like
        Crystal Ball function values
        
    Notes
    -----
    This version has the tail on the RIGHT side (z >= beta), appropriate
    for ionization signals where charge collection effects create a right tail.
    Standard Crystal Ball has LEFT tail for PMT resolution effects.
    """
    absb = np.abs(beta)
    z = (x - x0) / sigma
    
    # Gaussian core
    gauss = np.exp(-0.5 * z**2)
    
    # Power-law tail parameters
    A_tail = (m / absb)**m * np.exp(-0.5 * absb**2)
    B = m / absb - absb
    
    # RIGHT tail: use +z (not -z) in denominator
    tail = A_tail / (B + z)**m
    
    # Use Gaussian for z < beta, tail for z >= beta
    return N * np.where(z < absb, gauss, tail)


def find_main_cluster(counts, threshold_fraction=0.01):
    """
    Find the main connected cluster of bins in a histogram.
    
    Removes isolated bins separated by gaps (bins with counts below threshold).
    Useful for removing disconnected edge bins after background subtraction.
    
    Parameters
    ----------
    counts : array-like
        Array of histogram bin counts
    threshold_fraction : float, optional
        Bins with counts < threshold_fraction * max(counts) are considered "empty"
        Default is 0.01 (1% of maximum)
        
    Returns
    -------
    mask : ndarray of bool
        Boolean mask indicating which bins belong to the largest connected cluster
        
    Notes
    -----
    Uses scipy.ndimage.label for connected component analysis.
    """
    from scipy.ndimage import label
    
    if len(counts) == 0:
        return np.zeros(len(counts), dtype=bool)
    
    threshold = threshold_fraction * counts.max()
    
    # Mark bins as "occupied" if above threshold
    occupied = counts > threshold
    
    # Find all connected regions (groups of True values separated by False)
    labeled, num_features = label(occupied)
    
    if num_features == 0:
        return occupied
    
    # Find the largest connected component
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_component = np.argmax(component_sizes) + 1
    
    # Return mask for the largest component
    return labeled == largest_component

def _fit_background_gaussian(cbins: np.ndarray, 
                             counts: np.ndarray, 
                             bg_cutoff: float = 1.0) -> Tuple[float, float, Any]:
    """
    Fit Gaussian model to background region of histogram.
    
    Parameters
    ----------
    cbins : np.ndarray
        Bin centers
    counts : np.ndarray
        Bin counts
    bg_cutoff : float, optional
        Upper limit for background fitting region
        
    Returns
    -------
    tuple of (bg_center, bg_sigma, result)
        bg_center : float
            Background peak center
        bg_sigma : float
            Background width
        result : lmfit.ModelResult
            Full fit result object
    """
    # Restrict to background region
    bg_mask = cbins <= bg_cutoff
    cbins_bg = cbins[bg_mask]
    n_bg = counts[bg_mask]
    
    # Fit Gaussian
    gauss_bg = GaussianModel(prefix='bg_')
    params_bg = gauss_bg.make_params(
        bg_amplitude=n_bg.max(),
        bg_center=0.3,
        bg_sigma=0.2
    )
    params_bg['bg_center'].set(min=0, max=bg_cutoff)
    params_bg['bg_sigma'].set(min=0.05, max=0.5)
    
    result_bg = gauss_bg.fit(n_bg, params=params_bg, x=cbins_bg)
    
    bg_center = result_bg.params['bg_center'].value
    bg_sigma = result_bg.params['bg_sigma'].value
    
    return bg_center, bg_sigma, result_bg


def _fit_signal_crystalball(cbins: np.ndarray,
                            counts: np.ndarray,
                            lower_bound: float,
                            upper_limit: float = 10.0) -> Tuple[Dict[str, float], Any]:
    """
    Fit Crystal Ball model to signal region of histogram.
    
    Parameters
    ----------
    cbins : np.ndarray
        Bin centers
    counts : np.ndarray
        Bin counts (should be background-subtracted for two-stage)
    lower_bound : float
        Lower bound for signal fitting region
    upper_limit : float, optional
        Upper limit for signal fitting region
        
    Returns
    -------
    tuple of (params_dict, result)
        params_dict : dict
            Dictionary with fitted parameters:
            - 'peak_position', 'peak_stderr', 'sigma', 'beta', 'm'
            - 'chi2', 'redchi'
        result : lmfit.ModelResult
            Full fit result object
    """
    # Apply cluster detection + bounds
    
    main_cluster_mask = find_main_cluster(counts, threshold_fraction=0.01)
    signal_mask = main_cluster_mask & (cbins >= lower_bound) & (cbins <= upper_limit)
    print(f"User defined signal region: [{lower_bound}, {upper_limit}]")
    print(f" Found main cluster in f{cbins[signal_mask]}")

    cbins_sig = cbins[signal_mask]
    n_sig = counts[signal_mask]
    
    # Fit Crystal Ball
    cb_sig = lmfit.Model(v_crystalball_right, prefix='sig_')
    params_sig = cb_sig.make_params(sig_N=n_sig.max(), 
                                    sig_x0=1.8, 
                                    sig_sigma=0.5, 
                                    sig_beta=1.0, 
                                    sig_m=2.0)
    
    params_sig['sig_x0'].set(min=lower_bound, max=9.5)
    params_sig['sig_sigma'].set(min=0.2, max=4.5)
    params_sig['sig_beta'].set(min=0.3, max=5.0)
    params_sig['sig_m'].set(min=1.1, max=10.0)
    
    result_sig = cb_sig.fit(n_sig, params=params_sig, x=cbins_sig)
    
    params_dict = {
        'peak_position': result_sig.params['sig_x0'].value,
        'peak_stderr': result_sig.params['sig_x0'].stderr,
        'sigma': result_sig.params['sig_sigma'].value,
        'beta': result_sig.params['sig_beta'].value,
        'm': result_sig.params['sig_m'].value,
        'chi2': result_sig.chisqr,
        'redchi': result_sig.redchi
    }
    
    return params_dict, result_sig


def detect_background_peak(data, bin_cuts=(0, 10), nbins=100, bg_threshold=0.3):
    """
    Detect if histogram has significant background pileup peak at low values.
    
    Parameters
    ----------
    data : array-like
        S2 area values
    bin_cuts : tuple, optional
        (min, max) range for histogram
    nbins : int, optional
        Number of histogram bins
    bg_threshold : float, optional
        Threshold for detecting background peak presence
        If ratio of counts in [0, bg_threshold] to total counts > 0.1, 
        background subtraction is recommended
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'has_background': bool, whether background peak is significant
        - 'bg_fraction': float, fraction of counts in low-value region
        - 'recommendation': str, 'two_stage' or 'simple'
        
    Notes
    -----
    This function helps decide whether to use simple Crystal Ball fitting
    or two-stage fitting with background subtraction.
    """
    # Create histogram
    n, bins, cbins = _create_histogram(data, bin_cuts, nbins)
    
    # Calculate fraction of counts in low-value region
    low_mask = cbins <= bg_threshold
    bg_counts = n[low_mask].sum()
    total_counts = n.sum()
    bg_fraction = bg_counts / total_counts if total_counts > 0 else 0.0
    
    # Decision criterion: if > 7% of counts are in low region, use two-stage
    has_background = bg_fraction > 0.077
    
    return {
        'has_background': has_background,
        'bg_fraction': bg_fraction,
        'recommendation': 'two_stage' if has_background else 'simple'
    }

def fit_s2_simple_cb(data, bin_cuts=(0, 10), nbins=100):
    """
    Simple single Crystal Ball fit for S2 area distributions without background.
    
    Use this when there's no significant background pileup peak (e.g., low field conditions).
    
    Parameters
    ----------
    data : array-like
        S2 area values
    bin_cuts : tuple, optional
        (min, max) range for histogram fitting
    nbins : int, optional
        Number of histogram bins
        
    Returns
    -------
    dict
        Fit results with keys:
        - 'peak_position': float, fitted peak location (x0)
        - 'peak_stderr': float, standard error on peak position
        - 'sigma': float, width parameter
        - 'beta': float, tail steepness
        - 'm': float, tail power
        - 'chi2': float, chi-squared
        - 'redchi': float, reduced chi-squared
        - 'result': lmfit.ModelResult object
        - 'histogram': dict with 'bins', 'counts', 'bin_centers'
        - 'method': str, 'simple'
        
    Notes
    -----
    Fits right-tailed Crystal Ball directly to the full histogram.
    """
    import lmfit
    
    # Create histogram
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    n, bins = np.histogram(filtered, bins=nbins, range=bin_cuts)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    
    # Fit Crystal Ball to full distribution
    cb_model = lmfit.Model(v_crystalball_right, prefix='sig_')
    
    # Initial parameters
    params = cb_model.make_params(
        sig_N=n.max(),
        sig_x0=1.5,
        sig_sigma=0.5,
        sig_beta=1.0,
        sig_m=2.0
    )
    
    # Constraints
    params['sig_x0'].set(min=0.5, max=bin_cuts[1] - 1.0)
    params['sig_sigma'].set(min=0.1, max=2.0)
    params['sig_beta'].set(min=0.3, max=5.0)
    params['sig_m'].set(min=1.1, max=10.0)
    
    # Fit
    result = cb_model.fit(n, params=params, x=cbins)
    
    return {
        'peak_position': result.params['sig_x0'].value,
        'peak_stderr': result.params['sig_x0'].stderr,
        'sigma': result.params['sig_sigma'].value,
        'beta': result.params['sig_beta'].value,
        'm': result.params['sig_m'].value,
        'chi2': result.chisqr,
        'redchi': result.redchi,
        'result': result,
        'histogram': {
            'bins': bins,
            'counts': n,
            'bin_centers': cbins
        },
        'method': 'simple'
    }


def fit_s2_two_stage(data, bin_cuts=(0, 10), nbins=100, bg_cutoff=1.0, 
                     n_sigma=2.5, upper_limit=5.0):
    """
    Two-stage fit for S2 area distributions with background pileup peak.
    
    Stage 1: Fit Gaussian to background in restricted range
    Stage 2: Subtract background, fit Crystal Ball to signal with smart region selection
    
    Parameters
    ----------
    data : array-like
        S2 area values
    bin_cuts : tuple, optional
        (min, max) range for histogram
    nbins : int, optional
        Number of histogram bins
    bg_cutoff : float, optional
        Upper limit for background-only fit region (default: 1.0 mV·µs)
    n_sigma : float, optional
        Number of sigmas above background peak for signal region lower bound
        (default: 2.5, recommended range: 2.0-3.0)
    upper_limit : float, optional
        Upper limit for signal fit region (default: 5.0 mV·µs)
        
    Returns
    -------
    dict
        Fit results with keys:
        - 'peak_position': float, signal peak location (x0)
        - 'peak_stderr': float, standard error on peak position
        - 'sigma': float, signal width parameter
        - 'beta': float, tail steepness
        - 'm': float, tail power
        - 'chi2': float, chi-squared
        - 'redchi': float, reduced chi-squared
        - 'bg_center': float, background peak center
        - 'bg_sigma': float, background width
        - 'lower_bound': float, calculated lower bound for signal region
        - 'result_bg': lmfit.ModelResult, background fit result
        - 'result_sig': lmfit.ModelResult, signal fit result
        - 'histogram': dict with 'bins', 'counts', 'bin_centers', 'subtracted'
        - 'method': str, 'two_stage'
        
    Notes
    -----
    Uses background statistics (μ_bg + n_sigma*σ_bg) to automatically determine
    signal region, avoiding hardcoded thresholds while maintaining robustness.
    """
    
    # Create histogram
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    n, bins = np.histogram(filtered, bins=nbins, range=bin_cuts)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    
    # ===== STAGE 1: Fit background =====


    print(f"    Fitting background Gaussian in range [0, {bg_cutoff}] mV·µs")
    bg_center, bg_sigma, result_bg = _fit_background_gaussian(cbins, n, bg_cutoff)
    
    # ===== STAGE 2: Subtract background and fit signal =====
    # Subtract background from full histogram
    from lmfit.models import GaussianModel
    gauss_bg = GaussianModel(prefix='bg_')
    bg_full = gauss_bg.eval(x=cbins, params=result_bg.params)
    n_subtracted = np.maximum(n - bg_full, 0)  # No negative counts
    
    # Calculate smart lower bound based on background statistics
    lower_bound = bg_center + n_sigma * bg_sigma
    
    # Fit signal region
    signal_params, result_sig = _fit_signal_crystalball(cbins, n_subtracted, 
                                                         lower_bound, upper_limit)
    
    return {
        **signal_params,  # Unpack peak_position, peak_stderr, sigma, beta, m, chi2, redchi
        'bg_center': bg_center,
        'bg_sigma': bg_sigma,
        'lower_bound': lower_bound,
        'result_bg': result_bg,
        'result_sig': result_sig,
        'histogram': {
            'bins': bins,
            'counts': n,
            'bin_centers': cbins,
            'subtracted': n_subtracted
        },
        'method': 'two_stage'
    }


def fit_s2_area_auto(data, bin_cuts=(0, 10), nbins=100, **kwargs):
    """
    Automatic S2 area fitting with intelligent method selection.
    
    Automatically detects whether background subtraction is needed and uses
    the appropriate fitting method (simple Crystal Ball or two-stage).
    
    Parameters
    ----------
    data : array-like
        S2 area values
    bin_cuts : tuple, optional
        (min, max) range for histogram
    nbins : int, optional
        Number of histogram bins
    **kwargs : dict
        Additional parameters passed to fit_s2_simple_cb or fit_s2_two_stage:
        - bg_threshold : float, threshold for background detection (default: 0.3)
        - bg_cutoff : float, upper limit for background fit (default: 1.0)
        - n_sigma : float, sigmas above background for signal region (default: 2.5)
        - upper_limit : float, upper limit for signal fit (default: 5.0)
        
    Returns
    -------
    dict
        Fit results (same structure as fit_s2_simple_cb or fit_s2_two_stage)
        with additional key:
        - 'method': str, 'simple' or 'two_stage'
            
    Notes
    -----
    This is the recommended top-level function for S2 area fitting.
    It handles all cases automatically while providing detailed diagnostics.
    """
    # Detect if background subtraction is needed
    bg_threshold = kwargs.pop('bg_threshold', 0.3)
    detection = detect_background_peak(data, bin_cuts, nbins, bg_threshold)
    
    print(f"  Background detection: {detection['bg_fraction']*100:.1f}% of counts in low region")
    print(f"  Recommendation: {detection['recommendation']}")
    
    if detection['recommendation'] == 'two_stage':
        print("  Using two-stage fitting with background subtraction...")
        result = fit_s2_two_stage(data, bin_cuts, nbins, **kwargs)
    else:
        print("  Using simple Crystal Ball fitting...")
        result = fit_s2_simple_cb(data, bin_cuts, nbins)
    
    return result


# --------------------------#
# Set-level fitting wrapper
# --------------------------#
def fit_set_s2(s2: S2Areas,
               bin_cuts: tuple[float, float] = (0, 10),
               nbins: int = 100,
               flag_plot: bool = False,
               **kwargs) -> S2Areas:
    """
    Fit S2Areas distribution with automatic method selection.
    
    Automatically detects background and uses appropriate fitting method
    (simple Crystal Ball or two-stage with background subtraction).

    Args:
        s2: S2Areas object with raw areas
        bin_cuts: (min, max) for histogram
        nbins: number of bins
        exclude_index: DEPRECATED - kept for compatibility, not used
        flag_plot: if True, show histogram and fit
        **kwargs: Additional parameters for fitting (bg_cutoff, n_sigma, etc.)

    Returns:
        New S2Areas with fit results populated
    """
    # Check if data exists in range
    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        return replace(s2, fit_success=False)

    try:
        # Use automatic fitting with intelligent method selection
        result = fit_s2_area_auto(s2.areas, bin_cuts=bin_cuts, nbins=nbins, **kwargs)
        
        if flag_plot:
            from RaTag.plotting import plot_s2_fit_result
            plot_s2_fit_result(result, s2.areas, 
                             set_name=s2.source_dir.name if hasattr(s2, 'source_dir') else '')
        
        return replace(s2,
                      mean=result['peak_position'],
                      sigma=result['sigma'],
                      ci95=1.96 * result['peak_stderr'] if result['peak_stderr'] else 0.0,
                      fit_result=result,  # Store full result dict
                      fit_success=True)
    
    except Exception as e:
        # Print full traceback and helpful diagnostics for debugging
        import traceback
        print(f"  Warning: Fit failed for {s2.source_dir.name}: {e}")
        try:
            print(f"  Debug: s2.areas type={type(s2.areas)}, shape={getattr(s2.areas,'shape',None)}, dtype={getattr(s2.areas,'dtype',None)}")
            area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
            print(f"  Debug: filtered area_vec shape={getattr(area_vec,'shape',None)}, dtype={getattr(area_vec,'dtype',None)}")
            print(f"  Debug: bin_cuts={bin_cuts}, nbins={nbins}")
        except Exception:
            print("  Debug: failed to print area diagnostics")
        traceback.print_exc()
        return replace(s2, fit_success=False)


# -------------------------------------------------
# Run-level Gaussian fit
# -------------------------------------------------
def fit_run_s2(areas: Dict[str, S2Areas], fit_config: FitConfig = FitConfig(), 
               flag_plot: bool = False) -> Dict[str, S2Areas]:
    """
    Apply Gaussian fits across all sets in a run.

    Args:
        areas: Dict of {set_id: S2Areas} with raw areas.
        fit_config: FitConfig with fitting parameters.
        flag_plot: If True, plot histogram and fit for each set.

    Returns:
        Dict of {set_id: S2Areas} with fit results.
    """
    return {sid: fit_set_s2(s2, bin_cuts=fit_config.bin_cuts,
                               nbins=fit_config.nbins,
                               exclude_index=fit_config.exclude_index,
                               flag_plot=flag_plot)
                               for sid, s2 in areas.items() }
                            


# -------------------------------------------------
#  MULTI-ISOTOPE DATAFRAME FITTING
# -------------------------------------------------

def fit_multiiso_s2(df: pd.DataFrame,
                    isotope: str,
                    bin_cuts: tuple[float, float] = (0, 10),
                    nbins: int = 100,
                    **fit_kwargs) -> dict:
    """
    Fit S2 area distribution for a single isotope from multi-isotope DataFrame.
    
    Uses automatic method selection (simple vs two-stage) based on background detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Multi-isotope DataFrame with 's2_areas' column
    isotope : str
        Isotope name to filter and fit
    bin_cuts : tuple, optional
        (min, max) range for histogram
    nbins : int, optional
        Number of bins
    **fit_kwargs : dict
        Additional arguments for fit_s2_area_auto (bg_cutoff, n_sigma, etc.)
        
    Returns
    -------
    dict
        Fit results with keys matching fit_s2_area_auto output, plus:
        - 'isotope': str, the isotope name
        - 'n_events': int, number of events used
        
    Raises
    ------
    ValueError
        If no data found for isotope or fit fails
    """
    # Filter to isotope
    iso_data = df[df['isotope'] == isotope]['s2_areas'].dropna()
    
    if len(iso_data) == 0:
        raise ValueError(f"No data found for isotope {isotope}")
    
    print(f"  Fitting {isotope}: {len(iso_data)} events")
    
    # Use automatic fitting
    result = fit_s2_area_auto(iso_data.values, bin_cuts=bin_cuts, nbins=nbins, **fit_kwargs)
    
    # Add isotope metadata
    result['isotope'] = isotope
    result['n_events'] = len(iso_data)
    
    return result


