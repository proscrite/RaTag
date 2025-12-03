from typing import Dict, Tuple, Any
from dataclasses import replace
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from lmfit.models import GaussianModel # type: ignore

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
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
#  S1/S2 timing histogram fit
# -------------------------------------------------

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
# Set-level Gaussian fit  --#
# --------------------------#
def fit_set_s2(s2: S2Areas,
               bin_cuts: tuple[float, float] = (0, 4),
               nbins: int = 100,
               exclude_index: int = 1,
               flag_plot: bool = False) -> S2Areas:
    """
    Fit Gaussian to S2Areas distribution of a single set.

    Args:
        s2: S2Areas object with raw areas.
        bin_cuts: (min, max) for histogram.
        nbins: number of bins.
        exclude_index: skip first bins if pedestal leak.
        flag_plot: if True, show histogram and fit.

    Returns:
        New S2Areas with fit results populated.
    """
    # Check if data exists in range
    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        return replace(s2, fit_success=False)

    # Use shared Gaussian fitting function
    try:
        mean, sigma, ci95, cbins, n, result = fit_gaussian_to_histogram(data=s2.areas,
                                                                         bin_cuts=bin_cuts,
                                                                         nbins=nbins,
                                                                         exclude_index=exclude_index)
    except Exception as e:
        print(f"Warning: Gaussian fit failed for {s2.source_dir.name}: {e}")
        return replace(s2, fit_success=False)

    if flag_plot:
        from lmfit.models import GaussianModel
        n_full, bins, _ = plt.hist(area_vec, bins=nbins, alpha=0.6, color='g', label="Data")
        model = GaussianModel()
        plt.plot(cbins, model.eval(x=cbins, params=result.params), 'r--', label='fit')
        plt.gca().set(xlabel = 'Area (mV·ns)', ylabel = 'Counts')
        plt.legend()

    return replace(s2,
                   mean=mean,
                   sigma=sigma,
                   ci95=ci95,
                   fit_result=result,
                   fit_success=result.success)


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
#  X-RAY AREA FITTING
# -------------------------------------------------


