# RaTag/fitting.py
import warnings

import numpy as np
import lmfit
from dataclasses import replace
from typing import Dict, Any, Tuple
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


def v_crystalball_right(x, N, beta, m, x0, sigma):
    """Crystal Ball function with RIGHT tail for ionization signals."""
    absb = max(np.abs(beta), 1e-12) # Protect zero division
    z = (x - x0) / sigma
    gauss = np.exp(-0.5 * z**2)
    A_tail = (m / absb)**m * np.exp(-0.5 * absb**2)
    B = m / absb - absb
    denom_safe = np.maximum(B + z, 1e-12)
    tail = A_tail / (denom_safe)**m
    return N * np.where(z < absb, gauss, tail)

def _find_dynamic_lower_bound(cbins: np.ndarray, counts: np.ndarray, max_lower_bound: float) -> float:
    """Finds the 'valley' between the low-energy noise peak and the S2 signal peak."""
    
    # Smooth lightly to prevent false minima from statistical noise
    n_smooth = np.convolve(counts, np.ones(2)/2, mode='same')
    
    # Find local minima (invert the array to find peaks of the negative)
    minima_indices, _ = find_peaks(-n_smooth)
    
    if len(minima_indices) > 0:
        
        lower_bound = cbins[minima_indices[0]]
        if lower_bound < max_lower_bound:  # Guardrail against unphysical excessive lower bounds
            return lower_bound
    
    return 0.0 # Safe fallback if no distinct valley exists

def compute_fit_ci(param: lmfit.Parameter, bin_width: float, confidence_level: float = 1.96) -> float:
    """
    Computes a robust Confidence Interval using the quadrature sum of the 
    statistical fit error and the histogram binning resolution.
    """
    stderr = param.stderr or 0.0
    binning_err = bin_width / np.sqrt(12.0)
    return confidence_level * np.sqrt(stderr**2 + binning_err**2)

def fit_s2_crystalball(data: np.ndarray, 
                       bin_cuts: Tuple[float, float] = (0, 15), 
                       nbins: int = 100,
                       max_lower_bound: float = 1.5,
                       smooth: int = 3) -> Dict[str, Any]:
    """
    Fits a right-tailed Crystal Ball to the S2 signal.
    Dynamically finds the noise valley to exclude the low-energy peak.
    """
    # 1. Build Histogram
    filtered = data[(data >= bin_cuts[0]) & (data <= bin_cuts[1])]
    counts, bins = np.histogram(filtered, bins=nbins, range=bin_cuts)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    
    if len(filtered) < 10:
        raise ValueError("Not enough data to fit.")

    # 2. Find the Valley (Dynamic Thresholding)
    lower_bound = _find_dynamic_lower_bound(cbins, counts, max_lower_bound)

    # 3. Mask data for fitting (Only fit the signal region!)
    fit_mask = cbins >= lower_bound
    cbins_fit = cbins[fit_mask]
    counts_fit = counts[fit_mask]
    counts_smooth = np.convolve(counts_fit, np.ones(smooth)/smooth, mode='same')
    
    # 4. Initial Guesses
    peak_idx = np.argmax(counts_smooth)

    guess_x0 = cbins_fit[peak_idx]
    guess_N = counts_smooth[peak_idx]
    # print(f"  Initial guess: N={guess_N:.1f}, x0={guess_x0:.2f}, lower_bound={lower_bound:.2f}")
    
    # 5. Execute Fit
    model = lmfit.Model(v_crystalball_right, prefix='sig_')
    params = model.make_params(sig_N=guess_N, sig_x0=guess_x0,
                               sig_sigma=0.5, sig_beta=1.0, sig_m=2.0)
    
    # Constrain the peak to remain in the signal region
    params['sig_N'].set(min=0.0)                                # Amplitude must be positive
    params['sig_x0'].set(min=lower_bound, max=bin_cuts[1])      # Peak must be in the signal region
    params['sig_sigma'].set(min=0.05, max=15.0)                  # Width must be positive
    params['sig_beta'].set(min=0.1, max=10.0)                   # Tail onset must be positive
    params['sig_m'].set(min=1.001, max=50.0)                    # Tail power strictly > 1 (Prevents NaN)
    
    result = model.fit(counts_smooth, params, x=cbins_fit)
    print(f"  Fit success: {result.success}, χ²/DOF: {result.redchi:.2f}")
    # 6. Correct, Un-inflated CI Calculation (Stat Error + Binning Error)
    
    bin_width = cbins[1] - cbins[0]
    ci95 = compute_fit_ci(result.params['sig_x0'], bin_width)
    
    return {
        'peak_position': result.params['sig_x0'].value,
        'sigma': result.params['sig_sigma'].value,
        'ci95': ci95,
        'lower_bound': lower_bound,
        'chi2': result.chisqr,
        'redchi': result.redchi,
        'result': result
    }

# ----------------------------------------------------------------
# THGEM S1 cut optimization: 2 components in S1 area distribution
# ----------------------------------------------------------------

def optimize_s1_cut(s1_areas, fallback_cut=0.06):
    """
    Fits a 2-component GMM to the S1 areas and returns the optimal Bayesian cut
    where the probability of being a 'Hole' event overtakes a 'Web' event.
    """
    # Exclude absolute zero-noise and extreme outliers for a clean fit
    clean_s1 = s1_areas[(s1_areas > 0.005) & (s1_areas < 0.15)]
    
    # Safe fallback if a file is almost empty
    if len(clean_s1) < 50:
        return fallback_cut, None

    X = clean_s1.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gmm.fit(X)

    means = gmm.means_.flatten()
    order = np.argsort(means)
    web_idx, hole_idx = order[0], order[1]

    # Calculate probabilities across the physical range
    x_smooth = np.linspace(0.005, 0.15, 1000).reshape(-1, 1)
    probs = gmm.predict_proba(x_smooth)
    
    prob_web = probs[:, web_idx]
    prob_hole = probs[:, hole_idx]

    # Find where P(Hole) > P(Web)
    crossings = np.where(prob_hole > prob_web)[0]
    if len(crossings) > 0:
        optimal_cut = x_smooth[crossings[0]][0]
    else:
        optimal_cut = fallback_cut

    # Package the model data for the plotting function
    model_data = {
        'gmm': gmm,
        'clean_s1': clean_s1,
        'web_idx': web_idx,
        'hole_idx': hole_idx,
        'x_smooth': x_smooth.flatten()
    }
    
    return optimal_cut, model_data