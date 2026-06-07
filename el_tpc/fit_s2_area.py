# RaTag/fitting.py
import numpy as np
import lmfit
from dataclasses import replace
from typing import Dict, Any, Tuple
from scipy.signal import find_peaks


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

def _find_dynamic_lower_bound(cbins: np.ndarray, counts: np.ndarray, search_max: float = 2.0) -> float:
    """Finds the 'valley' between the low-energy noise peak and the S2 signal peak."""
    mask = cbins <= search_max
    if not np.any(mask):
        return 0.5
        
    c_search = cbins[mask]
    n_search = counts[mask]
    
    # Smooth lightly to prevent false minima from statistical noise
    n_smooth = np.convolve(n_search, np.ones(3)/3, mode='same')
    
    # Find local minima (invert the array to find peaks of the negative)
    minima_indices, _ = find_peaks(-n_smooth)
    
    if len(minima_indices) > 0:
        return c_search[minima_indices[0]]
        
    return 0.5 # Safe fallback if no distinct valley exists

def compute_fit_ci(param: lmfit.Parameter, bin_width: float, confidence_level: float = 1.96) -> float:
    """
    Computes a robust Confidence Interval using the quadrature sum of the 
    statistical fit error and the histogram binning resolution.
    """
    stderr = param.stderr or 0.0
    binning_err = bin_width / np.sqrt(12.0)
    return confidence_level * np.sqrt(stderr**2 + binning_err**2)

def fit_s2_crystalball(data: np.ndarray, 
                       bin_cuts: Tuple[float, float] = (0, 10), 
                       nbins: int = 100) -> Dict[str, Any]:
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
    lower_bound = _find_dynamic_lower_bound(cbins, counts)
    
    # 3. Mask data for fitting (Only fit the signal region!)
    fit_mask = cbins >= lower_bound
    cbins_fit = cbins[fit_mask]
    counts_fit = counts[fit_mask]
    
    # 4. Initial Guesses
    peak_idx = np.argmax(counts_fit)
    guess_x0 = cbins_fit[peak_idx]
    guess_N = counts_fit[peak_idx]
    # print(f"  Initial guess: N={guess_N:.1f}, x0={guess_x0:.2f}, lower_bound={lower_bound:.2f}")
    
    # 5. Execute Fit
    model = lmfit.Model(v_crystalball_right, prefix='sig_')
    params = model.make_params(sig_N=guess_N, sig_x0=guess_x0,
                               sig_sigma=0.5, sig_beta=1.0, sig_m=2.0)
    
    # Constrain the peak to remain in the signal region
    params['sig_N'].set(min=0.0)                                # Amplitude must be positive
    params['sig_x0'].set(min=lower_bound, max=bin_cuts[1])      # Peak must be in the signal region
    params['sig_sigma'].set(min=0.05, max=5.0)                  # Width must be positive
    params['sig_beta'].set(min=0.1, max=10.0)                   # Tail onset must be positive
    params['sig_m'].set(min=1.001, max=50.0)                    # Tail power strictly > 1 (Prevents NaN)
    
    result = model.fit(counts_fit, params, x=cbins_fit)
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