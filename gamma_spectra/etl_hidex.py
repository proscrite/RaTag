import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Dict

# --- PHYSICS CONSTANTS (Seconds) ---
LAMBDA_RA = np.log(2) / (3.6319 * 24 * 3600)  
LAMBDA_PB = np.log(2) / (10.64 / 24.0 * 3600) 

# --- PURE MATH & PHYSICS FUNCTIONS ---
def gaussian_linear(x: np.ndarray, a: float, mu: float, 
                    sigma: float, b: float, c: float) -> np.ndarray:
    """A Gaussian peak plus a linear background."""
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + b * x + c

def fit_gamma_peak(x: np.ndarray, y: np.ndarray) -> Tuple[tuple, tuple, float]:
    """Fits a Gaussian + linear background to the spectrum and returns the parameters, covariance matrix, and R²."""
    p0 = [float(np.max(y)), 220.0, 15.0, 0.0, float(np.min(y))]
    try:
        popt, pcov = curve_fit(gaussian_linear, x, y, p0=p0, maxfev=10000)
        ss_res = np.sum((y - gaussian_linear(x, *popt)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        # Return popt and pcov as ndarray-like objects so downstream
        # functions (which index pcov as a 2D array) work correctly.
        return popt, pcov, r_sq
    except RuntimeError:
        # On failure return zeroed parameters, a zero covariance matrix and
        # a low R² sentinel.
        return np.zeros(5, dtype=float), np.zeros((5, 5), dtype=float), 0.0

def bateman_count_rate(t: np.ndarray, n_ra_0: float, n_pb_0: float, e_ph: float = 1.0) -> np.ndarray:
    """Calculates the count rate for a Bateman equation with Ra and Pb populations."""
    
    activity_ra = LAMBDA_RA * n_ra_0 * np.exp(-LAMBDA_RA * t)

    activity_pb = (LAMBDA_PB * n_ra_0 * (LAMBDA_RA / (LAMBDA_PB - LAMBDA_RA)) * (np.exp(-LAMBDA_RA * t) - np.exp(-LAMBDA_PB * t))) + \
                  (LAMBDA_PB * n_pb_0 * np.exp(-LAMBDA_PB * t))
    
    return e_ph * (0.041 * activity_ra + 0.436 * activity_pb)

def fit_initial_populations(times: np.ndarray, count_rates: np.ndarray, rate_errors: np.ndarray) -> dict:
    """Fits the initial Ra and Pb populations from the count rate data using the Bateman model."""
    
    p0 = [np.max(count_rates) / (LAMBDA_RA * 0.041), 0.0]
   
    popt, pcov = curve_fit(bateman_count_rate, times, count_rates, 
                           p0=p0, sigma=rate_errors, absolute_sigma=True,
                            bounds=(0, [np.inf, np.inf]))
    
    return {
        'ra224_atoms_t0': popt[0], 
        'ra224_atoms_t0_err': np.sqrt(pcov[0][0]), 
        'pb212_atoms_t0': popt[1], 
        'pb212_atoms_t0_err': np.sqrt(pcov[1][1])
    }

def backcalculate_accumulation(n_ra_fit: float, n_pb_fit: float, delay_seconds: float, acc_seconds: float) -> dict:
    """Back-calculates the Ra and Pb populations at the end of the accumulation period, as well as saturation metrics."""
    n_ra_acc = n_ra_fit * np.exp(LAMBDA_RA * delay_seconds)
    pb_from_ra_during_delay = n_ra_acc * (LAMBDA_RA / (LAMBDA_PB - LAMBDA_RA)) * (np.exp(-LAMBDA_RA * delay_seconds) - np.exp(-LAMBDA_PB * delay_seconds))
    n_pb_acc = (n_pb_fit - pb_from_ra_during_delay) * np.exp(LAMBDA_PB * delay_seconds)
    max_ra_capacity = n_ra_acc / (1 - np.exp(-LAMBDA_RA * acc_seconds))
    return {
        'n_ra_end_acc': n_ra_acc, 
        'n_pb_end_acc': n_pb_acc,
        'ratio_ra_to_pb': n_ra_acc / max(n_pb_acc, 1.0),
        'max_ra_capacity': max_ra_capacity, 'saturation_pct': (n_ra_acc / max_ra_capacity) * 100
    }

def parse_timestamps(datetime_series: pd.Series) -> np.ndarray:
    """Takes a series of datetime strings and returns relative seconds."""
    datetimes = pd.to_datetime(datetime_series)
    return (datetimes - datetimes.iloc[0]).dt.total_seconds().values

def calculate_net_counts_error(pcov, a, sigma):
    """Calculates the error in net counts (area) using the covariance matrix from the fit."""
    var_a = pcov[0, 0]
    var_sigma = pcov[2, 2]
    cov_a_sigma = pcov[0, 2]
    
    # 2. Calculate Net Counts (Area)
    # Use abs(sigma) because optimization sometimes flips the sign of sigma, which is mathematically fine but breaks area.
    sigma_abs = np.abs(sigma)
    net_counts = a * sigma_abs * np.sqrt(2 * np.pi)
    
    # 3. Rigorous Error Propagation
    partial_a = sigma_abs * np.sqrt(2 * np.pi)
    partial_sigma = a * np.sqrt(2 * np.pi)
    
    net_counts_var = (partial_a**2 * var_a) + (partial_sigma**2 * var_sigma) + (2 * partial_a * partial_sigma * cov_a_sigma)
    net_counts_error = np.sqrt(net_counts_var) if net_counts_var > 0 else 0.0
    return net_counts_error

# --- TRANSFORM  ---
def transform_spectra_to_rates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw DataFrame (with the 'Spectrum' array column) and applies 
    the gamma peak fitting algorithm across all rows.
    """
    def apply_fit(row: pd.Series) -> pd.Series:
        spectrum = row['Spectrum']
        # Safety catch for entirely empty spectra
        if np.any(np.isnan(spectrum)):
            return pd.Series({'A': 0.0, 'mu': 0.0, 'sigma': 0.0, 'b': 0.0, 'c': 0.0, 'net_counts': 0.0, 'net_counts_error': 0.0, 'rate_cps': 0.0, 'rate_cps_error': 0.0, 'R_sq': 0.0})
        
        x = np.arange(len(spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, spectrum)
        
        a, mu, sigma, b, c = popt
        
        net_counts = a * sigma * np.sqrt(2 * np.pi)
        net_counts_error = calculate_net_counts_error(pcov, a, sigma)
        
        # Calculate count rate (Counts Per Second)
        count_time = float(row.get('CountTime_sec', 1.0))
        if count_time <= 0:
            count_time = 1.0  # Prevent division by zero just in case
            
        rate_cps = net_counts / count_time
        rate_cps_error = net_counts_error / count_time
        
        return pd.Series({
            'A': a,
            'mu': mu,
            'sigma': sigma,
            'b': b,
            'c': c,
            'net_counts': net_counts,
            'net_counts_error': net_counts_error,
            'rate_cps': rate_cps,
            'rate_cps_error': rate_cps_error,
            'R_sq': r_sq,
        })

    fits_df = raw_df.apply(apply_fit, axis=1)
    
    # Concatenate the new fit results with the original metadata (dropping the raw arrays to save memory)
    return pd.concat([raw_df.drop(columns=['Spectrum']), fits_df], axis=1)
