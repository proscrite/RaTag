import numpy as np
import lmfit
from lmfit.model import ModelResult
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

from RaTag.core.fitting import v_crystalball_left


def beta_continuum(x, A_beta, lambda_beta, E_min):
        """Simplified beta spectrum: exponential decay."""
        return A_beta * np.exp(-lambda_beta * (x - E_min))
    
@dataclass(frozen=True)
class EnergyCalibration:
    a: float
    b: float
    c: Optional[float] = None
    order: int = 1
    
    def apply(self, E_SCA: np.ndarray) -> np.ndarray:
        if self.order == 2 and self.c is not None:
            return self.a * E_SCA**2 + self.b * E_SCA + self.c
        return self.a * E_SCA + self.b
    
    def derivative(self, E_SCA: np.ndarray) -> np.ndarray:
        if self.order == 2:
            return 2 * self.a * E_SCA + self.b
        return np.full_like(E_SCA, self.a)

def select_roi(energies: np.ndarray, E_min: float, E_max: float, bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Pure array slicing and histogramming."""
    mask = (energies >= E_min) & (energies <= E_max)
    roi_energies = energies[mask]
    counts, bin_edges = np.histogram(roi_energies, bins=bins, range=(E_min, E_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def apply_calibration(energies_SCA: np.ndarray, a: float, b: float, c: Optional[float], order: int) -> np.ndarray:
    """Helper to apply flat polynomial constants to a raw array."""
    if order == 2 and c is not None:
        return a * energies_SCA**2 + b * energies_SCA + c
    return a * energies_SCA + b

def _setup_crystalball_params(params: lmfit.Parameters, prefix: str, 
                              N_value: float, x0_value: float, 
                              beta_value: float, m_value: float, sigma_value: float,
                              N_bounds: Optional[Tuple[float, float]] = None,
                              x0_bounds: Optional[Tuple[float, float]] = None,
                              vary_shape: bool = True, N_expr: Optional[str] = None) -> None:
    """Configure Crystal Ball parameters with optional constraints and bounds."""
    
    if N_expr is None:
        N_min, N_max = N_bounds if N_bounds else (N_value*0.1, N_value*10)
        params[f'{prefix}N'].set(value=N_value, min=N_min, max=N_max)
    else:
        params[f'{prefix}N'].set(expr=N_expr)
    
    # x0 can vary if bounds provided, even when vary_shape=False
    if x0_bounds:
        x0_min, x0_max = x0_bounds
        params[f'{prefix}x0'].set(value=x0_value, min=x0_min, max=x0_max, vary=True)
    else:
        params[f'{prefix}x0'].set(value=x0_value, vary=vary_shape)
    
    params[f'{prefix}beta'].set(value=beta_value, min=-5, max=-0.1, vary=vary_shape)
    params[f'{prefix}m'].set(value=m_value, min=1.0, max=10, vary=vary_shape)
    params[f'{prefix}sigma'].set(value=sigma_value, min=0.04, max=0.1, vary=vary_shape)


def _select_fitting_window(energies: np.ndarray, counts: np.ndarray,
                          center: float, window: Optional[Tuple[float, float]] = None,
                          default_width: float = 0.3) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Select energy window around peak and extract data. Returns (x, y, E_min, E_max)."""
    
    if window is None:
        E_min = center - default_width
        E_max = center + default_width
    else:
        E_min, E_max = window
    
    mask = (energies >= E_min) & (energies <= E_max)
    x = energies[mask]
    y = counts[mask].astype(float)
    
    if len(x) < 10:
        raise ValueError(f"Insufficient data in window [{E_min:.2f}, {E_max:.2f}] MeV")
    
    return x, y, E_min, E_max

def _extract_single_fit(composite_result: ModelResult, source_prefix: str, x_data: np.ndarray, y_data: np.ndarray) -> ModelResult:
    """Extracts a single peak from a composite fit and formats it with the standard 'cb_' prefix."""
    single_model = lmfit.Model(v_crystalball_left, prefix='cb_')
    params = single_model.make_params()
    for param_name in ['N', 'beta', 'm', 'x0', 'sigma']:
        val = composite_result.params[f'{source_prefix}{param_name}'].value
        params.add(f'cb_{param_name}', value=val, vary=False)
    return single_model.fit(y_data, params=params, x=x_data)

def fit_single_crystalball(energies: np.ndarray, counts: np.ndarray,
                          peak_position: float, energy_window: Optional[Tuple[float, float]] = None,
                          beta_init: float = -1.5, m_init: float = 2.0,
                          sigma_init: Optional[float] = None) -> ModelResult:
    
    """Fits a single Crystal Ball function to the provided energy spectrum."""

    x, y, E_min, E_max = _select_fitting_window(energies, counts, peak_position,
                                                 energy_window, default_width=0.3)
    
    model = lmfit.Model(v_crystalball_left, prefix='cb_')
    params = model.make_params()
    
    sigma_val = sigma_init if sigma_init else 0.05
    _setup_crystalball_params(params, 'cb_', N_value=y.max(), x0_value=peak_position,
                             beta_value=beta_init, m_value=m_init, sigma_value=sigma_val,
                             x0_bounds=(E_min, E_max), vary_shape=True)
    result = model.fit(y, params=params, x=x)
    return result


def fit_po212_alpha_beta(energies: np.ndarray, counts: np.ndarray,
                         alpha_position: float, energy_window: Optional[Tuple[float, float]] = None,
                         beta_init: float = -1.5, m_init: float = 2.0, sigma_init: Optional[float] = None) -> ModelResult:
    """Fits a combined model of Po-212 alpha peak (Crystal Ball) and beta continuum to the energy spectrum."""

    x, y, E_min, E_max = _select_fitting_window(energies, counts, alpha_position, energy_window)
    y_peak = y.max()
    
    alpha_model = lmfit.Model(v_crystalball_left, prefix='cb_')
    beta_model = lmfit.Model(beta_continuum, prefix='beta_', independent_vars=['x', 'E_min'])
    model = alpha_model + beta_model
    
    params = model.make_params()
    
    sigma_val = sigma_init if sigma_init else 0.15
    _setup_crystalball_params(params, 'cb_', N_value=y_peak, x0_value=alpha_position,
                             beta_value=beta_init, m_value=m_init, sigma_value=sigma_val,
                             x0_bounds=(E_min, E_max), vary_shape=True)
    
    params['beta_A_beta'].set(value=y_peak*0.3, min=0.0, max=y_peak*1.0)
    params['beta_lambda_beta'].set(value=1.0, min=0.1, max=5.0)
    
    result = model.fit(y, params=params, x=x, E_min=E_min)
    return result

def fit_multi_crystalball_progressive(energies_roi: np.ndarray,
                                      counts_roi: np.ndarray,
                                      peak_definitions: List[Dict]) -> Dict[str, ModelResult]:
    """Fits multiple peaks progressively."""
    individual_fits = {}
    
    for peak_def in peak_definitions:
        name = peak_def['name']
        fit_func = fit_po212_alpha_beta if name == 'Po212' else fit_single_crystalball
        
        try:
            result = fit_func(energies=energies_roi, counts=counts_roi,
                              energy_window=peak_def['window'],
                              sigma_init=peak_def.get('sigma_init', None),
                              **{'alpha_position' if name == 'Po212' else 'peak_position': peak_def['position']})
            individual_fits[name] = result
        except Exception as e:
            print(f"  ✗ {name}: Fit failed - {e}")
            
    if not individual_fits:
        raise RuntimeError("All individual fits failed")
    return individual_fits

def derive_energy_calibration(fit_results: Dict[str, ModelResult],
                              peak_definitions: List[Dict],
                              order: int = 2) -> EnergyCalibration:
    """Derives polynomial calibration coefficients."""
    literature_energies = {p['name']: p['ref_energy'] for p in peak_definitions}
    E_SCA_list, E_true_list = [], []
    
    for name, result in fit_results.items():
        if name in literature_energies:
            E_SCA_list.append(result.params['cb_x0'].value)
            E_true_list.append(literature_energies[name])
            
    coeffs = np.polyfit(E_SCA_list, E_true_list, deg=order)
    
    if order == 2:
        return EnergyCalibration(a=coeffs[0], b=coeffs[1], c=coeffs[2], order=2)
    return EnergyCalibration(a=coeffs[0], b=coeffs[1], c=None, order=1)

def refine_overlapping_pair(energies: np.ndarray, counts: np.ndarray, 
                            prelim1: ModelResult, prelim2: ModelResult) -> tuple[ModelResult, ModelResult]:
    """
    Takes preliminary independent fits and refines them via simultaneous fitting.
    Uses the preliminary parameters to establish exact initial guesses and bounds.
    """
    x0_1, sig_1 = prelim1.params['cb_x0'].value, prelim1.params['cb_sigma'].value
    x0_2, sig_2 = prelim2.params['cb_x0'].value, prelim2.params['cb_sigma'].value
    
    w_min = min(x0_1 - 3*sig_1, x0_2 - 3*sig_2)
    w_max = max(x0_1 + 3*sig_1, x0_2 + 3*sig_2)
    
    mask = (energies >= w_min) & (energies <= w_max)
    x, y = energies[mask], counts[mask].astype(float)
    
    m1 = lmfit.Model(v_crystalball_left, prefix='p1_')
    m2 = lmfit.Model(v_crystalball_left, prefix='p2_')
    model = m1 + m2
    params = model.make_params()
    
    _setup_crystalball_params(params, 'p1_', N_value=prelim1.params['cb_N'].value, 
                              x0_value=x0_1, beta_value=prelim1.params['cb_beta'].value, 
                              m_value=prelim1.params['cb_m'].value, sigma_value=sig_1, vary_shape=True)
                              
    _setup_crystalball_params(params, 'p2_', N_value=prelim2.params['cb_N'].value, 
                              x0_value=x0_2, beta_value=prelim2.params['cb_beta'].value, 
                              m_value=prelim2.params['cb_m'].value, sigma_value=sig_2, vary_shape=True)

    params['p1_beta'].expr = 'p2_beta'
    params['p1_m'].expr = 'p2_m'
    params['p1_sigma'].expr = 'p2_sigma'

    comp_res = model.fit(y, params=params, x=x)
    return _extract_single_fit(comp_res, 'p1_', x, y), _extract_single_fit(comp_res, 'p2_', x, y)


def derive_isotope_ranges(fit_results: dict[str, ModelResult],
                          calibration: EnergyCalibration,
                          n_sigma: Union[float, dict] = 2.0,
                          crossover_V: Optional[float] = None,
                          overlap_pair: tuple[str, str] = ('Th228', 'Ra224')) -> tuple[dict, dict, float]:
    """Computes ranges in V and E scales. Applies an exact crossover boundary if provided."""
    ranges_V, ranges_E, resolutions = {}, {}, []
    
    for name, fit in fit_results.items():
        x0_SCA = fit.params['cb_x0'].value
        sigma_SCA = fit.params['cb_sigma'].value

        if isinstance(n_sigma, dict):
            sig_left, sig_right = n_sigma.get(name, n_sigma.get('default', (2.0, 2.0)))
        else:
            sig_left = sig_right = float(n_sigma)

        x0_true = calibration.apply(np.array([x0_SCA]))[0]
        sigma_true = abs(calibration.derivative(np.array([x0_SCA]))[0]) * sigma_SCA
        
        resolutions.append((2.355 * sigma_true) / x0_true)
        
        ranges_V[name] = (x0_SCA - (sig_left * sigma_SCA), x0_SCA + (sig_right * sigma_SCA))
        ranges_E[name] = (x0_true - (sig_left * sigma_true), x0_true + (sig_right * sigma_true))
        
    # Apply the mathematical hard-clip if a crossover boundary was provided
    if crossover_V is not None and overlap_pair[0] in ranges_V and overlap_pair[1] in ranges_V:
        iso1, iso2 = overlap_pair
        crossover_E = calibration.apply(np.array([crossover_V]))[0]

        ranges_V[iso1] = (ranges_V[iso1][0], min(ranges_V[iso1][1], crossover_V))
        ranges_V[iso2] = (max(ranges_V[iso2][0], crossover_V), ranges_V[iso2][1])
        
        ranges_E[iso1] = (ranges_E[iso1][0], min(ranges_E[iso1][1], crossover_E))
        ranges_E[iso2] = (max(ranges_E[iso2][0], crossover_E), ranges_E[iso2][1])

    # Clean rounding for JSON
    ranges_V = {k: (round(float(v[0]), 3), round(float(v[1]), 3)) for k, v in ranges_V.items()}
    ranges_E = {k: (round(float(v[0]), 3), round(float(v[1]), 3)) for k, v in ranges_E.items()}
        
    return ranges_V, ranges_E, round(float(np.mean(resolutions)), 4)

def resolve_likelihood_crossover(fit1: ModelResult, fit2: ModelResult) -> float:
    """Finds the instrumental voltage (SCA) where the two fitted PDFs cross over."""
    x0_1 = fit1.params['cb_x0'].value
    x0_2 = fit2.params['cb_x0'].value
    
    # Generate high-resolution grid between the two peaks
    V_grid = np.linspace(min(x0_1, x0_2), max(x0_1, x0_2), 5000)

    P1 = fit1.eval(x=V_grid)
    P2 = fit2.eval(x=V_grid)

    # Locate minimum difference
    diff = np.abs(P1 - P2)
    return float(V_grid[np.argmin(diff)])
