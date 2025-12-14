"""
Alpha spectrum fitting and isotope range derivation.

This module provides pure functions for:
1. Multi-peak Crystal Ball fitting to alpha energy spectra
2. Automatic derivation of isotope energy ranges from fits
3. Purity-based windowing for clean isotope separation (Option C)

Data flow:
    energy_maps/*.bin → load_energies → fit_spectrum → derive_ranges → isotope_ranges

Structure:
- Pure functions (no classes, no state mutation)
- Composable with functools.partial and pipe
- Type hints for clarity
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import lmfit
from lmfit.model import ModelResult

from RaTag.alphas.energy_map_reader import load_energy_index


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class EnergyCalibration:
    """
    Energy calibration: Linear or Quadratic
    
    Linear: E_true = a * E_SCA + b
    Quadratic: E_true = a * E_SCA² + b * E_SCA + c
    
    Attributes:
        a: Quadratic coefficient [MeV / mV²] (or slope for linear)
        b: Linear coefficient [MeV / mV] (or intercept for linear)
        c: Intercept [MeV] (only for quadratic, None for linear)
        anchors: Dict of isotope_name -> (E_SCA [mV], E_true_literature [MeV])
        residuals: Array of residuals [MeV] for each anchor point (E_true - E_pred)
        order: 1 for linear, 2 for quadratic
    """
    a: float
    b: float
    c: Optional[float] = None
    anchors: Optional[Dict[str, Tuple[float, float]]] = None
    residuals: Optional[np.ndarray] = None
    order: int = 1
    
    def apply(self, E_SCA: np.ndarray) -> np.ndarray:
        """Apply calibration: E_true [MeV] = f(E_SCA [mV])"""
        if self.order == 2 and self.c is not None:
            return self.a * E_SCA**2 + self.b * E_SCA + self.c
        else:
            return self.a * E_SCA + self.b
    
    def inverse(self, E_true: np.ndarray) -> np.ndarray:
        """Inverse calibration: E_SCA [mV] = f^(-1)(E_true [MeV])"""
        if self.order == 2 and self.c is not None:
            # Solve quadratic: a*x² + b*x + (c - E_true) = 0
            # x = (-b + sqrt(b² - 4a(c - E_true))) / (2a)
            discriminant = self.b**2 - 4*self.a*(self.c - E_true)
            return (-self.b + np.sqrt(discriminant)) / (2*self.a)
        else:
            return (E_true - self.b) / self.a
    
    def derivative(self, E_SCA: np.ndarray) -> np.ndarray:
        """
        Compute derivative dE_true/dE_SCA at given E_SCA values. Used for propagating uncertainties.
        """
        if self.order == 2:
            # d/dE_SCA (a*E_SCA² + b*E_SCA + c) = 2*a*E_SCA + b
            return 2 * self.a * E_SCA + self.b
        else:
            # d/dE_SCA (a*E_SCA + b) = a
            return np.full_like(E_SCA, self.a)
    
    def __str__(self) -> str:
        if self.order == 2:
            formula = f"E_true [MeV] = {self.a:.6e} * E_SCA² + {self.b:.6f} * E_SCA + {self.c:.6f}"
        else:
            formula = f"E_true [MeV] = {self.a:.6f} * E_SCA + {self.b:.6f}"
        
        anchor_info = f"Anchors: {list(self.anchors.keys())}" if self.anchors else "No anchors"
        
        if self.residuals is not None:
            rms = np.sqrt(np.mean(self.residuals**2))
            residual_info = f"RMS residual: {rms*1000:.1f} keV"
        else:
            residual_info = "No residuals"
        
        return f"{formula}\n{anchor_info}\n{residual_info}"


@dataclass(frozen=True)
class IsotopeRange:
    """
    Immutable isotope energy range with metadata.
    
    Attributes:
        name: Isotope identifier (e.g., "Th228", "Ra224")
        E_min: Lower energy bound [MeV or SCA units]
        E_max: Upper energy bound [MeV or SCA units]
        E_peak: Fitted peak position [MeV or SCA units]
        sigma: Fitted peak width [MeV or SCA units]
        purity: Fraction of energy range with >min_purity (0-1)
                0.0 if no pure region found (fallback to n_sigma range)
    """
    name: str
    E_min: float
    E_max: float
    E_peak: float
    sigma: float
    purity: float = 1.0
    
    def __repr__(self):
        purity_str = f", purity={self.purity:.1%}" if self.purity < 1.0 else ""
        return (f"IsotopeRange({self.name}: [{self.E_min:.1f}, {self.E_max:.1f}] keV, "
                f"peak={self.E_peak:.1f}±{self.sigma:.1f}{purity_str})")


@dataclass(frozen=True)
class SpectrumData:
    """
    Immutable alpha energy spectrum data.
    
    Attributes:
        energies: Array of all alpha energies [keV]
        energy_range: (E_min, E_max) for ROI selection [keV]
        source: Description of data source (e.g., "RUN18/FieldScan_Gate0050_Anode1950")
    """
    energies: np.ndarray
    energy_range: Tuple[float, float]
    source: str = "unknown"
    
    def select_roi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select region of interest and create histogram.
        
        Returns:
            Tuple of (bin_centers, counts) ready for fitting
        """
        E_min, E_max = self.energy_range
        mask = (self.energies >= E_min) & (self.energies <= E_max)
        roi_energies = self.energies[mask]
        
        # Create histogram (120 bins is typical for alpha spectra)
        counts, bin_edges = np.histogram(roi_energies, bins=120, range=(E_min, E_max))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, counts


# ============================================================================
# PURE FUNCTIONS - DATA LOADING
# ============================================================================

def load_spectrum_from_energy_maps(energy_maps_dir: str,
                                   energy_range: Tuple[float, float],
                                   fmt: str = '8b',
                                   scale: float = 0.1) -> SpectrumData:
    """
    Load alpha energy spectrum from binary energy map files.
    
    Args:
        energy_maps_dir: Directory containing energy_map_*.bin files
                        (e.g., "RUN18/energy_maps/FieldScan_Gate0050_Anode1950")
        energy_range: (E_min, E_max) ROI for fitting [keV]
        fmt: Binary format - "8b" or "6b"
        scale: Scale factor for "6b" format [keV/LSB]
        
    Returns:
        SpectrumData with loaded energies
        
    Example:
        >>> spectrum = load_spectrum_from_energy_maps(
        ...     "RUN18/energy_maps/FieldScan_Gate0050_Anode1950",
        ...     energy_range=(4000, 8000)
        ... )
    """
    # Load energies from binary chunks (cached internally by energy_map_reader)
    _, energies = load_energy_index(energy_maps_dir, fmt=fmt, scale=scale)
    
    return SpectrumData(
        energies=energies,
        energy_range=energy_range,
        source=energy_maps_dir
    )


def load_spectrum_from_run(run,
                          energy_range: Tuple[float, float],
                          aggregate: bool = True) -> SpectrumData:
    """
    Load alpha spectrum from Run object (convenience wrapper).
    
    Args:
        run: Run object with energy maps already created
        energy_range: (E_min, E_max) ROI [keV]
        aggregate: If True, concatenate all sets. If False, return first set only.
        
    Returns:
        SpectrumData (aggregated or single-set)
    """
    
    if aggregate:
        all_energies = []
        for set_pmt in run.sets:
            energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
            if energy_maps_dir.exists():
                _, energies = load_energy_index(str(energy_maps_dir), fmt='8b')
                all_energies.append(energies)
        
        if not all_energies:
            raise ValueError("No energy maps found in run")
        
        return SpectrumData(
            energies=np.concatenate(all_energies),
            energy_range=energy_range,
            source=f"{run.run_id}_aggregated"
        )
    else:
        # Use first set
        set_pmt = run.sets[0]
        energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
        return load_spectrum_from_energy_maps(str(energy_maps_dir), energy_range)

# ============================================================================
# PURE FUNCTIONS - CRYSTAL BALL MODEL
# ============================================================================

def v_crystalball(x: np.ndarray, N: float, beta: float, m: float, 
                  x0: float, sigma: float) -> np.ndarray:
    """
    Vectorized Crystal Ball PDF for lmfit.
    
    Crystal Ball function: Gaussian core with power-law tail.
    Used for modeling energy-degraded alpha peaks in detectors.
    
    Args:
        x: Energy values [keV]
        N: Normalization amplitude
        beta: Tail parameter (negative for left tail)
        m: Tail exponent (>1)
        x0: Peak position [keV]
        sigma: Peak width [keV]
        
    Returns:
        Array of PDF values (same shape as x)
    """
    absb = np.abs(beta)
    z = (x - x0) / sigma
    
    # Gaussian core
    gauss = np.exp(-0.5 * z**2)
    
    # Power-law tail
    A_tail = (m / absb)**m * np.exp(-0.5 * absb**2)
    B = m / absb - absb
    tail = A_tail / (B - z)**m
    
    # Piecewise: Gaussian for z > -|beta|, tail for z <= -|beta|
    return N * np.where(z > -absb, gauss, tail)


# ============================================================================
# PURE FUNCTIONS - FITTING HELPERS
# ============================================================================

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
    params[f'{prefix}sigma'].set(value=sigma_value, min=0.01, max=0.3, vary=vary_shape)
# ============================================================================
# PURE FUNCTIONS - SINGLE-PEAK FITTING
# ============================================================================

def fit_single_crystalball(energies: np.ndarray, counts: np.ndarray,
                          peak_position: float, energy_window: Optional[Tuple[float, float]] = None,
                          beta_init: float = -1.5, m_init: float = 2.0,
                          sigma_init: Optional[float] = None) -> ModelResult:
    """
    Fit single Crystal Ball peak to spectrum region.
    
    This is the building block for progressive multi-peak fitting.
    Fits one peak at a time with better convergence than simultaneous fitting.
    
    Args:
        energies: Energy bin centers [mV] (SCA scale)
        counts: Histogram counts
        peak_position: Expected peak position [mV] (SCA scale)
        energy_window: (E_min, E_max) window around peak [mV] (SCA scale)
                      If None, uses ±0.3 mV around peak_position
        beta_init: Initial tail parameter (negative for left tail)
        m_init: Initial tail exponent
        sigma_init: Initial peak width [mV] (SCA scale). If None, uses 0.05 mV
        
    Returns:
        ModelResult for single peak
        
    Example:
        >>> # Fit Th228 peak in SCA scale
        >>> result = fit_single_crystalball(
        ...     energies, counts,
        ...     peak_position=5.423,  # mV (SCA scale)
        ...     energy_window=(5.0, 5.8)
        ... )
    """
    x, y, E_min, E_max = _select_fitting_window(energies, counts, peak_position,
                                                 energy_window, default_width=0.3)
    
    model = lmfit.Model(v_crystalball, prefix='cb_')
    params = model.make_params()
    
    sigma_val = sigma_init if sigma_init else 0.05
    _setup_crystalball_params(params, 'cb_', N_value=y.max(), x0_value=peak_position,
                             beta_value=beta_init, m_value=m_init, sigma_value=sigma_val,
                             x0_bounds=(E_min, E_max), vary_shape=True)
    
    result = model.fit(y, params=params, x=x)
    return result

# ============================================================================
# SPECIAL CASE: Po-212 α+β DECAY
# ============================================================================

def fit_po212_alpha_beta(energies: np.ndarray,
                        counts: np.ndarray,
                        alpha_position: float,
                        energy_window: Optional[Tuple[float, float]] = None,
                        beta_init: float = -1.5,
                        m_init: float = 2.0,
                        sigma_init: Optional[float] = None) -> ModelResult:
    """
    Fit Bi-212 decay spectrum: Po-212 alpha peak + beta continuum.
    
    **IMPORTANT**: This function works in **SCA SCALE** (instrumental units).
    All energy parameters (alpha_position, energy_window, sigma_init) should be in SCA units.
    
    Bi-212 has two decay modes:
    - β⁻ decay (64.06%) → Po-212 → Pb-208 (α, 8.785 MeV true energy, 100%)
    - α decay (35.94%) → Tl-208 (6.051 MeV, 6.090 MeV)
    
    The observed spectrum around Po-212 peak shows:
    - Crystal Ball peak: Po-212 alpha (from Bi-212 beta decay)
    - Exponential background: Bi-212 beta continuum (Qβ = 2.25 MeV)
    
    This is a composite model: α (Crystal Ball) + β (exponential tail).
    
    Args:
        energies: Energy bin centers in SCA SCALE (instrumental units)
        counts: Histogram counts
        alpha_position: Po-212 α peak position in SCA SCALE
                       (e.g., ~7.5 SCA ≈ 8.785 MeV true energy)
        energy_window: (E_min, E_max) fitting window in SCA SCALE
                      If None, uses asymmetric window (alpha_position - 0.3, alpha_position + 0.3)
        beta_init: Initial tail parameter for Crystal Ball
        m_init: Initial tail exponent for Crystal Ball
        sigma_init: Initial peak width in SCA SCALE. If None, uses 0.15
        
    Returns:
        ModelResult with composite α+β model (fitted in SCA scale)
        
    Note:
        The beta continuum extends from low energy to the alpha peak.
        Use a wide asymmetric window to capture both components.
        
    Example:
        >>> # Fit in SCA scale
        >>> result = fit_po212_alpha_beta(
        ...     energies_SCA, counts,
        ...     alpha_position=7.5,  # Position in SCA scale
        ...     energy_window=(6.5, 8.0),  # Window in SCA scale
        ...     sigma_init=0.15  # Width in SCA scale
        ... )
        >>> # Use fitted position for calibration
        >>> E_Po212_SCA = result.params['cb_x0'].value
    """
    x, y, E_min, E_max = _select_fitting_window(
        energies, counts, alpha_position, energy_window
    )
    
    y_peak = y.max()
    
    def beta_continuum(x, A_beta, lambda_beta):
        """Simplified beta spectrum: exponential decay."""
        return A_beta * np.exp(-lambda_beta * (x - E_min))
    
    alpha_model = lmfit.Model(v_crystalball, prefix='cb_')
    beta_model = lmfit.Model(beta_continuum, prefix='beta_')
    model = alpha_model + beta_model
    
    params = model.make_params()
    
    sigma_val = sigma_init if sigma_init else 0.15
    _setup_crystalball_params(params, 'cb_', N_value=y_peak, x0_value=alpha_position,
                             beta_value=beta_init, m_value=m_init, sigma_value=sigma_val,
                             x0_bounds=(E_min, E_max), vary_shape=True)
    
    params['beta_A_beta'].set(value=y_peak*0.3, min=0.0, max=y_peak*1.0)
    params['beta_lambda_beta'].set(value=1.0, min=0.1, max=5.0)
    
    result = model.fit(y, params=params, x=x)
    return result


# ============================================================================
# PURE FUNCTIONS - MULTI-PEAK FITTING (Progressive approach)
# ============================================================================

def fit_multi_crystalball_progressive(energies: np.ndarray,
                                     counts: np.ndarray,
                                     peak_definitions: List[Dict],
                                     global_beta_m: bool = True) -> Dict[str, ModelResult]:
    """
    Fit multiple peaks progressively with automatic Po-212 composite model handling.
    
    Automatically detects 'Po212' peak and routes to fit_po212_alpha_beta().
    All other peaks use standard Crystal Ball fitting.
    
    Args:
        energies: Full energy axis [MeV]
        counts: Full histogram counts
        peak_definitions: List of dicts, each with:
            - 'name': Isotope name (e.g., "Th228", "Po212")
            - 'position': Expected peak position [MeV]
            - 'window': (E_min, E_max) fitting window [MeV]
            - 'sigma_init': (optional) Initial width guess [MeV]
        global_beta_m: If True, compute average beta/m from individual fits
        
    Returns:
        Dictionary of {isotope_name: ModelResult} for each peak
        
    Example:
        >>> peak_defs = [
        ...     {'name': 'Th228', 'position': 4.5, 'window': (4.0, 4.7)},
        ...     {'name': 'Po212', 'position': 7.5, 'window': (6.3, 8.0)},  # Auto-routed!
        ... ]
        >>> results = fit_multi_crystalball_progressive(energies, counts, peak_defs)
    """
    individual_fits = {}
    beta_values = []
    m_values = []
    
    for peak_def in peak_definitions:
        name = peak_def['name']
        
        # Select fitting function: Po-212 uses composite α+β model, others use standard CB
        fit_func = fit_po212_alpha_beta if name == 'Po212' else fit_single_crystalball
        label_suffix = " (α+β)" if name == 'Po212' else ""
        
        # Common arguments (both functions use compatible signatures)
        fit_kwargs = {
                        'energies': energies,
                        'counts': counts,
                        'energy_window': peak_def['window'],
                        'sigma_init': peak_def.get('sigma_init', None)
                    }
        # Position argument has different names
        position_key = 'alpha_position' if name == 'Po212' else 'peak_position'
        fit_kwargs[position_key] = peak_def['position']
        
        try:
            result = fit_func(**fit_kwargs)
            
            # Store result and extract tail parameters
            individual_fits[name] = result
            beta_values.append(result.params['cb_beta'].value)
            m_values.append(result.params['cb_m'].value)
            
            print(f"  ✓ {name}{label_suffix}: x0={result.params['cb_x0'].value:.4f} MeV, "
                  f"σ={result.params['cb_sigma'].value:.4f} MeV")
        except Exception as e:
            print(f"  ✗ {name}: Fit failed - {e}")
    
    if not individual_fits:
        raise RuntimeError("All individual fits failed")
    
    if global_beta_m and len(beta_values) > 1:
        avg_beta = np.mean(beta_values)
        avg_m = np.mean(m_values)
        print(f"\n  Average tail params: β={avg_beta:.3f}, m={avg_m:.3f}")
    
    return individual_fits

# ============================================================================
# ENERGY CALIBRATION
# ============================================================================

def derive_energy_calibration(fit_results: Dict[str, ModelResult],
                              literature_energies: Dict[str, float],
                              use_peaks: Optional[List[str]] = None,
                              order: int = 1) -> EnergyCalibration:
    """
    Derive energy calibration from fitted peaks and literature values.
    
    Supports both linear and quadratic calibration:
    - Linear (order=1): E_true = a * E_SCA + b
    - Quadratic (order=2): E_true = a * E_SCA² + b * E_SCA + c
    
    Quadratic is recommended when including Po-212 to handle detector non-linearity.
    
    Parameters:
    -----------
    fit_results : Dict[str, ModelResult]
        Dictionary of fitted peaks from fit_multi_crystalball_progressive()
    literature_energies : Dict[str, float]
        True energies from literature [MeV]
        Example: {'Th228': 5.423, 'Ra224': 5.686, 'Rn220': 6.405, 'Po216': 6.906, 'Po212': 8.785}
    use_peaks : Optional[List[str]]
        Subset of peaks to use for calibration (default: all available)
        Recommend using 3+ peaks for quadratic, 2-3 for linear
    order : int
        Polynomial order: 1 for linear, 2 for quadratic (default: 1)
    
    Returns:
    --------
    EnergyCalibration
        Calibration object with apply() and inverse() methods
        
    Example:
    --------
    >>> # Linear calibration (2-4 peaks)
    >>> cal_linear = derive_energy_calibration(
    ...     fit_results,
    ...     {'Th228': 5.423, 'Ra224': 5.686, 'Rn220': 6.405, 'Po216': 6.906},
    ...     order=1
    ... )
    >>> 
    >>> # Quadratic calibration (3+ peaks, including Po212)
    >>> cal_quad = derive_energy_calibration(
    ...     fit_results,
    ...     {'Th228': 5.423, 'Ra224': 5.686, 'Rn220': 6.405, 'Po216': 6.906, 'Po212': 8.785},
    ...     order=2
    ... )
    >>> E_true = cal_quad.apply(E_SCA)  # Apply to data [mV] -> [MeV]
    """
    if order not in [1, 2]:
        raise ValueError(f"Only order=1 (linear) and order=2 (quadratic) supported, got {order}")
    
    # Select peaks to use
    if use_peaks is None:
        use_peaks = list(fit_results.keys())
    
    # Extract anchor points: (E_SCA [mV], E_true [MeV])
    anchors = {}
    E_SCA_list = []
    E_true_list = []
    
    for name in use_peaks:
        if name not in fit_results:
            print(f"Warning: Peak '{name}' not in fit results, skipping")
            continue
        if name not in literature_energies:
            print(f"Warning: No literature value for '{name}', skipping")
            continue
        
        # Handle different parameter names:
        # - Regular CB fits: 'cb_x0'
        # - Po212 composite fit: 'cb_x0' (same as regular)
        params = fit_results[name].params
        if 'cb_x0' in params:
            E_SCA = params['cb_x0'].value
        else:
            E_SCA = params['cb_x0'].value
        
        E_true = literature_energies[name]
        
        anchors[name] = (E_SCA, E_true)
        E_SCA_list.append(E_SCA)
        E_true_list.append(E_true)
    
    min_points = order + 1
    if len(E_SCA_list) < min_points:
        raise ValueError(f"Need at least {min_points} anchor points for order={order} calibration, got {len(E_SCA_list)}")
    
    # Polynomial least-squares fit using np.polyfit
    E_SCA_arr = np.array(E_SCA_list)
    E_true_arr = np.array(E_true_list)
    
    # polyfit returns coefficients in descending order: [a, b, c] for order=2
    coeffs = np.polyfit(E_SCA_arr, E_true_arr, deg=order)
    
    if order == 2:
        a, b, c = coeffs
        E_pred = a * E_SCA_arr**2 + b * E_SCA_arr + c
    else:
        a, b = coeffs
        c = None
        E_pred = a * E_SCA_arr + b
    
    # Compute residuals per anchor point (E_true - E_pred)
    residuals = E_true_arr - E_pred
    
    calibration = EnergyCalibration(a=a, b=b, c=c, 
                                    anchors=anchors,
                                    residuals=residuals,
                                    order=order)
    
    return calibration

# ============================================================================
# PURE FUNCTIONS - PURITY CALCULATION 
# ============================================================================

def compute_purity_at_energies(energies: np.ndarray,
                               fit_result: ModelResult,
                               peak_index: int) -> np.ndarray:
    """
    Compute isotope purity at each energy point (Option B).
    
    Purity = fraction of PDF contributed by specific peak.
    Used to identify "pure" regions where one isotope dominates.
    
    Args:
        energies: Energy points to evaluate [keV]
        fit_result: Multi-peak fit result from fit_multi_crystalball
        peak_index: Which peak to compute purity for (1-indexed)
        
    Returns:
        Array of purity values [0, 1] where:
        - 1.0 = 100% of signal from this peak
        - 0.5 = 50% from this peak, 50% from others
        - 0.0 = No signal from this peak
        
    Example:
        >>> # Find where Th228 (peak 1) contributes >95%
        >>> purity = compute_purity_at_energies(E_grid, fit_result, peak_index=1)
        >>> pure_region = E_grid[purity > 0.95]
    """
    # Evaluate each component at the given energies
    components = fit_result.eval_components(x=energies)
    
    # Sum all components to get total PDF
    total_pdf = sum(components.values())
    
    # Get specific peak contribution
    peak_name = f'cb{peak_index}_'
    if peak_name not in components:
        raise ValueError(f"Peak {peak_index} not found in fit result")
    
    peak_pdf = components[peak_name]
    
    # Purity = fraction from this peak (avoid division by zero)
    purity = peak_pdf / (total_pdf + 1e-10)
    
    return purity


# ============================================================================
# PURE FUNCTIONS - RANGE DERIVATION (Option C - Windowed)
# ============================================================================

def derive_isotope_ranges(fit_results: Dict[str, ModelResult],
                         calibration: EnergyCalibration,
                         literature_energies: Dict[str, float],
                         n_sigma: float = 1.0) -> Dict[str, IsotopeRange]:
    """
    Derive isotope energy ranges from individual peak fits in calibrated space.
    
    This is the simplified version that matches the notebook workflow:
    1. Extract fitted parameters (x0, sigma) in SCA scale
    2. Transform to calibrated energy using calibration.apply() and calibration.derivative()
    3. Define range as [x0 - n_sigma*sigma, x0 + n_sigma*sigma] in calibrated space
    
    Args:
        fit_results: Dictionary of individual peak fits {isotope_name: ModelResult}
                    Fits should be in SCA scale with 'cb_' parameter prefix
        calibration: EnergyCalibration object for transforming from SCA to calibrated space
        literature_energies: Dictionary of literature energies {isotope_name: E_true [MeV]}
                            Used for comparison/validation
        n_sigma: Number of sigmas for range definition (default: 1.0)
        
    Returns:
        Dictionary {isotope_name: IsotopeRange} with ranges in calibrated energy
        
    Example:
        >>> ranges = derive_isotope_ranges(
        ...     fit_results,
        ...     calibration,
        ...     literature_energies={'Th228': 5.423, 'Ra224': 5.686},
        ...     n_sigma=1.0
        ... )
        >>> print(ranges["Th228"])
        IsotopeRange(Th228: [5.200, 5.600] MeV, peak=5.423±0.085, purity=100%)
    """
    isotope_ranges = {}
    
    for name, fit_result in fit_results.items():
        # Extract fitted parameters (in SCA scale)
        params = fit_result.params
        x0_SCA = params['cb_x0'].value
        sigma_SCA = params['cb_sigma'].value
        
        # Convert to true energy using calibration
        x0_true = calibration.apply(np.array([x0_SCA]))[0]
        
        # Propagate uncertainty using derivative (handles both linear and quadratic)
        sigma_true = abs(calibration.derivative(np.array([x0_SCA]))[0]) * sigma_SCA
        
        # Define range as ± n_sigma in TRUE energy
        E_min = x0_true - n_sigma * sigma_true
        E_max = x0_true + n_sigma * sigma_true
        
        # Get literature value for comparison (if available)
        E_lit = literature_energies.get(name, 0.0)
        
        isotope_ranges[name] = IsotopeRange(
            name=name,
            E_min=E_min,
            E_max=E_max,
            E_peak=x0_true,
            sigma=sigma_true,
            purity=1.0  # Individual fit assumed pure
        )
    
    return isotope_ranges


# ============================================================================
# CONVENIENCE FUNCTIONS - COMBINED WORKFLOWS
# ============================================================================

def ranges_to_dict(ranges: Dict[str, IsotopeRange]) -> Dict[str, Tuple[float, float]]:
    """
    Convert IsotopeRange objects to simple {isotope: (E_min, E_max)} dict.
    
    This format is compatible with existing multiiso workflows that expect
    isotope_ranges as {isotope: (Emin, Emax)}.
    
    Args:
        ranges: Dictionary of IsotopeRange objects
        
    Returns:
        Dictionary {isotope: (E_min, E_max)} in keV
        
    Example:
        >>> ranges_dict = ranges_to_dict(ranges)
        >>> # Use with existing workflows:
        >>> df = workflow_s2_area_multiiso(set_pmt, isotope_ranges=ranges_dict)
    """
    return {name: (r.E_min, r.E_max) for name, r in ranges.items()}


# ============================================================================
# OVERLAP RESOLUTION - LIKELIHOOD CROSSOVER METHOD
# ============================================================================

def _extract_fit_params_calibrated(fit: ModelResult, calibration: EnergyCalibration) -> Tuple[float, float, float, float]:
    """Extract Crystal Ball parameters transformed to calibrated energy space."""
    x0_sca = fit.params['cb_x0'].value
    sigma_sca = fit.params['cb_sigma'].value
    beta = fit.params['cb_beta'].value
    m = fit.params['cb_m'].value
    
    x0_cal = calibration.apply(np.array([x0_sca]))[0]
    sigma_cal = abs(calibration.derivative(x0_sca)) * sigma_sca
    
    return x0_cal, sigma_cal, beta, m


def _compute_search_range(x0_1: float, x0_2: float, sigma_1: float, sigma_2: float) -> Tuple[float, float]:
    """Compute default search range for intersection: midpoint ± 2σ."""
    midpoint = (x0_1 + x0_2) / 2
    sigma_avg = (sigma_1 + sigma_2) / 2
    return (midpoint - 2*sigma_avg, midpoint + 2*sigma_avg)


def _find_pdf_crossover(E_grid: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> float:
    """Find energy where |P1 - P2| is minimum (crossover point)."""
    diff = np.abs(P1 - P2)
    idx_cross = np.argmin(diff)
    return E_grid[idx_cross]


def find_crystalball_intersection(fit1: ModelResult, fit2: ModelResult,
                                  calibration: EnergyCalibration,
                                  search_range: Optional[Tuple[float, float]] = None,
                                  n_points: int = 10000) -> float:
    """
    Find energy where two Crystal Ball PDFs have equal likelihood (Bayes-optimal boundary).
    
    Args:
        fit1, fit2: ModelResult for the two isotopes
        calibration: Energy calibration to transform peaks to true energy
        search_range: (E_min, E_max) in MeV to search. If None, use midpoint ± 2σ
        n_points: Number of points for grid search
        
    Returns:
        Boundary energy in MeV where P(E|iso1) = P(E|iso2)
    """
    # Extract parameters in calibrated space
    x0_1_cal, sigma_1_cal, beta_1, m_1 = _extract_fit_params_calibrated(fit1, calibration)
    x0_2_cal, sigma_2_cal, beta_2, m_2 = _extract_fit_params_calibrated(fit2, calibration)
    
    # Determine search range
    if search_range is None:
        search_range = _compute_search_range(x0_1_cal, x0_2_cal, sigma_1_cal, sigma_2_cal)
    
    # Evaluate PDFs on grid
    E_grid = np.linspace(search_range[0], search_range[1], n_points)
    P1 = v_crystalball(E_grid, N=1.0, beta=beta_1, m=m_1, x0=x0_1_cal, sigma=sigma_1_cal)
    P2 = v_crystalball(E_grid, N=1.0, beta=beta_2, m=m_2, x0=x0_2_cal, sigma=sigma_2_cal)
    
    return _find_pdf_crossover(E_grid, P1, P2)


def _validate_overlap_pair(iso1: str, iso2: str, 
                          fit_results: Dict[str, ModelResult],
                          isotope_ranges: Dict[str, IsotopeRange]) -> Optional[Tuple[IsotopeRange, IsotopeRange]]:
    """Validate that pair has required data and check for overlap."""
    if iso1 not in fit_results or iso2 not in fit_results:
        print(f"Warning: Cannot resolve {iso1}-{iso2} overlap (missing fit results)")
        return None
    
    if iso1 not in isotope_ranges or iso2 not in isotope_ranges:
        print(f"Warning: Cannot resolve {iso1}-{iso2} overlap (missing ranges)")
        return None
    
    range1 = isotope_ranges[iso1]
    range2 = isotope_ranges[iso2]
    
    if range1.E_max <= range2.E_min:
        print(f"  {iso1}-{iso2}: No overlap detected (gap present)")
        return None
    
    return range1, range2


def _create_resolved_range(original: IsotopeRange, new_boundary: float, is_upper: bool) -> IsotopeRange:
    """Create new IsotopeRange with updated boundary."""
    return IsotopeRange(name=original.name,
                        E_min=original.E_min if is_upper else new_boundary,
                        E_max=new_boundary if is_upper else original.E_max,
                        E_peak=original.E_peak,
                        sigma=original.sigma,
                        purity=original.purity)


def _log_overlap_resolution(iso1: str, iso2: str, range1: IsotopeRange, range2: IsotopeRange, E_boundary: float):
    """Print overlap resolution summary."""
    print(f"  {iso1}-{iso2}: Overlap resolved")
    print(f"    Original: {iso1} [---, {range1.E_max:.3f}], {iso2} [{range2.E_min:.3f}, ---]")
    print(f"    Boundary: {E_boundary:.3f} MeV (likelihood crossover)")
    print(f"    Resolved: {iso1} [---, {E_boundary:.3f}], {iso2} [{E_boundary:.3f}, ---]")


def resolve_overlapping_ranges(isotope_ranges: Dict[str, IsotopeRange],
                               fit_results: Dict[str, ModelResult],
                               calibration: EnergyCalibration,
                               overlap_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, IsotopeRange]:
    """
    Resolve overlapping isotope ranges using likelihood crossover boundaries.
    
    Args:
        isotope_ranges: Initial windowed ranges (μ ± nσ)
        fit_results: Crystal Ball fit results for each isotope
        calibration: Energy calibration
        overlap_pairs: List of (isotope1, isotope2) tuples. Defaults to [('Th228', 'Ra224')]
                      
    Returns:
        Updated isotope_ranges with extended boundaries (no overlaps)
    """
    if overlap_pairs is None:
        overlap_pairs = [('Th228', 'Ra224')]
    
    resolved_ranges = dict(isotope_ranges)
    
    for iso1, iso2 in overlap_pairs:
        # Validate pair and check for overlap
        result = _validate_overlap_pair(iso1, iso2, fit_results, resolved_ranges)
        if result is None:
            continue
        
        range1, range2 = result
        
        # Find likelihood crossover boundary
        E_boundary = find_crystalball_intersection(fit_results[iso1], fit_results[iso2], calibration,
                                                   search_range=(range1.E_peak - 0.5, range2.E_peak + 0.5))
        
        # Log resolution
        _log_overlap_resolution(iso1, iso2, range1, range2, E_boundary)
        
        # Update ranges with new boundary
        resolved_ranges[iso1] = _create_resolved_range(range1, E_boundary, is_upper=True)
        resolved_ranges[iso2] = _create_resolved_range(range2, E_boundary, is_upper=False)
    
    return resolved_ranges


# ============================================================================
# HIERARCHICAL FITTING - FULL SPECTRUM FITTING
# ============================================================================

def prepare_hierarchical_fit(fit_results: Dict[str, ModelResult],
                             calibration: EnergyCalibration,
                             spectrum_calibrated: np.ndarray,
                             exclude_from_shape: Optional[List[str]] = None) -> Tuple[Dict[str, float], float, float, np.ndarray, np.ndarray]:
    """
    Prepare parameters for hierarchical fitting from preliminary fits.
    
    Computes average shape parameters (β, m) excluding outliers, transforms sigmas to 
    calibrated space, and creates calibrated histogram.
    
    Returns: (calibrated_sigmas, beta_avg, m_avg, energies_calibrated, counts_calibrated)
    """
    if exclude_from_shape is None:
        exclude_from_shape = ['Po212']
    
    # Compute average shape parameters (exclude outliers)
    fit_results_clean = {k: v for k, v in fit_results.items() 
                        if k not in exclude_from_shape}
    
    beta_avg = np.mean([r.params['cb_beta'].value for r in fit_results_clean.values()])
    m_avg = np.mean([r.params['cb_m'].value for r in fit_results_clean.values()])
    
    print(f"\nAverage shape parameters (excluding {exclude_from_shape}):")
    print(f"  β_avg = {beta_avg:.3f}")
    print(f"  m_avg = {m_avg:.3f}")
    
    # Transform sigmas to calibrated space using calibration derivative
    calibrated_sigmas = {}
    for name, result in fit_results.items():
        x0_SCA = result.params['cb_x0'].value
        sigma_SCA = result.params['cb_sigma'].value
        calibrated_sigmas[name] = abs(calibration.derivative(np.array([x0_SCA]))[0]) * sigma_SCA
    
    # Create calibrated histogram
    counts_calibrated, bin_edges = np.histogram(spectrum_calibrated, bins=120, range=(4.5, 9.2))
    energies_calibrated = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return calibrated_sigmas, beta_avg, m_avg, energies_calibrated, counts_calibrated


def _print_hierarchical_fit_summary(result: ModelResult,
                                    peak_definitions: List[Dict[str, Any]]) -> None:
    n_free = len([p for p in result.params.values() if p.vary])
    n_constrained = len([p for p in result.params.values() if p.expr is not None])
    print(f"\n  Hierarchical Fit:")
    print(f"    Total peaks: {len(peak_definitions)}")
    print(f"    Free parameters: {n_free}")
    print(f"    Constrained parameters: {n_constrained}")
    print(f"    χ²_reduced: {result.redchi:.3f}")
    if hasattr(result, 'bic'):
        print(f"    BIC: {result.bic:.1f}")

def fit_full_spectrum_hierarchical(energies: np.ndarray, counts: np.ndarray,
                                   peak_definitions: List[Dict[str, Any]],
                                   calibrated_sigmas: Dict[str, float],
                                   beta_avg: float, m_avg: float,
                                   normalize: bool = True,
                                   x0_tolerance: float = 0.02) -> ModelResult:
    """
    Fit full spectrum with fixed shapes and constrained amplitudes.
    
    Caller must pre-compute: beta_avg, m_avg (from fit_results), calibrated_sigmas (via calibration.derivative()).
    peak_definitions: [{'name', 'ref_energy', 'branching_ratio'(optional)}, ...]
    x0_tolerance: Allowed position variation around literature values [MeV] (default: 0.02)
    """
    y = counts.astype(float)
    if normalize and y.max() > 0:
        y = y / y.max()
    
    if not peak_definitions:
        raise ValueError("No peaks defined - cannot build model")
    
    models = []
    all_params = lmfit.Parameters()
    
    for peak_def in peak_definitions:
        name = peak_def['name']
        E_peak = peak_def['ref_energy']
        
        # Infer base_isotope: for satellites, remove '_sat' suffix
        if '_sat' in name:
            base_isotope = name.replace('_sat', '')
        else:
            base_isotope = name
        
        # Compute N_expr from branching_ratio if provided
        branching_ratio = peak_def.get('branching_ratio', None)
        if branching_ratio is not None:
            # Satellite peak: N = base_isotope_N * branching_ratio
            N_expr = f'{base_isotope}_N * {branching_ratio}'
        else:
            N_expr = None
        
        if '_sat' in name:
            x0_bounds = None
        else:
            x0_bounds = (E_peak - x0_tolerance, E_peak + x0_tolerance)

        if base_isotope not in calibrated_sigmas:
            raise ValueError(f"base_isotope '{base_isotope}' not found in calibrated_sigmas")
        
        sigma_cal = calibrated_sigmas[base_isotope]
        
        prefix = f"{name}_"
        peak_model = lmfit.Model(v_crystalball, prefix=prefix)
        peak_params = peak_model.make_params()
        
        _setup_crystalball_params(peak_params, prefix, N_value=1.0, x0_value=E_peak,
                                 beta_value=beta_avg, m_value=m_avg, sigma_value=sigma_cal,
                                 N_bounds=(0.0, 1e6), x0_bounds=x0_bounds, vary_shape=False, N_expr=N_expr)
        
        models.append(peak_model)
        all_params.update(peak_params)
    
    composite_model = sum(models[1:], models[0])
    result = composite_model.fit(y, all_params, x=energies)
    
    _print_hierarchical_fit_summary(result, peak_definitions)
    
    return result

