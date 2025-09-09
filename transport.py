import numpy as np
from .config import DRIFT_VELOCITY_PARAMS
from typing import Sequence
from scipy.signal import find_peaks, peak_widths
# -------------------------------
# Drift velocity models
# -------------------------------

def transport_saturation(rE: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """
    Saturating + rational function for electron drift velocity.
    Args:
        rE : reduced electric field (V/cm/bar)
    """
    return p0 * (1.0 - np.exp(-p1 * rE)) + (p2 * rE) / (1.0 + p3 * rE)

def compute_reduced_field(field: float, gas_density: float = 4.91e19 ) -> float:
    """Compute reduced field E/n [V/cm^-2].
    Default gas_density corresponds to 2 bar at room temperature.
    """
    return field / gas_density * 1e17  # Convert to V/cm^2


def redfield_to_speed(rE: float, params: dict = DRIFT_VELOCITY_PARAMS) -> float:   # This will be renamed as this fit only applies to mid-range reduced fields
    """Compute drift velocity from reduced field using given params.
    Args:
        rE: reduced electric field [Td]
        params: dict with keys p0, p1, p2, p3 for transport_saturation()
    Returns: drift velocity [mm/μs]
    """
    return transport_saturation(rE, **params)

def drift_curve(rE_list: Sequence[float], params: dict) -> np.ndarray:
    """Vectorized evaluation of drift velocities for a list of reduced fields."""
    return np.array([redfield_to_speed(rE, params) for rE in rE_list])

# -------------------------------
# Diffusion models
# -------------------------------
def longitudinal_diffusion_coeff(rE: float, a: float, b: float) -> float:
    """
    Longitudinal diffusion coefficient [mm^2/μs].
    Model form: D_L(rE) = a / sqrt(rE) + b
    (placeholder; replace with experimental fit).
    """
    return a / np.sqrt(rE) + b

def transverse_diffusion_coeff(rE: float, c: float, d: float) -> float:
    """
    Transverse diffusion coefficient [mm^2/μs].
    Model form: D_T(rE) = c / sqrt(rE) + d
    """
    return c / np.sqrt(rE) + d


# -------------------------------
# Cloud evolution
# -------------------------------
def diffusion_sigma(D: float, drift_time: float) -> float:
    """
    RMS spread of the electron cloud [mm].
    σ = sqrt(2 * D * t)
    """
    return np.sqrt(2 * D * drift_time)

def s2_pulse_width(z_drift: float, rE: float, drift_params: dict,
                   diff_params: dict) -> float:
    """
    Estimate the S2 pulse width [μs] from drift + longitudinal diffusion.

    Args:
        z_drift: drift length [mm]
        rE: reduced electric field [V/cm/bar]
        drift_params: dict with drift velocity params
        diff_params: dict with diffusion params {'a':..., 'b':...}

    Returns:
        σ_t: temporal width of the S2 [μs]
    """
    v_d = redfield_to_speed(rE, drift_params)          # [mm/μs]
    t_d = z_drift / v_d                          # drift time [μs]

    D_L = longitudinal_diffusion_coeff(rE, **diff_params)  # [mm^2/μs]
    σ_z = diffusion_sigma(D_L, t_d)              # [mm]
    σ_t = σ_z / v_d                              # [μs]

    return σ_t
