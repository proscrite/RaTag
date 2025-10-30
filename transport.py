import numpy as np
from typing import Sequence
from dataclasses import replace
from .datatypes import Run
from .config import DRIFT_VELOCITY_PARAMS
from scipy.optimize import brentq

# -------------------------------
# Gas properties
# -------------------------------

k_B = 1.380649e-23  # Boltzmann constant, J/K

def gas_density_cm3(pressure_bar: float, temperature_K: float) -> float:
    """Return gas number density in cm^-3 from P [bar], T [K]."""
    P_Pa = pressure_bar * 1e5
    n_m3 = P_Pa / (k_B * temperature_K)
    return n_m3 * 1e-6  # cm^-3

def gas_density_N(n_cm3: float) -> float:
    """Convert number density in cm^-3 to N [atoms/cm^3]."""
    return n_cm3 * 6.022e23  # Convert to atoms/cm^3 using Avogadro's number

def with_gas_density(run: Run) -> Run:
    gd = gas_density_cm3(run.pressure, run.temperature)
    return replace(run, gas_density=gd)


# -------------------------------
# Drift velocity models
# -------------------------------

def transport_saturation(rE: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """
    Saturating + rational function for electron drift velocity.
    Args:
        rE : reduced electric field (Td)
    """
    return p0 * (1.0 - np.exp(-p1 * rE)) + (p2 * rE) / (1.0 + p3 * rE)

def compute_reduced_field(field_Vpcm: float, density_cm3: float) -> float:
    """
    Compute reduced field [V·cm²].
    """
    return field_Vpcm / density_cm3


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

def speed_to_redfield(v_drift: float, 
                     params: dict = DRIFT_VELOCITY_PARAMS,
                     rE_min: float = 0.01,
                     rE_max: float = 3.0) -> float:
    """
    Invert the drift velocity model to find reduced field [Td].
    
    Args:
        v_drift: drift velocity [cm/μs]
        params: dict with keys p0, p1, p2, p3 for transport_saturation()
        rE_min: minimum reduced field to search [Td]
        rE_max: maximum reduced field to search [Td]
    
    Returns:
        rE: reduced electric field [Td]
    """
    def residual(rE):
        return transport_saturation(rE, **params) - v_drift
    
    try:
        return brentq(residual, rE_min, rE_max)
    except ValueError as e:
        raise ValueError(f"Could not find solution for v_drift={v_drift} cm/μs. "
                        f"Valid range: [{transport_saturation(rE_min, **params):.3f}, "
                        f"{transport_saturation(rE_max, **params):.3f}] cm/μs") from e

def speed_to_redfield_vectorized(v_drifts: np.ndarray,
                                params: dict = DRIFT_VELOCITY_PARAMS,
                                rE_min: float = 0.01,
                                rE_max: float = 3.0) -> np.ndarray:
    """
    Vectorized version for multiple drift velocities.
    
    Args:
        v_drifts: array of drift velocities [cm/μs]
        params: dict with keys p0, p1, p2, p3
        rE_min, rE_max: search bounds [Td]
    
    Returns:
        array of reduced electric fields [Td]
    """
    return np.array([speed_to_redfield(v, params, rE_min, rE_max) 
                     for v in v_drifts])


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
