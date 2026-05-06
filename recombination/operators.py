# core/physics_operators.py

import pandas as pd
from .datatypes import CorrectionFactor, CorrectionModel

# --- CONSTANTS ---
ALPHA_DECAY_KINEMATICS = {
    "Th228": {"q_value_kev": 5520.15, "m_parent_amu": 228.0, "daughter": "Ra-224"},
    "Ra224": {"q_value_kev": 5788.87, "m_parent_amu": 224.0, "daughter": "Rn-220"}
}

###############################################################################################################
# Physics Operators (Pure Functions)
###############################################################################################################


def calc_alpha_recoil_energy(q_value_kev: float, m_parent_amu: float, m_alpha_amu: float = 4.0) -> float:
    """
    Computes the recoil energy of the daughter nucleus following an alpha decay
    using 2-body conservation of momentum.
    
    Pure function: (Float, Float, [Float]) -> Float
    """
    return q_value_kev * (m_alpha_amu / m_parent_amu)

def get_isotope_recoil_energy(isotope_name: str) -> float:
    """
    Retrieves the theoretical recoil energy (keV) for a given isotope.
    Pure function: (String) -> Float
    """
    if isotope_name not in ALPHA_DECAY_KINEMATICS:
        raise KeyError(f"Kinematics for isotope '{isotope_name}' are not defined in the physics core.")
        
    params = ALPHA_DECAY_KINEMATICS[isotope_name]
    return calc_alpha_recoil_energy(params["q_value_kev"], params["m_parent_amu"])


def calc_expected_electrons(recoil_energy_kev: float, 
                            w_value_ev: float, 
                            p_desorp: CorrectionFactor = 1) -> float:
    """
    Computes theoretical expected electrons from a recoil event given the recoil energy and W-value.
    The effective energy deposited in the gas is scaled by the desorption probability.
    
    Pure function: (Float, Float, CorrectionFactor) -> Float
    """
    recoil_energy_ev = recoil_energy_kev * 1000.0
    
    # Scale the recoil energy by the probability that the ion successfully 
    # desorbs and deposits its energy in the active xenon volume.
    effective_energy = recoil_energy_ev * p_desorp.value
    
    return effective_energy / w_value_ev


############################################################################################################
# Recombination Pipeline Operators
############################################################################################################

def apply_gs2_conversion(df_s2: pd.DataFrame, gs2_factor: CorrectionFactor) -> pd.DataFrame:
    """
    Converts raw S2 areas into measured electron pairs.
    Pure function: (Data State, CorrectionFactor) -> New Data State
    """
    # 1. Allocate a new state to avoid mutating the input DataFrame
    df_out = df_s2.copy()
    
    # 2. Apply the core physics transformation
    df_out['N_e_meas'] = df_out['s2_mean'] / gs2_factor.value
    
    # 3. Apply legacy error propagation (can be upgraded to full propagation later)
    # dN_e_meas = S2_ci95 / g_s2
    df_out['dN_e_meas'] = df_out['s2_ci95'] / gs2_factor.value
    
    # 4. Auditability trail
    df_out['gs2_applied'] = gs2_factor.value
    
    return df_out

def apply_el_yield_conversion(df_s2: pd.DataFrame, 
                              gs2_artifact: CorrectionFactor, 
                              g_ratio_artifact: CorrectionFactor,
                              el_trend_model: CorrectionModel,
                              e_el_v_cm: float) -> pd.DataFrame:
    """
    Converts raw S2 areas into measured electrons using the full EL yield model:
    Y(E_EL) = B(g_s2) * (1 / t(theta, phi)) * f(E_EL)
    """
    df_out = df_s2.copy()
    
    # 1. Absolute Scale (X-ray response at reference field)
    base_yield = gs2_artifact.value
    
    # 2. Geometric Correction (Scale to Recoil topology)
    # If g_ratio = 0.89 (Xrays are 89% as efficient as recoils), then Recoil Yield = Base / 0.89
    recoil_yield_ref = base_yield / g_ratio_artifact.value
    
    # 3. Dynamic Field Correction
    relative_trend = el_trend_model.evaluate(e_el_v_cm)
    total_yield = recoil_yield_ref * relative_trend
    
    # 4. Convert S2 to Electrons
    df_out['N_e_meas'] = df_out['s2_mean'] / total_yield
    df_out['dN_e_meas'] = df_out['s2_ci95'] / total_yield
    
    # 5. Auditability Trail
    df_out['total_yield_applied'] = total_yield
    
    return df_out


def apply_transmission_efficiency(df_electrons: pd.DataFrame, 
                                  trans_model: CorrectionModel, 
                                  e_el: float) -> pd.DataFrame:
    """
    Corrects measured electrons for gate transparency (dependent on drift field).
    Pure function: (Data State, CorrectionModel, Float) -> New Data State
    """
    df_out = df_electrons.copy()
    
    # Evaluate transparency for each drift field point.

    df_out['eps_t'] = df_out['drift_field'].apply(
        lambda e_d: float(trans_model.evaluate(e_d, e_el))
    )
    
    # True number of drifting electrons before hitting the gate
    df_out['N_e_drift'] = df_out['N_e_meas'] / df_out['eps_t']
    
    # Propagate error (fractional errors add in quadrature, assuming eps_t error is negligible for now)
    df_out['dN_e_drift'] = df_out['dN_e_meas'] / df_out['eps_t'] 
    
    return df_out


###########################################################################################################
# Final Operators
###########################################################################################################


def compute_recombination_fraction(df_electrons: pd.DataFrame, n_e_exp: float) -> pd.DataFrame:
    """
    Computes the recombination fraction (r) and its statistical uncertainty.
    Pure function: (Data State, Float) -> New Data State
    """
    # 1. Allocate a new state to guarantee immutability
    df_out = df_electrons.copy()
    
    # 2. Compute recombination fraction: r = 1 - (N_e_drift / N_e_exp)
    df_out['recomb_factor'] = 1.0 - (df_out['N_e_drift'] / n_e_exp)
    
    # 3. Propagate statistical uncertainty: dr = dN_e_drift / N_e_exp
    df_out['recomb_error'] = df_out['dN_e_drift'] / n_e_exp
    
    # 4. Auditability trail
    df_out['n_e_exp_applied'] = n_e_exp
    
    return df_out
