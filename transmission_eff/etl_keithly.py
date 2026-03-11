import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple

def load_keithly_data(file_path):
    """Load Keithly data from a text file and return a DataFrame."""
    df = pd.read_csv(file_path, sep='\t', skiprows=0)
    df['Current_A'] = pd.to_numeric(df['Current_A'], errors='coerce')
    df['Current_pA'] = df['Current_A'] * 1e12  # Convert A to pA
    return df

def drop_outliers(data, column, threshold=3):
    """Remove outliers from a DataFrame based on a specified column and threshold."""
    mean = data[column].mean()
    std = data[column].std()
    outliers = data[np.abs(data[column] - mean) > threshold * std]
    return data[~data.index.isin(outliers.index)]


def process_single_measurement(file_path: Path) -> Tuple[float, float]:
    """Process a single Keithly measurement file and return mean current in pA and its std."""
    data = load_keithly_data(file_path)
    data_drop = drop_outliers(data, 'Current_pA')
    mean_current =  data_drop['Current_pA'].mean() 
    std_current = data_drop['Current_pA'].std()
    
    return mean_current, std_current

import numpy as np

def calculate_transmission_efficiency(i_mean: float, i_std: float, i0_mean: float, i0_std: float) -> tuple[float, float]:
    """
    Calculates T = 1 - I/I0 and applies standard error propagation.
    Returns (Transmission, Transmission_Error).
    """
    if i0_mean == 0:
        return 0.0, 0.0  # Safe guard, though physically unlikely
        
    t = 1.0 - (i_mean / i0_mean)
    t_err = np.sqrt((i_std / i0_mean)**2 + (i_mean * i0_std / (i0_mean**2))**2)
    
    return t, t_err

def calculate_electric_fields(v_cathode: float, v_gate: float, v_anode: float, 
                              el_gap_cm: float = 0.8, drift_gap_cm: float = 1.4) -> tuple[float, float, float]:
    """
    Calculates field magnitudes and their ratio.
    Returns (E_drift_V_cm, E_el_V_cm, R_factor).
    """
    e_drift = abs(v_cathode - v_gate) / drift_gap_cm
    e_el = abs(v_gate - v_anode) / el_gap_cm
    r_factor =  e_el / e_drift if e_drift != 0 else 0.0
    
    return e_drift, e_el, r_factor