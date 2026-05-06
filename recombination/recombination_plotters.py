from typing import Dict, Callable

from matplotlib.pylab import xlabel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# In your recombination_plotters.py (or similar module)

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

def plot_metric_vs_field(run_data: Dict[str, pd.DataFrame], 
                         run_manifest: Dict[str, Dict],
                         y_col: str, y_err_col: str, ylabel: str,
                         title: str) -> plt.Figure:
    """
    Universal plotter for comparing a specific metric across multiple runs
    as a function of the drift field.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_id, df in run_data.items():
        if y_col not in df.columns:
            print(f"Warning: {y_col} not found in {run_id}. Skipping plot line.")
            continue
            
        e_el = run_manifest[run_id].get('EL_field_Vcm', 'Unknown')
        isotope = run_manifest[run_id].get('isotope', 'Unknown')
        
        ax.errorbar(df['drift_field'], df[y_col], yerr=df[y_err_col], 
                    fmt='o-', capsize=4, label=f"{run_id}: {isotope}, $E_{{EL}}$ = {e_el} V/cm" )

    ax.set(title=title, xlabel="Drift Field (V/cm)", ylabel=ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Run Metadata")
    
    fig.tight_layout()
    return fig