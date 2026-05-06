from typing import Dict, Any, Callable
from pathlib import Path
import pandas as pd
import yaml
import json

from .datatypes import CorrectionFactor, CorrectionModel
from RaTag.transmission_eff.transmission_workflow import build_transparency_evaluator

def load_integration_config(config_path: Path) -> Dict[str, Any]:
    """Impure function to read the YAML configuration state."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_run_data(run_manifest: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """
    Impure function to load all CSVs defined in the manifest into memory.
    Returns a dictionary mapping run_id to its S2 Area DataFrame.
    """
    run_data = {}
    for run_id, metadata in run_manifest.items():
        path = Path(metadata['s2_areas_path'])
        if path.exists():
            run_data[run_id] = pd.read_csv(path)
        else:
            print(f"Warning: Data file for {run_id} not found at {path}")
    return run_data


def parse_gs2_artifact(filepath: Path) -> CorrectionFactor:
    """
    Adapter to parse the X-ray g_s2 calibration JSON.
    Pure function: (File State) -> CorrectionFactor
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    return CorrectionFactor(
        name="g_S2",
        value=data['gs2'],
        uncertainty=data['d_gs2'],
        units=data['units_gs2'],
        metadata={
            "W_value": data['Wi'], 
            "W_units": data['units_Wi'],
            "provenance": str(filepath)
        }
    )



def parse_transmission_artifact(filepath: Path ) -> CorrectionModel:
    """
    Adapter to parse the transmission efficiency JSON and return a callable 2D surface model.
    It wraps the core builder function to ensure physics logic is never duplicated.
    
    Pure function: (File State) -> CorrectionModel
    """
    # 1. Read the immutable file state
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if data.get("model") != "dynamic_sigmoid":
        raise ValueError(f"Unsupported transmission model type: {data.get('model')}")
        
    evaluator_closure = build_transparency_evaluator(filepath)
    
    # 3. Lock into the Functional Programming contract
    return CorrectionModel(
        name="Evaluated_Surface_Grid_Transmission",
        evaluate=evaluator_closure,
        metadata={
            "provenance": str(filepath),
            "model_type": data["model"]
        }
    )
def parse_desorption_artifact(filepath: Path ) -> CorrectionFactor:
    """
    Adapter to parse the alpha spectroscopy accumulation summary and extract 
    the mean desorption probability for Ra-224.
    
    Pure function: (File State) -> CorrectionFactor
    """
    df = pd.read_csv(filepath)
    
    # 1. Extract and convert percentage to fraction
    desorp_pct = df['desorption_probability_pct']
    mean_desorp_frac = desorp_pct.mean() / 100.0
    
    # 2. Compute uncertainty (Standard Error of the Mean)
    n_measurements = len(desorp_pct)
    std_desorp_frac = desorp_pct.std() / 100.0
    sem_desorp = std_desorp_frac / (n_measurements ** 0.5) if n_measurements > 1 else 0.0
    
    return CorrectionFactor(
        name="P_desorp",
        value=mean_desorp_frac,
        uncertainty=sem_desorp,
        units="fraction",
        metadata={
            "provenance": str(filepath),
            "n_measurements": n_measurements
        }
    )

def parse_geometric_artifact(filepath: Path ) -> CorrectionFactor:
    """
    Adapter to parse the X-ray Monte Carlo geometric efficiency.
    Extracts the g_ratio (X-ray / Recoil solid angle acceptance).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # The JSON defines g_ratio as X-ray response relative to Recoil response
    return CorrectionFactor(
        name="Geometric_Ratio_Xray_to_Recoil",
        value=data["g_ratio"],
        uncertainty=0.0, # Add uncertainty later if MC stats allow
        units="fraction",
        metadata={"provenance": str(filepath)}
    )

def parse_el_trend_artifact(filepath: Path ) -> CorrectionModel:
    """
    Adapter to parse the relative EL yield trend f(E_EL).
    Expects a JSON with 'slope', 'intercept', and 'reference_field'.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    slope = data["slope"]
    intercept = data["intercept"]
    ref_field = data.get("reference_field", 2375.0)
    
    # Define the pure closure for the relative trend normalized to 1.0 at reference field
    def evaluate_el_trend(e_el_v_cm: float) -> float:
        raw_val = slope * e_el_v_cm + intercept
        ref_val = slope * ref_field + intercept
        return raw_val / ref_val

    return CorrectionModel(
        name="Relative_EL_Yield_Trend",
        evaluate=evaluate_el_trend,
        metadata={"provenance": str(filepath), "reference_field": ref_field}
    )