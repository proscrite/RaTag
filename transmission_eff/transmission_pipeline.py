from pathlib import Path
from typing import List, Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from dataclasses import replace


from .datatypes import TransmissionPoint, TransmissionRun
from .transmission_workflow import (build_run_catalog, export_transparency_model, 
                                    generate_diagnostic_plots, generate_physics_surface, 
                                    generate_primary_plots)


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# The Orchestrator
# ---------------------------------------------------------
def execute_transmission_pipeline(config_path: str, output_dir: str = "artifacts/transmission_efficiency"):
    """Main orchestrator for the transmission efficiency analysis."""
    
    config = load_config(config_path)
    base_dir = Path(config['data_dir'])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Task 1: Build the Data Catalog
    runs_catalog = build_run_catalog(config['runs'], base_dir)
    
    # Task 2: Extract & Export the Physics-Informed Model
    updated_catalog = export_transparency_model(runs_catalog, out_path / "transmission_model.json")
    
    # Task 3: Generate Visual Diagnostics (Absorbs primary and derived plots)
    generate_diagnostic_plots(updated_catalog, config['plots'], out_path)
    
    generate_physics_surface(updated_catalog, output_dir=out_path)  

    # Task 4: Generate other plots (for completeness, not strictly required)
    generate_primary_plots(runs_catalog, config['plots'], output_dir=out_path)

    print(f"Pipeline execution complete. Artifacts saved to {out_path}")

if __name__ == "__main__":
    yaml_file = "/Users/pabloherrero/sabat/RaTagging/RaTag/transmission_eff/configs/tranmission_measurements.yaml"
    outpath = "/Users/pabloherrero/sabat/RaTagging/artifacts/transmission_efficiency"
    print("Starting Transmission Pipeline...")
    
    # all_runs = generate_primary_plots(config, output_dir=outpath)
    
    execute_transmission_pipeline(yaml_file, output_dir=outpath)
    print("Pipeline Complete!")