#!/usr/bin/env python3
"""
RaTag YAML-Driven Pipeline Orchestrator.
The YAML configuration file is the absolute source of truth for both parameters and execution state.
"""
import argparse
from pathlib import Path
from datetime import datetime

from RaTag.io.file_ops import load_yaml
from RaTag.io.bootstrap import bootstrap_from_config

# Import our beautifully encapsulated high-level pipelines
from RaTag.alphas.alphas_pipeline import pipeline_alpha_calibration
from RaTag.el_tpc.xray_pipeline import pipeline_xray_calibration
from RaTag.thgem_tpc.recoil_pipeline import pipeline_recoil_analysis
# from RaTag.thgem_tpc.coincidence_pipeline import pipeline_coincidence_recoil
from RaTag.thgem_tpc.multiiso_pipeline import pipeline_multi_isotope


def main():
    # The ONLY CLI argument is the path to the config file
    parser = argparse.ArgumentParser(description='RaTag YAML-Driven Pipeline Orchestrator')
    parser.add_argument('config', type=Path, help='Path to YAML config file')
    parser.add_argument('--allowed-sets', nargs='+', default=None, help='Specific set directories to process')
    args = parser.parse_args()

    # 1. Load the Absolute Source of Truth
    config = load_yaml(args.config)
    exec_cfg = config.get('execution', {})
    
    print(f"\n{'='*60}")
    print(f"STARTING YAML-DRIVEN ANALYSIS: {config.get('run_id', 'UNKNOWN')}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    # 2. Bootstrap the base Run object (Handles instantiation and I/O discovery)
    run = bootstrap_from_config(args.config, allowed_sets=args.allowed_sets)

    # 3. Route sequentially based strictly on the YAML execution block
    if exec_cfg.get('run_alphas', False):
        run = pipeline_alpha_calibration(run, config=config)
        
    if exec_cfg.get('run_xrays', False):
        run = pipeline_xray_calibration(run, config=config)
        
    if exec_cfg.get('run_recoils', False):
        # run = pipeline_recoil_analysis(run, config=config)   # Deprecated: This is the decoupled Timing-Integration pipeline. We now use the new unified coincidence pipeline.
        run = pipeline_recoil_analysis(run, config=config)  # New single-pass pipeline
    if exec_cfg.get('run_multiiso', False):
        run = pipeline_multi_isotope(run, config=config)  # New multi-isotope pipeline

    

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE: {run.run_id}")
    print(f"Artifacts and cached state saved to: {run.root_directory}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()