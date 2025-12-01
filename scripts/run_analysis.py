#!/usr/bin/env python3
"""
Run analysis script - Execute complete RaTag pipeline from config file.

This script loads a YAML configuration file and runs the complete analysis
pipeline including preparation (S1/S2 timing) and integration (S2 areas).

Usage:
    # Full pipeline
    python scripts/run_analysis.py path/to/config.yaml
    
    # Only preparation stage
    python scripts/run_analysis.py path/to/config.yaml --prepare-only
    
    # Only integration stage (assumes preparation done)
    python scripts/run_analysis.py path/to/config.yaml --integrate-only

Example:
    python scripts/run_analysis.py ../configs/run8_analysis.yaml
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime

from RaTag.core.datatypes import Run
from RaTag.core.config import IntegrationConfig, FitConfig
from RaTag.workflows.run_construction import initialize_run
from RaTag.pipelines.run_preparation import prepare_run, prepare_run_multiiso
from RaTag.pipelines.recoil_only import recoil_pipeline, recoil_pipeline_multiiso
from RaTag.pipelines.isotope_preparation import prepare_isotope_separation


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_run_from_config(config: dict) -> Run:
    """Create Run object from configuration."""
    exp = config['experiment']
    
    return Run(
        run_id=config['run_id'],
        root_directory=Path(config['data']['raw_data_path']),
        el_field=exp['el_field'],
        target_isotope=exp['target_isotope'],
        pressure=exp['pressure'],
        temperature=exp['temperature'],
        sampling_rate=exp['sampling_rate'],
        el_gap=exp['el_gap'],
        drift_gap=exp['drift_gap'],
        width_s2=exp['width_s2'],
        W_value=exp['W_value'],
        E_gamma_xray=exp['E_gamma_xray'],
        sets=[]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run RaTag analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', type=Path, help='Path to YAML config file')
    parser.add_argument('--alphas-only', action='store_true',
                       help='Only run alpha energy mapping and calibration')
    
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only run preparation (skip integration)')
    parser.add_argument('--integrate-only', action='store_true',
                       help='Only run integration (assumes preparation done)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prepare_only and args.integrate_only:
        parser.error("Cannot specify both --prepare-only and --integrate-only")
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Check if multi-isotope mode
    is_multiiso = config.get('multi_isotope', {}).get('enabled', False)
    isotope_ranges = None
    
    # Create Run object
    run = create_run_from_config(config)
    print("\nInitializing run (populating sets from directory)...")    
    run = initialize_run(run, max_files=None)
    print(f"Found {len(run.sets)} sets")

    if is_multiiso:
        # Convert isotope ranges from list to tuple
        isotope_ranges = {
            isotope: tuple(range_vals) 
            for isotope, range_vals in config['multi_isotope']['isotope_ranges'].items()
        }
        print(f"Multi-isotope mode enabled with ranges: {isotope_ranges}")
        
        # Generate energy maps if configured
        energy_cfg = config['multi_isotope'].get('energy_mapping', {})
        if energy_cfg.get('generate', True):
            run = prepare_isotope_separation(run,
                                            files_per_chunk=energy_cfg.get('files_per_chunk', 10),
                                            fmt=energy_cfg.get('format', '8b'),
                                            scale=energy_cfg.get('scale', 0.1),
                                            pattern=energy_cfg.get('pattern', '*Ch4.wfm'))
        if args.alphas_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER ALPHA ENERGY MAPPING (--alphas-only flag)")
            print(f"{'='*60}")
            return
    
    # Log start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"STARTING ANALYSIS: {config['run_id']}")
    print(f"Time: {timestamp}")
    print(f"Config: {args.config}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # PREPARATION STAGE
    # ========================================================================
    if not args.integrate_only:
        print("\n" + "="*60)
        print("STAGE 1: PREPARATION")
        print("="*60)
        
        prep_cfg = config['pipeline']['preparation']
        
        if is_multiiso:
            run = prepare_run_multiiso(run,
                                       isotope_ranges=isotope_ranges,
                                       max_files=None,
                                       max_frames_s1=prep_cfg['max_frames_s1'],
                                       max_frames_s2=prep_cfg['max_frames_s2'],
                                       threshold_s1=prep_cfg['threshold_s1'],
                                       threshold_s2=prep_cfg['threshold_s2'],
                                       s2_duration_cuts=tuple(prep_cfg['s2_duration_cuts']))
        else:
            run = prepare_run(run,
                            max_files=None,
                            max_frames_s1=prep_cfg['max_frames_s1'],
                            max_frames_s2=prep_cfg['max_frames_s2'],
                            threshold_s1=prep_cfg['threshold_s1'],
                            threshold_s2=prep_cfg['threshold_s2'],
                            s2_duration_cuts=tuple(prep_cfg['s2_duration_cuts']))
        
        print("\n✓ Preparation complete")
        
        if args.prepare_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER PREPARATION (--prepare-only flag)")
            print(f"{'='*60}")
            return
    
    # ========================================================================
    # INTEGRATION STAGE
    # ========================================================================
    if not args.prepare_only:
        print("\n" + "="*60)
        print("STAGE 2: INTEGRATION")
        print("="*60)
        
        int_cfg = config['pipeline']['integration']
        
        # Create config objects
        integration_config = IntegrationConfig(
            n_pedestal=int_cfg['integration_config']['n_pedestal'],
            ma_window=int_cfg['integration_config']['ma_window'],
            bs_threshold=int_cfg['integration_config']['bs_threshold'],
            dt=int_cfg['integration_config']['dt']
        )
        
        fit_config = FitConfig(
            bin_cuts=tuple(int_cfg['fit_config']['bin_cuts']),
            nbins=int_cfg['fit_config']['nbins'],
            exclude_index=int_cfg['fit_config']['exclude_index']
        )

        
        if is_multiiso:
            run = recoil_pipeline_multiiso(run,
                                          isotope_ranges=isotope_ranges,
                                          max_files=int_cfg['max_files'],
                                          integration_config=integration_config,
                                          fit_config=fit_config)
        else:
            run = recoil_pipeline(run,
                                max_files=int_cfg['max_files'],
                                integration_config=integration_config,
                                fit_config=fit_config)
        
        print("\n✓ Integration complete")
    #/Users/pabloherrero/sabat/RaTagging/scope_data/waveforms/RUN18_multi/processed_data/FieldScan_Gate0050_Anode1950_metadata.json.
    # ========================================================================
    # SUMMARY
    # ========================================================================
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE: {config['run_id']}")
    print(f"End time: {timestamp_end}")
    print(f"Results saved in: {run.root_directory}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
