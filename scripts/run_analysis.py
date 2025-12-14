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
    
    # Only X-ray classification (assumes preparation done)
    python scripts/run_analysis.py path/to/config.yaml --xray-only

Example:
    python scripts/run_analysis.py ../configs/run8_analysis.yaml
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime

from RaTag.core.datatypes import Run
from RaTag.core.config import IntegrationConfig, FitConfig, XRayConfig, AlphaCalibrationConfig
from RaTag.workflows.run_construction import initialize_run
from RaTag.pipelines.run_preparation import prepare_run, prepare_run_multiiso
from RaTag.pipelines.recoil_only import recoil_pipeline, recoil_pipeline_multiiso
from RaTag.pipelines.xray_only import xray_pipeline, xray_pipeline_multiiso
from RaTag.pipelines.alpha_calibration import alpha_calibration


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
    parser.add_argument('--xray-only', action='store_true',
                       help='Only run X-ray classification pipeline')
    
    
    args = parser.parse_args()
    
    # Validate arguments - only one "only" flag allowed
    only_flags = [args.alphas_only, args.prepare_only, args.integrate_only, args.xray_only]
    if sum(only_flags) > 1:
        parser.error("Cannot specify multiple --*-only flags")
    
    # Determine which stages to run
    run_all = not any(only_flags)  # If no flags, run everything
    
    stages = {
        'alphas': args.alphas_only or (run_all and False),  # Only when multiiso AND (flag OR run_all)
        'preparation': args.prepare_only or run_all,
        'integration': args.integrate_only or run_all,
        'xray': args.xray_only or run_all
    }
    
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

    # ========================================================================
    # ALPHA ENERGY MAPPING (if multi-isotope enabled)
    # ========================================================================
    if is_multiiso:
        # Convert isotope ranges from list to tuple
        isotope_ranges = {
            isotope: tuple(range_vals) 
            for isotope, range_vals in config['multi_isotope']['isotope_ranges'].items()
        }
        print(f"Multi-isotope mode enabled with ranges: {isotope_ranges}")
        
        # Update stages to enable alphas if multiiso and run_all
        if run_all:
            stages['alphas'] = True
        
        if stages['alphas']:
            # Generate energy maps and calibration if configured
            energy_cfg = config['multi_isotope'].get('energy_mapping', {})
            if energy_cfg.get('generate', True):
                # Create calibration config from YAML
                calib_config = AlphaCalibrationConfig(
                    files_per_chunk=energy_cfg.get('files_per_chunk', 10),
                    fmt=energy_cfg.get('format', '8b'),
                    scale=energy_cfg.get('scale', 0.1),
                    pattern=energy_cfg.get('pattern', '*Ch4.wfm'),
                    nbins=energy_cfg.get('nbins', 120),
                    n_sigma=energy_cfg.get('n_sigma', 1.0),
                    use_quadratic=energy_cfg.get('use_quadratic', True)
                )
                
                run = alpha_calibration(run,
                                       savgol_window=energy_cfg.get('savgol_window', 501),
                                       energy_range=tuple(energy_cfg.get('energy_range', [4, 8.2])),
                                       calibration_config=calib_config)
            
            if args.alphas_only:
                print(f"\n{'='*60}")
                print("STOPPING AFTER ALPHA CALIBRATION (--alphas-only flag)")
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
    if stages['preparation']:
        print("\n" + "="*60)
        print("STAGE 1: PREPARATION")
        print("="*60)
        
        prep_cfg = config['pipeline']['preparation']
        
        if is_multiiso:
            run = prepare_run_multiiso(run,
                                       isotope_ranges=isotope_ranges,
                                       max_files=None,
                                       max_frames_s1=int(prep_cfg['max_frames_s1']),
                                       max_frames_s2=int(prep_cfg['max_frames_s2']),
                                       threshold_s1=float(prep_cfg['threshold_s1']),
                                       threshold_s2=float(prep_cfg['threshold_s2']),
                                       s2_duration_cuts=tuple(prep_cfg['s2_duration_cuts']))
        else:
            run = prepare_run(run,
                            max_files=None,
                            max_frames_s1=int(prep_cfg['max_frames_s1']),
                            max_frames_s2=int(prep_cfg['max_frames_s2']),
                            threshold_s1=float(prep_cfg['threshold_s1']),
                            threshold_s2=float(prep_cfg['threshold_s2']),
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
    if stages['integration']:
        print("\n" + "="*60)
        print("STAGE 2: INTEGRATION")
        print("="*60)
        
        int_cfg = config['pipeline']['integration']
        
        # Create config objects
        integration_config = IntegrationConfig(
            n_pedestal=int(int_cfg['integration_config']['n_pedestal']),
            ma_window=int(int_cfg['integration_config']['ma_window']),
            bs_threshold=float(int_cfg['integration_config']['bs_threshold']),
            dt=float(int_cfg['integration_config']['dt'])
        )
        
        fit_config = FitConfig(
            bin_cuts=tuple(int_cfg['fit_config']['bin_cuts']),
            nbins=int(int_cfg['fit_config']['nbins']),
            bg_threshold=float(int_cfg['fit_config'].get('bg_threshold', 0.3)),
            bg_cutoff=float(int_cfg['fit_config'].get('bg_cutoff', 1.0)),
            n_sigma=float(int_cfg['fit_config'].get('n_sigma', 2.5)),
            upper_limit=float(int_cfg['fit_config'].get('upper_limit', 5.0))
        )

        
        if is_multiiso:
            run = recoil_pipeline_multiiso(run,
                                          isotope_ranges=isotope_ranges,
                                          max_frames=int_cfg['max_frames'],
                                          integration_config=integration_config,
                                          fit_config=fit_config)
        else:
            run = recoil_pipeline(run,
                                max_frames=int_cfg['max_frames'],
                                integration_config=integration_config,
                                fit_config=fit_config)
        
        print("\n✓ Integration complete")
        
        if args.integrate_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER INTEGRATION (--integrate-only flag)")
            print(f"{'='*60}")
            return
    
    
    # ========================================================================
    # X-RAY CLASSIFICATION STAGE
    # ========================================================================
    if stages['xray']:
        print("\n" + "="*60)
        print("STAGE 3: X-RAY CLASSIFICATION")
        print("="*60)
        
        xray_cfg = config['pipeline'].get('xray_classification', {})
        
        # Create XRayConfig
        xray_config = XRayConfig(
            bs_threshold=float(xray_cfg.get('bs_threshold', 0.5)),
            max_area_s1=float(xray_cfg.get('max_area_s1', 1e5)),
            max_area_s2=float(xray_cfg.get('max_area_s2', 1e5)),
            min_xray_area=float(xray_cfg.get('min_xray_area', 0.0)),
            min_s2_sep=float(xray_cfg.get('min_s2_sep', 1.0)),
            min_s1_sep=float(xray_cfg.get('min_s1_sep', 0.5)),
            n_pedestal=int(xray_cfg.get('n_pedestal', 200)),
            ma_window=int(xray_cfg.get('ma_window', 10)),
            dt=float(xray_cfg.get('dt', 2e-4))
        )
        
        if is_multiiso:
            run = xray_pipeline_multiiso(run,
                                         isotope_ranges=isotope_ranges,
                                         max_frames=xray_cfg.get('max_frames'),
                                         xray_config=xray_config)
        else:
            run = xray_pipeline(run,
                               max_frames=xray_cfg.get('max_frames'),
                               xray_config=xray_config)
        
        print("\n✓ X-ray classification complete")
        
        if args.xray_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER X-RAY CLASSIFICATION (--xray-only flag)")
            print(f"{'='*60}")
            return
    
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
