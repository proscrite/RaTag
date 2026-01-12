#!/usr/bin/env python3
"""
Run analysis script - Execute complete RaTag pipeline from config file.

This script loads a YAML configuration file and runs the complete analysis
pipeline including preparation (S1/S2 timing) and integration (S2 areas).

Usage:
    # RECOMMENDED WORKFLOW (multi-isotope runs):
    # 1. Generate calibration with overlap-resolved ranges
    python scripts/run_analysis.py path/to/config.yaml --alphas-only
    # 2. Review plots, then run full pipeline (uses computed ranges)
    python scripts/run_analysis.py path/to/config.yaml
    
    # Full pipeline (single-isotope or using YAML ranges)
    python scripts/run_analysis.py path/to/config.yaml
    
    # Refit calibration without regenerating energy maps (.bin files)
    python scripts/run_analysis.py path/to/config.yaml --alphas-only --force-refit
    
    # Override: use YAML ranges instead of computed ranges
    python scripts/run_analysis.py path/to/config.yaml --use-yaml-ranges
    
    # Only preparation stage
    python scripts/run_analysis.py path/to/config.yaml --prepare-only
    
    # Only integration stage (assumes preparation done)
    python scripts/run_analysis.py path/to/config.yaml --recoil-only
    
    # Only X-ray classification (assumes preparation done)
    python scripts/run_analysis.py path/to/config.yaml --xray-only

Example:
    # Two-phase workflow (recommended for multi-isotope)
    python scripts/run_analysis.py ../configs/run18_analysis.yaml --alphas-only
    python scripts/run_analysis.py ../configs/run18_analysis.yaml
    
    # Update n_sigma and refit without regenerating energy maps
    python scripts/run_analysis.py ../configs/run18_analysis.yaml --alphas-only --force-refit
    
    # Edge case: fallback to YAML ranges if computed ranges fail
    python scripts/run_analysis.py ../configs/run18_analysis.yaml --use-yaml-ranges
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
from dataclasses import replace

from typing import Optional

from RaTag.core.datatypes import Run
from RaTag.core.config import IntegrationConfig, FitConfig, XRayConfig, AlphaCalibrationConfig
from RaTag.workflows.run_construction import initialize_run
from RaTag.workflows.spectrum_calibration import load_computed_ranges
from RaTag.pipelines.run_preparation import prepare_run, prepare_run_multiiso
from RaTag.pipelines.recoil_only import recoil_pipeline, recoil_pipeline_multiiso
from RaTag.pipelines.xray_only import xray_pipeline, xray_pipeline_multiiso
from RaTag.pipelines.alpha_calibration import alpha_calibration, alpha_calibration_singleiso
from RaTag.pipelines.unified_xray_and_recoil import unified_pipeline
from RaTag.pipelines.recombination_analysis import recombination_pipeline

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
        recoil_energy=exp.get('recoil_energy', 96.8),
        sets=[]
    )


def determine_isotope_ranges(run: Run, config: dict, use_yaml_override: bool) -> tuple[dict, bool]:
    """
    Determine which isotope ranges to use: computed or YAML.
    
    Parameters
    ----------
    run : Run
        Run object with run_id and root_directory
    config : dict
        Configuration dictionary with YAML ranges
    use_yaml_override : bool
        If True, force use of YAML ranges
        
    Returns
    -------
    isotope_ranges : dict
        Isotope ranges as {isotope: (E_min, E_max)}
    using_computed : bool
        True if using computed ranges, False if using YAML
    """
    # Check if computed ranges exist
    computed_ranges_available = False
    try:
        ranges_file = run.root_directory / 'processed_data' / 'spectrum_calibration' / f'{run.run_id}_isotope_ranges.npz'
        computed_ranges_available = ranges_file.exists()
    except Exception:
        pass
    
    if computed_ranges_available and not use_yaml_override:
        # Use computed ranges
        isotope_ranges = load_computed_ranges(run.run_id, run.root_directory)
        return isotope_ranges, True
    else:
        # Use YAML ranges
        isotope_ranges = {
            isotope: tuple(range_vals) 
            for isotope, range_vals in config['multi_isotope']['isotope_ranges'].items()
        }
        return isotope_ranges, False


def print_ranges_status(isotope_ranges: dict, using_computed: bool, use_yaml_override: bool, run_id: str):
    """Print status message about which ranges are being used."""
    if using_computed:
        print(f"\n{'='*70}")
        print("✓ Using COMPUTED overlap-resolved ranges (recommended)")
        print(f"  Source: {run_id}_isotope_ranges.npz")
        print(f"  Ranges: {isotope_ranges}")
        print(f"{'='*70}")
    elif use_yaml_override:
        print(f"\n{'='*70}")
        print("⚠️  Using YAML ranges (user override via --use-yaml-ranges)")
        print(f"  Ranges: {isotope_ranges}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("ℹ️  Using YAML ranges (no calibration found)")
        print(f"  Ranges: {isotope_ranges}")
        print(f"  💡 Tip: Run --alphas-only first for optimal overlap-resolved ranges")
        print(f"{'='*70}")


def prompt_verify_computed_ranges(run: Run, isotope_ranges: dict):
    """Interactive prompt to verify computed ranges before proceeding."""
    print(f"\n{'='*70}")
    print("⚠️  PROCEEDING WITH COMPUTED RANGES FOR DOWNSTREAM ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  📊 VERIFY CALIBRATION PLOTS:")
    print(f"     • processed_data/spectrum_calibration/{run.run_id}_calibration_summary.png")
    print(f"     • processed_data/spectrum_calibration/{run.run_id}_overlap_resolution.png")
    print(f"\n  ✓ Computed ranges (overlap-resolved):")
    for iso, rng in isotope_ranges.items():
        print(f"     {iso:8s}: [{rng[0]:.3f}, {rng[1]:.3f}] MeV")
    print(f"\n  ⚠️  These ranges will be used for alpha tagging in all downstream stages")
    print(f"     Press Enter to continue or Ctrl+C to abort...")
    print(f"{'='*70}")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Analysis aborted by user")
        raise SystemExit(0)
    print("✓ Continuing with full pipeline...\n")


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
    parser.add_argument('--recoil-only', action='store_true',
                       help='Only run integration (assumes preparation done)')
    parser.add_argument('--xray-only', action='store_true',
                       help='Only run X-ray classification pipeline')
    parser.add_argument('--force-refit', action='store_true',
                       help='Force recomputation of calibration/ranges/plots (energy maps cached)')
    parser.add_argument('--use-yaml-ranges', action='store_true',
                       help='Override: use YAML ranges instead of computed overlap-resolved ranges')
    parser.add_argument('--only-unified', action='store_true',
                       help='Run only the unified X-ray + S2 workflow (bypasses separate pipelines)')
    parser.add_argument('--recombination-only', action='store_true',
                       help='Only run recombination analysis (assumes prior stages done)')
    
    
    args = parser.parse_args()
    
    # Validate arguments - only one "only" flag allowed
    only_flags = [args.alphas_only, args.prepare_only, args.recoil_only, args.xray_only, args.only_unified, args.recombination_only]
    if sum(only_flags) > 1:
        parser.error("Cannot specify multiple --*-only flags")
    run_all = not any(only_flags)
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Check if multi-isotope mode
    is_multiiso = config.get('multi_isotope', {}).get('enabled', False)
    isotope_ranges = None
    
    stages = {
        'alphas': args.alphas_only or run_all,  # Run alphas for both modes when run_all or alphas_only
        'preparation': args.prepare_only or run_all,
        'recoil': args.recoil_only or run_all,
        'xray': args.xray_only or run_all,
        'recombination': args.recombination_only or run_all
    }
    
    # Log start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"STARTING ANALYSIS: {config['run_id']}")
    print(f"Time: {timestamp}")
    print(f"Config: {args.config}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}\n")

    # Create Run object
    run = create_run_from_config(config)
    print("\nInitializing run (populating sets from directory)...")    
    run = initialize_run(run, max_files=None)
    print(f"Found {len(run.sets)} sets")

    # ========================================================================
    # ALPHA ENERGY MAPPING AND CALIBRATION
    # ========================================================================
    if stages['alphas']:
        # Get energy mapping config and create calibration config
        energy_cfg = config.get('energy_mapping', {})
        calib_config = AlphaCalibrationConfig(
            files_per_chunk=energy_cfg.get('files_per_chunk', 10),
            fmt=energy_cfg.get('format', '8b'),
            scale=energy_cfg.get('scale', 0.1),
            pattern=energy_cfg.get('pattern', '*Ch4.wfm' if is_multiiso else '*.wfm'),
            nbins=energy_cfg.get('nbins', 120),
            n_sigma=energy_cfg.get('n_sigma', 1.0),
            use_quadratic=energy_cfg.get('use_quadratic', True)
        )
        
        if is_multiiso:
            # Run multi-isotope alpha calibration
            run = alpha_calibration(run,
                                   savgol_window=energy_cfg.get('savgol_window', 501),
                                   energy_range=tuple(energy_cfg.get('energy_range', [4, 8.2])),
                                   calibration_config=calib_config,
                                   force_refit=args.force_refit)
            
            # Reload computed ranges for downstream use
            isotope_ranges = load_computed_ranges(run.run_id, run.root_directory)
            print(f"\n✓ Calibration complete - computed ranges available for downstream stages")
        else:
            # Run single-isotope alpha monitoring
            alpha_calibration_singleiso(run,
                                       savgol_window=energy_cfg.get('savgol_window', 501),
                                       energy_range=tuple(energy_cfg.get('energy_range', [4, 8.2])),
                                       calibration_config=calib_config,
                                       force_refit=args.force_refit)
        
        if args.alphas_only:
            if is_multiiso:
                print(f"\n{'='*60}")
                print("STOPPING AFTER ALPHA CALIBRATION (--alphas-only flag)")
                print(f"  📊 Review plots in processed_data/spectrum_calibration/")
                print(f"  💡 Remove --alphas-only to continue with full pipeline using computed ranges")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print("STOPPING AFTER ALPHA MONITORING (--alphas-only flag)")
                print(f"  📊 Review alpha spectrum plots in processed_data/spectrum_calibration/")
                print(f"{'='*60}")
            return
    
    # For multi-isotope: determine ranges, setup verification
    if is_multiiso:
        if not stages['alphas']:
            # Determine which isotope ranges to use (if alphas stage was skipped)
            isotope_ranges, using_computed = determine_isotope_ranges(run, config, args.use_yaml_ranges)
            print_ranges_status(isotope_ranges, using_computed, args.use_yaml_ranges, run.run_id)
        
        # Interactive verification prompt when using computed ranges
        if isotope_ranges and not args.use_yaml_ranges and not args.alphas_only:
            prompt_verify_computed_ranges(run, isotope_ranges)
    
    # ========================================================================
    # PREPARATION STAGE
    # ========================================================================
    if stages['preparation']:
        print("\n" + "="*60)
        print("STAGE 1: PREPARATION")
        print("="*60)
        
        prep_cfg = config['pipeline']['preparation']
        
        if is_multiiso:
            assert isotope_ranges is not None, "isotope_ranges must be set for multi-isotope mode"
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
    # ========================================================================
    # PIPELINE SELECTION LOGIC
    # ========================================================================
    # The following blocks are mutually exclusive:
    # - If run_all (default) or --only-unified is set, run the unified pipeline and skip the old pipelines.
    # - If --recoil-only or --xray-only is set, run only the corresponding pipeline.
    # This prevents double processing of the same data.
    # ========================================================================
    if run_all or args.only_unified:
        print("\n" + "="*60)
        print("STAGE 2: UNIFIED X-RAY + S2 INTEGRATION")
        print("="*60)
        run = unified_pipeline(run,
                                max_frames=int_cfg['max_frames'],
                                integration_config=integration_config,
                                xray_config=xray_config,
                                fit_config=fit_config,
                                isotope_ranges=isotope_ranges,)
        print("\n✓ Unified integration complete")
        if args.only_unified:
            print(f"\n{'='*60}")
            print("STOPPING AFTER UNIFIED INTEGRATION (--only-unified flag)")
            print(f"{'='*60}")
            return
    

    # ========================================================================
    # INTEGRATION STAGE (only runs if unified pipeline is not selected)
    # ========================================================================
    elif stages['recoil']:
        print("\n" + "="*60)
        print("STAGE 2: RECOIL INTEGRATION")
        print("="*60)
        
        if is_multiiso:
            assert isotope_ranges is not None, "isotope_ranges must be set for multi-isotope mode"
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
        
        if args.recoil_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER INTEGRATION (--recoil-only flag)")
            print(f"{'='*60}")
            return
    
    # ========================================================================
    # X-RAY CLASSIFICATION STAGE (only runs if unified pipeline is not selected)
    # ========================================================================
    elif stages['xray']:
        print("\n" + "="*60)
        print("STAGE 3: X-RAY CLASSIFICATION")
        print("="*60)
        
        if is_multiiso:
            assert isotope_ranges is not None, "isotope_ranges must be set for multi-isotope mode"
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
    # RECOMBINATION ANALYSIS STAGE
    # ========================================================================
    if stages['recombination']:
        print("\n" + "="*60)
        print("STAGE 4: RECOMBINATION ANALYSIS")
        print("="*60)

        path_gs2 = config.get('recombination_analysis', {}).get('gs2_path', None)
        if path_gs2 is None:
            raise ValueError("gs2_path should be provided as a config parameter")
        df_recomb = recombination_pipeline(run, path_gs2=path_gs2)

        print("\n✓ Recombination analysis complete")
        if args.recombination_only:
            print(f"\n{'='*60}")
            print("STOPPING AFTER RECOMBINATION ANALYSIS (--recombination-only flag)")
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
