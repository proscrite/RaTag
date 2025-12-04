#!/usr/bin/env python3
"""
Refit S2 area distributions from cached NPZ files.

This script loads a run configuration and refits the S2 area histograms
without reintegrating from raw waveforms. Useful after:
- Patching data units
- Changing fit parameters (bin_cuts, nbins, etc.)
- Regenerating plots with new settings

The script now uses automatic method selection:
- Simple Crystal Ball fit for clean distributions
- Two-stage fit (background subtraction + Crystal Ball) for bimodal distributions

Usage:
    python scripts/refit_s2_areas.py path/to/config.yaml [OPTIONS]
    
Example:
    # Use fit config from YAML
    python scripts/refit_s2_areas.py ../configs/run18_analysis.yaml
    
    # Override bin cuts
    python scripts/refit_s2_areas.py ../configs/run18_analysis.yaml --bin-cuts 0 5
    
    # Adjust background detection threshold
    python scripts/refit_s2_areas.py ../configs/run18_analysis.yaml --bg-threshold 0.2
    
    # Change signal region cutoff (default 2.5 sigma)
    python scripts/refit_s2_areas.py ../configs/run18_analysis.yaml --n-sigma 3.0
"""

import argparse
import yaml
from pathlib import Path

from RaTag.core.datatypes import Run
from RaTag.core.config import FitConfig
from RaTag.workflows.run_construction import initialize_run
from RaTag.pipelines.recoil_only import recoil_pipeline_replot


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
        description='Refit S2 areas from cached NPZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('config', type=Path, help='Path to YAML config file')
    parser.add_argument('--bin-cuts', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Override histogram bin range (mV·µs)')
    parser.add_argument('--nbins', type=int,
                       help='Override number of histogram bins')
    parser.add_argument('--exclude-index', type=int,
                       help='Override bin index to exclude from fit (deprecated)')
    parser.add_argument('--bg-threshold', type=float,
                       help='Background fraction threshold for two-stage method (0-1)')
    parser.add_argument('--bg-cutoff', type=float,
                       help='Upper limit for background fitting (mV·µs)')
    parser.add_argument('--n-sigma', type=float,
                       help='Sigmas above background for signal region cutoff')
    parser.add_argument('--upper-limit', type=float,
                       help='Upper limit for signal fitting (mV·µs)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create Run object
    run = create_run_from_config(config)
    print(f"\nInitializing run: {run.run_id}")
    run = initialize_run(run, max_files=None)
    print(f"Found {len(run.sets)} sets")
    
    # Create FitConfig (from YAML or command line overrides)
    fit_cfg = config['pipeline']['integration']['fit_config']
    
    bin_cuts = tuple(args.bin_cuts) if args.bin_cuts else tuple(fit_cfg['bin_cuts'])
    nbins = args.nbins if args.nbins else fit_cfg['nbins']
    bg_threshold = args.bg_threshold if args.bg_threshold is not None else fit_cfg.get('bg_threshold', 0.3)
    bg_cutoff = args.bg_cutoff if args.bg_cutoff is not None else fit_cfg.get('bg_cutoff', 1.0)
    n_sigma = args.n_sigma if args.n_sigma is not None else fit_cfg.get('n_sigma', 2.5)
    upper_limit = args.upper_limit if args.upper_limit is not None else fit_cfg.get('upper_limit', 5.0)
    
    fit_config = FitConfig(
        bin_cuts=bin_cuts,
        nbins=nbins,
        bg_threshold=bg_threshold,
        bg_cutoff=bg_cutoff,
        n_sigma=n_sigma,
        upper_limit=upper_limit
    )
    
    print(f"\nFit configuration:")
    print(f"  bin_cuts: {fit_config.bin_cuts} mV·µs")
    print(f"  nbins: {fit_config.nbins}")
    print(f"  bg_threshold: {fit_config.bg_threshold} (background detection)")
    print(f"  bg_cutoff: {fit_config.bg_cutoff} mV·µs")
    print(f"  n_sigma: {fit_config.n_sigma} (signal region cutoff)")
    print(f"  upper_limit: {fit_config.upper_limit} mV·µs")
    
    # Run refit pipeline
    print("\n" + "="*60)
    print("STARTING REFIT PIPELINE")
    print("="*60)
    
    run = recoil_pipeline_replot(run, fit_config=fit_config)
    
    print("\n" + "="*60)
    print("REFIT COMPLETE")
    print(f"Results saved in: {run.root_directory}")
    print("="*60)


if __name__ == "__main__":
    main()
