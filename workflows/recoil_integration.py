"""
Ion recoil S2 area integration and analysis workflow.

This module provides workflows for:
1. Complete set-level S2 integration workflow (ETL: extract → transform → load)
2. Run-level orchestration
3. Field-dependent summary plotting

Structure mirrors timing_estimation.py:
- Set-level workflows handle complete ETL including immediate persistence
- Run-level functions orchestrate with caching
- Plotting uses existing functions from plotting module
"""

from typing import Dict, Optional
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from RaTag.core.datatypes import SetPmt, S2Areas, Run
from RaTag.core.config import IntegrationConfig, FitConfig
from RaTag.core.dataIO import iter_frameproxies, store_s2area, load_s2area, save_figure, save_set_metadata
from RaTag.core.uid_utils import make_uid
from RaTag.core.fitting import fit_set_s2, fit_multiiso_s2
from RaTag.core.functional import apply_workflow_to_run, map_isotopes_in_run, compute_max_files
from RaTag.alphas.energy_join import  generic_multiiso_workflow
from RaTag.waveform.integration import integrate_s2_in_frame
from RaTag.plotting import plot_s2_vs_drift, plot_hist_fit, plot_grouped_histograms


# ============================================================================
# DIRECTORY MANAGEMENT HELPERS
# ============================================================================

def _setup_set_directories(set_pmt: SetPmt) -> tuple[Path, Path]:
    """
    Setup output directories for set-level processing.
    
    Args:
        set_pmt: Source set
        
    Returns:
        Tuple of (plots_dir, data_dir)
    """

    plots_dir = set_pmt.source_dir.parent / "plots" / "all" / "s2_areas"
    plots_dir.mkdir(parents=True, exist_ok=True)
    

    data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir


def _setup_run_directories(run: Run) -> tuple[Path, Path]:
    """
    Setup output directories for run-level processing.
    
    Args:
        run: Source run
        
    Returns:
        Tuple of (plots_dir, data_dir)
    """

    plots_dir = run.root_directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = run.root_directory / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir


def _extract_fit_metadata(fit_result: dict, isotope: str) -> dict:
    """
    Extract key fit metrics from fit result for metadata storage.
    
    Extracts only essential values:
    - mean (peak_position)
    - ci95 (95% confidence interval from peak_stderr)
    - lower_bound (only if two_stage method was used)
    
    Args:
        fit_result: Dictionary containing fit results
        isotope: Isotope name for key prefixing
        
    Returns:
        Dictionary with metadata keys ready for storage
    """
    mean = fit_result['peak_position']
    ci95 = 1.96 * fit_result['peak_stderr'] if fit_result['peak_stderr'] else 0.0
    
    metadata = {
        f'area_s2_{isotope}_mean': mean,
        f'area_s2_{isotope}_ci95': ci95
    }
    
    # Only store lower_bound if two_stage method was used
    if fit_result.get('method') == 'two_stage' and 'lower_bound' in fit_result:
        metadata[f'area_s2_{isotope}_lower_bound'] = fit_result['lower_bound']
    
    return metadata


def _integrate_s2_in_set(set_pmt: SetPmt,
                          max_frames: Optional[int],
                          integration_config: IntegrationConfig) -> S2Areas:
    """
    Integrate S2 areas for all frames in a set.
    
    Args:
        set_pmt: Source set
        max_frames: Optional limit on number of frames (None = process all)
        integration_config: Integration parameters
        
    Returns:
        S2Areas object with integrated areas
    """
    # Check preconditions
    if 't_s2_start' not in set_pmt.metadata or 't_s2_end' not in set_pmt.metadata:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing S2 window metadata")
    
    # Extract and ensure proper float type (metadata might store as string)
    s2_start = float(set_pmt.metadata['t_s2_start'])
    s2_end = float(set_pmt.metadata['t_s2_end'])
    
    print(f"  Integrating S2 window: [{s2_start:.2f}, {s2_end:.2f}] µs")
    
    # Compute how many files to process (rounds up to complete files)
    max_files, actual_frames = compute_max_files(max_frames, set_pmt.nframes)
    
    if max_frames is not None:
        print(f"  Processing {max_files} files (~{actual_frames} frames)")
    
    # Integrate all frames
    areas = []
    uids = []
    for frame_wf in iter_frameproxies(set_pmt, chunk_dir=None, max_files=max_files):
        try:
            uid = make_uid(frame_wf.file_seq, frame_wf.frame_idx)
            frame_pmt = frame_wf.load_pmt_frame()
            area = integrate_s2_in_frame(frame_pmt, s2_start, s2_end, integration_config)
            areas.append(area)
            uids.append(uid)
        except Exception as e:
            print(f"    ⚠ Frame integration failed: {e}")
            continue
    
    if len(areas) == 0:
        raise ValueError(f"No frames integrated successfully")
    
    print(f"    ✓ Integrated {len(areas)} frames")
    
    # Create S2Areas object
    s2 = S2Areas(source_dir=set_pmt.source_dir,
                 areas=np.array(areas).flatten(),
                 uids=np.array(uids).flatten(),
                 method="recoil_integration",
                 params={
                    "s2_window": (s2_start, s2_end),
                    "width_s2": s2_end - s2_start,
                    "n_pedestal": integration_config.n_pedestal,
                    "ma_window": integration_config.ma_window,
                    "bs_threshold": integration_config.bs_threshold,
                    "dt": integration_config.dt,
                }
            )
    return s2

# ============================================================================
# SET-LEVEL WORKFLOW (Complete ETL with immediate persistence)
# ============================================================================

def workflow_s2_integration(set_pmt: SetPmt,
                           max_frames: Optional[int] = None,
                           integration_config: IntegrationConfig = IntegrationConfig()) -> SetPmt:
    """
    Complete S2 integration workflow for a single set.
    
    1. Integrate S2 areas from all frames
    2. Save raw S2 areas to disk
    
    Note: Fitting is done separately via fit_s2_in_run()
    
    Prerequisites:
    - Set metadata must contain t_s2_start and t_s2_end
    
    Args:
        set_pmt: Set with S2 timing metadata
        max_frames: Optional limit on number of frames (None = process all)
        integration_config: Integration parameters
        plots_dir: Directory for plots (unused here, for compatibility)
        data_dir: Directory for S2 areas data
        
    Returns:
        SetPmt (unchanged - data saved to disk)
    """
    
    # Setup directories if not provided
    plot_dir, data_dir = _setup_set_directories(set_pmt)
    
    # Integrate
    s2 = _integrate_s2_in_set(set_pmt, max_frames=max_frames, 
                              integration_config=integration_config)

    # Save raw areas immediately (store_s2area prints the save message)
    store_s2area(s2, set_pmt=set_pmt, output_dir=data_dir)
    
    return set_pmt


def _fit_and_save_s2_histogram(set_pmt: SetPmt,
                               s2: S2Areas,
                               fit_config: FitConfig,
                               plots_dir: Path) -> SetPmt:
    """Fit S2 area distribution and save histogram plot."""

    # Fit using automatic method selection
    s2_fitted = fit_set_s2(s2,
                           bin_cuts=fit_config.bin_cuts,
                           nbins=fit_config.nbins,
                           flag_plot=False)
    
    if s2_fitted.fit_success:
        # Extract method info if available
        method = s2_fitted.fit_result.get('method', 'unknown') if isinstance(s2_fitted.fit_result, dict) else 'gaussian'
        print(f"    ✓ Fit ({method}): μ={s2_fitted.mean:.3f} ± {s2_fitted.ci95:.3f} mV·µs")
    else:
        print(f"    ✗ Fit failed")
    
    # Save histogram plot using the new plotting function
    fig, _ = plot_hist_fit(s2_fitted,
                           nbins=fit_config.nbins,
                           bin_cuts=fit_config.bin_cuts)
    
    save_figure(fig, plots_dir / f"{set_pmt.source_dir.name}_s2_histogram.png")
    plt.close(fig)  # Close figure to free memory
    print(f"    📊 Saved histogram plot")
    
    # Update metadata - include method info
    fit_metadata = {
        'area_s2_mean': s2_fitted.mean,
        'area_s2_ci95': s2_fitted.ci95,
        'area_s2_sigma': s2_fitted.sigma,
        'area_s2_fit_success': s2_fitted.fit_success
    }
    
    # Add method-specific metadata if available
    if isinstance(s2_fitted.fit_result, dict):
        fit_metadata['area_s2_fit_method'] = s2_fitted.fit_result.get('method', 'unknown')
        if s2_fitted.fit_result.get('method') == 'two_stage':
            fit_metadata['area_s2_bg_center'] = s2_fitted.fit_result.get('bg_center')
            fit_metadata['area_s2_lower_bound'] = s2_fitted.fit_result.get('lower_bound')
    
    new_metadata = {**set_pmt.metadata, **fit_metadata}
    updated_set = replace(set_pmt, metadata=new_metadata)
    
    # Save updated metadata
    from RaTag.core.dataIO import save_set_metadata
    save_set_metadata(updated_set)
    
    return updated_set


def workflow_s2_area_multiiso(set_pmt: SetPmt,
                              isotope_ranges: Dict[str, tuple]) -> pd.DataFrame:
    """Multi-isotope S2 area workflow: load → map → plot."""
    return generic_multiiso_workflow(set_pmt,
                                     data_filename="s2_areas.npz",
                                     value_keys=["s2_areas"],
                                     isotope_ranges=isotope_ranges,
                                     output_suffix="s2_areas_multi",
                                     plot_columns=["s2_areas"],
                                     bins=100)


def workflow_fit_multiiso_s2(set_pmt: SetPmt,
                              isotope_ranges: Dict[str, tuple],
                              fit_config: FitConfig = FitConfig()) -> Dict[str, dict]:
    """
    Fit S2 area distributions for each isotope and save plot with fits.
    
    Loads isotope-separated data, fits each isotope distribution, and creates
    a plot with histograms overlaid with fit curves.
    
    Args:
        set_pmt: Set with isotope-mapped data
        isotope_ranges: Dictionary of {isotope: (Emin, Emax)}
        fit_config: Fitting configuration
        
    Returns:
        Dictionary of {isotope: fit_result_dict}
    """
    
    # Load parquet file
    data_dir = set_pmt.source_dir.parent / "processed_data"
    multiiso_dir = data_dir / "multiiso"
    parquet_path = multiiso_dir / f"{set_pmt.source_dir.name}_s2_areas_multi.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Isotope data not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n  Fitting isotope distributions for {set_pmt.source_dir.name}...")
    
    fit_results = {}
    metadata_updates = {}
    
    for isotope in sorted(df['isotope'].unique()):
        try:
            result = fit_multiiso_s2(df, isotope,
                                     bin_cuts=fit_config.bin_cuts,
                                     nbins=fit_config.nbins,
                                     bg_cutoff=getattr(fit_config, 'bg_cutoff', 1.0),
                                     n_sigma=getattr(fit_config, 'n_sigma', 2.5))
            fit_results[isotope] = result
            
            # Extract key metrics for metadata storage
            isotope_metadata = _extract_fit_metadata(result, isotope)
            metadata_updates.update(isotope_metadata)
            
        except Exception as e:
            print(f"    ⚠ {isotope} fit failed: {e}")
            continue
    
    # Plot histograms with fit overlays and save metadata
    if fit_results:
        print(f"    ✓ Fitted {len(fit_results)} isotopes: {', '.join(fit_results.keys())}")
        
        # Save fit results to metadata
        updated_set = replace(set_pmt, metadata={**set_pmt.metadata, **metadata_updates})
        save_set_metadata(updated_set)
        
        # Generate plot with fit overlays
        plots_dir = set_pmt.source_dir.parent / "plots" / "multiiso" / "s2_areas_multi"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plot_grouped_histograms(df, ['s2_areas'], bins=100, fit_results=fit_results)
        save_figure(fig, plots_dir / f"{set_pmt.source_dir.name}_s2_areas_multi.png")
        plt.close(fig)
        print(f"    📊 Saved multi-isotope fit plot")
    
    return fit_results


# ============================================================================
# RUN-LEVEL ORCHESTRATION (with caching)
# ============================================================================

def integrate_s2_in_run(run: Run,
                       range_sets: slice = None,
                       max_frames: Optional[int] = None,
                       integration_config: IntegrationConfig = IntegrationConfig()) -> Run:
    """
    Integrate S2 areas for all sets (no fitting yet).
    
    Args:
        run: Run object with timing already estimated
        range_sets: Optional slice to process subset of sets
        max_frames: Optional limit on frames per set (None = process all)
        integration_config: Integration configuration
        
    Returns:
        Updated Run (data saved to disk)
    """
    
    # Filter sets if range specified
    if range_sets is not None:
        filtered_run = replace(run, sets=run.sets[range_sets])
    else:
        filtered_run = run
    
    return apply_workflow_to_run(filtered_run,
                                 workflow_func=workflow_s2_integration,
                                 workflow_name="S2 area integration",
                                 cache_key="area_s2_mean",
                                 data_file_suffix="s2_areas.npz",
                                 max_frames=max_frames,
                                 integration_config=integration_config)


def fit_s2_in_run(run: Run,
                  fit_config: FitConfig = FitConfig()) -> Run:
    """Fit Gaussian to S2 area distributions for all sets."""
    print("\n" + "="*60)
    print("S2 AREA FITTING")
    print("="*60)
    
    plots_dir = run.root_directory / "plots" / "all" / "s2_areas"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    updated_sets = []
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check cache
        if 'area_s2_mean' in set_pmt.metadata:
            print(f"  📂 Loaded from cache")
            updated_sets.append(set_pmt)
            continue
        
        # Load S2 areas
        data_dir = set_pmt.source_dir.parent / "processed_data"
        s2 = load_s2area(set_pmt, input_dir=data_dir)
        
        if s2 is None:
            print(f"  ⚠ No S2 areas found - run integration first")
            updated_sets.append(set_pmt)
            continue
        
        # Fit and save
        updated_set = _fit_and_save_s2_histogram(set_pmt, s2, fit_config, plots_dir)
        updated_sets.append(updated_set)
    
    return replace(run, sets=updated_sets)


def run_s2_area_multiiso(run: Run, 
                        isotope_ranges: dict) -> Run:
    """
    Run-level wrapper for distributing S2 areas by isotope.
    
    Maps S2 areas to isotopes and saves parquet files + plots.
    """
    return map_isotopes_in_run(run,
                               workflow_func=workflow_s2_area_multiiso,
                               workflow_name="S2 area isotope mapping",
                               isotope_ranges=isotope_ranges)


def fit_multiiso_s2_in_run(run: Run,
                           isotope_ranges: dict,
                           fit_config: FitConfig = FitConfig()) -> Run:
    """
    Run-level wrapper for fitting isotope-separated S2 distributions.
    
    Fits each isotope distribution and saves plots with fit overlays.
    """
    return map_isotopes_in_run(run,
                               workflow_func=workflow_fit_multiiso_s2,
                               workflow_name="Multi-isotope S2 fitting",
                               isotope_ranges=isotope_ranges,
                               fit_config=fit_config)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _collect_s2_data(run: Run) -> pd.DataFrame:
    """
    Collect S2 vs drift field data from set metadata.
    
    Args:
        run: Run with sets containing area_s2_mean in metadata
        
    Returns:
        DataFrame with columns: set_name, drift_field, s2_mean, s2_ci95, s2_sigma
    """
    data_rows = []
    for s in run.sets:
        # Check if integration results exist in metadata
        if 'area_s2_mean' not in s.metadata:
            continue
        
        if not s.metadata.get('area_s2_fit_success', False):
            continue
        
        data_rows.append({
            'set_name': s.source_dir.name,
            'drift_field': s.drift_field,
            's2_mean': s.metadata['area_s2_mean'],
            's2_ci95': s.metadata.get('area_s2_ci95', 0.0),
            's2_sigma': s.metadata.get('area_s2_sigma', 0.0)
        })
    
    df = pd.DataFrame(data_rows)
    return df.sort_values('drift_field') if len(df) > 0 else df


def _collect_multiiso_s2_data(run: Run, isotopes: list[str]) -> pd.DataFrame:
    """
    Collect per-isotope S2 vs drift field data from set metadata.
    
    Args:
        run: Run with sets containing area_s2_{isotope}_mean in metadata
        isotopes: List of isotope names to extract
        
    Returns:
        DataFrame with columns: set_name, drift_field, isotope, s2_mean, s2_ci95
    """
    data_rows = []
    for s in run.sets:
        drift_field = s.drift_field
        if drift_field is None:
            continue
        
        for isotope in isotopes:
            mean_key = f'area_s2_{isotope}_mean'
            ci95_key = f'area_s2_{isotope}_ci95'
            
            if mean_key in s.metadata and ci95_key in s.metadata:
                data_rows.append({
                    'set_name': s.source_dir.name,
                    'drift_field': drift_field,
                    'isotope': isotope,
                    's2_mean': s.metadata[mean_key],
                    's2_ci95': s.metadata[ci95_key]
                })
    
    df = pd.DataFrame(data_rows)
    return df.sort_values('drift_field') if len(df) > 0 else df


# ============================================================================
# RUN-LEVEL SUMMARY PLOTTING
# ============================================================================

def summarize_s2_vs_field(run: Run, 
                          plots_dir: Optional[Path] = None) -> Run:
    """
    Generate S2 area vs drift field summary plot and save data to CSV.
    """
    print("\n" + "="*60)
    print("FIELD-DEPENDENT ANALYSIS")
    print("="*60)
    
    # Setup directories
    plots_dir, data_dir = _setup_run_directories(run)
    
    # Collect data from set metadata
    df = _collect_s2_data(run)
    
    if len(df) == 0:
        print("  ⚠ No valid results to plot")
        return run
    
    # Save CSV
    csv_file = data_dir / f"{run.run_id}_s2_vs_drift.csv"
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"  💾 Saved data to {csv_file.name}")
    
    # Generate plot
    fig, _ = plot_s2_vs_drift(df, run.run_id)
    
    plot_file = plots_dir / f"{run.run_id}_s2_vs_drift.png"
    save_figure(fig, plot_file)
    plt.close(fig)  # Close figure to free memory
    print(f"  📊 Saved plot to {plot_file.name}")
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    • Sets with successful fits: {len(df)}")
    print(f"    • Drift field range: {df['drift_field'].min():.1f} - {df['drift_field'].max():.1f} V/cm")
    print(f"    • S2 mean range: {df['s2_mean'].min():.3f} - {df['s2_mean'].max():.3f} mV·µs")
    
    return run


def summarize_multiiso_s2_vs_field(run: Run, 
                                   isotopes: list[str],
                                   suffix: str = "") -> Run:
    """
    Generate multi-isotope S2 area vs drift field summary plot.
    
    Args:
        run: Run with per-isotope fit results in set metadata
        isotopes: List of isotope names to plot (e.g., ['Ra224', 'Th228'])
        suffix: Suffix for plot filename (e.g., '_all' or '_clear')
        
    Returns:
        Updated Run (plots saved to disk)
    """
    # Setup directories
    plots_dir = run.root_directory / "plots" / "multiiso"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect and plot isotopes
    df = _collect_multiiso_s2_data(run, isotopes)
    
    if len(df) == 0:
        print(f"  ⚠ No fit results found for isotopes: {', '.join(isotopes)}")
        return run
    
    # Create plot
    fig, _ = plot_s2_vs_drift(df, run.run_id, 
                             title_suffix=f" ({', '.join(isotopes)})", 
                             hue='isotope')
    
    # Save plot
    plot_file = plots_dir / f"{run.run_id}_multiiso_s2_vs_drift{suffix}.png"
    save_figure(fig, plot_file)
    plt.close(fig)
    print(f"  📊 Saved plot: {plot_file.name}")
    
    return run
