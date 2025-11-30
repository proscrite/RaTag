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
from RaTag.core.dataIO import iter_frameproxies, store_s2area, load_s2area, store_isotope_df, save_figure
from RaTag.core.uid_utils import make_uid
from RaTag.core.fitting import fit_set_s2
from RaTag.core.functional import map_over
from RaTag.core.energy_map_reader import get_energies_for_uids 
from RaTag.waveform.integration import integrate_s2_in_frame
from RaTag.plotting import plot_s2_vs_drift, plot_hist_fit, plot_grouped_histograms


# ============================================================================
# DIRECTORY MANAGEMENT HELPERS
# ============================================================================

def _setup_set_directories(set_pmt: SetPmt,
                          plots_dir: Optional[Path] = None,
                          data_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Setup output directories for set-level processing.
    
    Args:
        set_pmt: Source set
        plots_dir: Optional custom plots directory
        data_dir: Optional custom data directory
        
    Returns:
        Tuple of (plots_dir, data_dir)
    """
    if plots_dir is None:
        plots_dir = set_pmt.source_dir.parent / "plots" / "s2_histograms"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if data_dir is None:
        data_dir = set_pmt.source_dir.parent / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir


def _setup_run_directories(run: Run,
                          plots_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Setup output directories for run-level processing.
    
    Args:
        run: Source run
        plots_dir: Optional custom plots directory
        
    Returns:
        Tuple of (plots_dir, data_dir)
    """
    if plots_dir is None:
        plots_dir = run.root_directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = run.root_directory / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, data_dir


# ============================================================================
# SET-LEVEL WORKFLOW (Complete ETL with immediate persistence)
# ============================================================================

def workflow_s2_integration(set_pmt: SetPmt,
                           max_files: Optional[int] = None,
                           integration_config: IntegrationConfig = IntegrationConfig(),
                           fit_config: FitConfig = FitConfig(),
                           plots_dir: Optional[Path] = None,
                           data_dir: Optional[Path] = None,
                           isotope_ranges: Optional[Dict[str, tuple]] = None,
                           chunk_dir: Optional[str] = None) -> SetPmt:
    """
    Complete S2 integration workflow for a single set.
    
    1. Integrate S2 areas from all frames
    2. Fit Gaussian to distribution
    3. Save results + histogram plot immediately
    4. Store fit results in set metadata
    
    Prerequisites:
    - Set metadata must contain t_s2_start and t_s2_end
    
    Args:
        set_pmt: Set with S2 timing metadata
        max_files: Optional limit on number of files
        integration_config: Integration parameters
        fit_config: Fitting parameters
        plots_dir: Directory for histogram plots
        data_dir: Directory for S2 areas data
        
    Returns:
        SetPmt with updated metadata containing area_s2_mean, area_s2_ci95, area_s2_sigma
    """
    # Check preconditions
    if 't_s2_start' not in set_pmt.metadata or 't_s2_end' not in set_pmt.metadata:
        raise ValueError(f"Set {set_pmt.source_dir.name} missing S2 window metadata")
    
    s2_start = set_pmt.metadata['t_s2_start']
    s2_end = set_pmt.metadata['t_s2_end']
    
    print(f"  Integrating S2 window: [{s2_start:.2f}, {s2_end:.2f}] µs")
    
    # Setup directories
    plots_dir, data_dir = _setup_set_directories(set_pmt, plots_dir, data_dir)
    
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
    
    # Save raw areas immediately
    store_s2area(s2, set_pmt=set_pmt, output_dir=data_dir)
    print(f"    💾 Saved S2 areas to disk")
    
    # -------- NEW MULTI-ISOTOPE EXTENSION --------
    if hasattr(s2, "uids") and hasattr(s2, "areas") and isotope_ranges is not None:
        npz_path = data_dir / f"{set_pmt.source_dir.name}_s2area.npz"
        arr = np.load(npz_path, allow_pickle=True)

        df_area = map_results_to_isotopes(uids=arr["uids"],
                                          values=arr["areas"],
                                          chunk_dir=chunk_dir or str(set_pmt.source_dir),
                                          isotope_ranges=isotope_ranges,
                                          value_columns=["s2_area"])

        store_results_df(df_area, data_dir / f"{set_pmt.source_dir.name}_s2area_isotopes.parquet")
        plot_grouped_histograms(df_area, ["s2_area"], bins=40)
    # ----------------------------------------------

    # Fit Gaussian
    s2_fitted = fit_set_s2(s2,
                           bin_cuts=fit_config.bin_cuts,
                           nbins=fit_config.nbins,
                           exclude_index=fit_config.exclude_index,
                           flag_plot=False)
    
    if s2_fitted.fit_success:
        print(f"    ✓ Fit: μ={s2_fitted.mean:.3f} ± {s2_fitted.ci95:.3f} mV·µs")
    else:
        print(f"    ✗ Fit failed")
    
    # Save histogram plot
    fig, _ = plot_hist_fit(s2_fitted,
                           nbins=fit_config.nbins,
                           bin_cuts=fit_config.bin_cuts)
    
    save_figure(fig, plots_dir / f"{set_pmt.source_dir.name}_s2_histogram.png")
    plt.close(fig)  # Close figure to free memory
    print(f"    📊 Saved histogram plot")
    
    # Update metadata with fit results
    new_metadata = {
        **set_pmt.metadata,
        'area_s2_mean': s2_fitted.mean,
        'area_s2_ci95': s2_fitted.ci95,
        'area_s2_sigma': s2_fitted.sigma,
        'area_s2_fit_success': s2_fitted.fit_success
    }
    
    updated_set = replace(set_pmt, metadata=new_metadata)
    
    # Save updated metadata
    from RaTag.core.dataIO import save_set_metadata
    save_set_metadata(updated_set)
    
    return updated_set


# ============================================================================
# RUN-LEVEL ORCHESTRATION (with caching)
# ============================================================================

def integrate_s2_in_run(run: Run,
                       range_sets: slice = None,
                       max_files: Optional[int] = None,
                       integration_config: IntegrationConfig = IntegrationConfig(),
                       fit_config: FitConfig = FitConfig()) -> Run:
    """
    Integrate S2 areas for all sets in a run with caching.
    
    Args:
        run: Run with prepared sets (must have S2 window metadata)
        range_sets: Optional slice to select subset of sets
        max_files: Optional limit on files per set (testing)
        integration_config: Integration parameters
        fit_config: Fitting parameters
        
    Returns:
        Run with updated sets containing area_s2_mean in metadata
    """
    print("\n" + "="*60)
    print("S2 AREA INTEGRATION AND FITTING")
    print("="*60)
    
    # Setup directories
    plots_dir = run.root_directory / "plots" / "s2_histograms"
    data_dir = run.root_directory / "processed_data"
    
    # Select sets to process
    sets_to_process = run.sets[range_sets] if range_sets else run.sets
    
    # Process each set with caching
    updated_sets = []
    for i, set_pmt in enumerate(sets_to_process, 1):
        print(f"\nSet {i}/{len(sets_to_process)}: {set_pmt.source_dir.name}")
        
        # Check cache via metadata
        if 'area_s2_mean' in set_pmt.metadata:
            print(f"  📂 Loaded from cache")
            updated_sets.append(set_pmt)
            continue
        
        # Run complete workflow
        try:
            updated_set = workflow_s2_integration(set_pmt,
                                                 max_files=max_files,
                                                 integration_config=integration_config,
                                                 fit_config=fit_config,
                                                 plots_dir=plots_dir,
                                                 data_dir=data_dir)
            updated_sets.append(updated_set)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            updated_sets.append(set_pmt)
    
    print(f"\n✓ Integration complete: {len(updated_sets)}/{len(sets_to_process)} sets")
    
    return replace(run, sets=updated_sets)


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
    plots_dir, data_dir = _setup_run_directories(run, plots_dir)
    
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
