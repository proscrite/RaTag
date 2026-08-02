import numpy as np
from dataclasses import replace
import pandas as pd

from RaTag.core.datatypes import Run, SetPmt, SetAlpha
from RaTag.core.config import FitConfig

from RaTag.core.decorators import *
from RaTag.io import file_ops
from RaTag.plotting import plot_s2_vs_drift
from RaTag.core.paths import get_fit_path, get_output_root
from RaTag.plotting import build_fig_grid, plot_s2areas_summary
from RaTag.io.file_ops import load_s2areas, load_fit_result, save_figure
from RaTag.el_tpc.fit_s2_area import v_crystalball_right


# ============================================================================
# 1. SET-LEVEL SPAWNER
# ============================================================================
@allow_force
@load_cached_isotope_arrays(signal_type='s2_areas')
@write_isotope_arrays(signal_type='s2_areas')
def resolve_multiiso_separation(set_pmt: SetPmt, set_alpha: SetAlpha, force: bool = False) -> list[tuple[SetPmt, dict]]:
    """
    Worker function: Merges PMT areas and Alpha energies on UIDs, 
    slices by isotope ranges, and yields cloned sets and their arrays.
    """
    # 1. Load both arrays explicitly (since we need two different sets)
    pmt_arrays = file_ops.load_npz_arrays(set_pmt, 's2_areas')
    alpha_arrays = file_ops.load_npz_arrays(set_alpha, 'alpha_energies')

    # 2. Safely merge on UIDs
    df_pmt = pd.DataFrame({'s2_areas': pmt_arrays['s2_areas'], 'uids': pmt_arrays['uids']})
    df_alpha = pd.DataFrame({'energies': alpha_arrays['energies'], 'uids': alpha_arrays['uids']})
    df_merged = pd.merge(df_pmt, df_alpha, on='uids', how='inner')

    spawned_data = []
    
    # 3. Slice by isotope and spawn
    for isotope, (v_min, v_max) in set_alpha.isotope_ranges_V.items():
        mask = (df_merged['energies'] >= v_min) & (df_merged['energies'] <= v_max)
        df_iso = df_merged[mask]
        
        cloned_set = replace(set_pmt, target_isotope=isotope, multiiso=True,
                             area_s2_mean=None, area_s2_ci95=None, 
                             area_s2_sigma=None, area_s2_fit_success=None)
        iso_arrays = {
            'uids': df_iso['uids'].values,
            's2_areas': df_iso['s2_areas'].values,
        }
        
        spawned_data.append((cloned_set, iso_arrays))
        print(f"    ✓ Separated {isotope}: {len(df_iso)} events")
        
    return spawned_data


# ============================================================================
# 2. RUN-LEVEL ORCHESTRATOR
# ============================================================================

def map_multiiso_separation(run: Run, force: bool = False) -> dict[str, Run]:
    """
    Strictly zips PMT and Alpha sets together, separating the data and 
    spawning a distinct Run object for each isotope.
    """
    print(f"\n" + "="*60 + f"\nSEPARATING MULTI-ISOTOPE SETS: {run.run_id}\n" + "="*60)
    
    # Dictionary to hold the grouped sets: { 'Th228': [SetPmt, ...], 'Ra224': [...] }
    isotope_groups = {}
    
    # STRICT PAIRING BY DEFINITION
    for set_pmt, set_alpha in zip(run.sets, run.alpha_sets):
        print(f"\n  Mapping {set_pmt.source_dir.name}...")
        
        # The decorator returns a flat list of the spawned SetPmt clones
        spawned_sets = resolve_multiiso_separation(set_pmt, set_alpha, force=force)
        
        # Route each clone into its respective isotope group
        for s_pmt in spawned_sets:
            iso = s_pmt.target_isotope
            if iso not in isotope_groups:
                isotope_groups[iso] = []
            isotope_groups[iso].append(s_pmt)
            
    # Convert the grouped sets into fully independent Run objects
    spawned_runs = {}
    for iso, grouped_sets in isotope_groups.items():
        # Spawn a new Run, explicitly tagging the target isotope and assigning the subsets.
        iso_run_id = f"{run.run_id}_{iso}"
        spawned_runs[iso] = replace(run, run_id=iso_run_id,
                                    target_isotope=iso, 
                                    sets=grouped_sets, alpha_sets=[] )
        print(f"\n  ✓ Spawned new run object: {iso_run_id} with {len(grouped_sets)} sets.")
        
    return spawned_runs

# ============================================================================
# 3. Combined Plotter Orchestrator
# ============================================================================
@allow_force
@load_cached_plots(subfolder="s2_areas", expected_suffixes=["s2_vs_field_multiiso"])
@write_plots(subfolder="s2_areas")
def map_multiiso_s2_vs_field(bare_run: Run, spawned_runs: dict[str, Run], 
                             force: bool = False) -> tuple[Run, dict]:
    """
    Aggregates S2 metadata across all spawned isotope runs, builds a unified DataFrame,
    and plots the comparative S2 Area vs Drift Field.
    """
    print(f"\n" + "="*60 + f"\nPLOTTING MULTI-ISOTOPE S2 VS FIELD: {bare_run.run_id}\n" + "="*60)

    data = []
    for iso, iso_run in spawned_runs.items():
        for s_pmt in iso_run.sets:
            # Only include sets where the Crystal Ball fit succeeded
            if getattr(s_pmt, 'area_s2_fit_success', False):
                data.append({
                    'drift_field': s_pmt.drift_field,
                    's2_mean': s_pmt.area_s2_mean,
                    's2_ci95': s_pmt.area_s2_ci95,
                    'isotope': iso
                })

    if not data:
        print(f"  [Skip] No successful S2 fits found across isotopes for {bare_run.run_id}.")
        return bare_run, {}

    df = pd.DataFrame(data)

    # 2. Explicit Plotting Call (Depth of 1)
    fig, _ = plot_s2_vs_drift(df=df, 
                              run_id=bare_run.run_id, title_suffix=" (Multi-Isotope)", 
                              hue='isotope')

    # 3. Explicit I/O Routing
    df.to_csv(get_output_root(bare_run.root_directory) / "isotope_areas" / f"{bare_run.run_id}_s2_vs_field_multiiso.csv", index=False)
    return bare_run, {"s2_vs_field_multiiso": fig}


@allow_force
@load_cached_plots(subfolder="s2_areas", expected_suffixes=["histograms_multiiso"])
@write_plots(subfolder="s2_areas")
def map_multiiso_hist_grid(bare_run: Run, spawned_runs: dict[str, Run], 
                           config: FitConfig = FitConfig(), force: bool = False) -> tuple[Run, dict]:
    """
    Aggregates S2 area histograms from all spawned isotopes onto a single grid.
    Matches the physical grid of the bare run and overlays the isotope-specific data.
    """
    print(f"\n" + "="*60 + f"\nPLOTTING MULTI-ISOTOPE HISTOGRAMS: {bare_run.run_id}\n" + "="*60)
    
    # 1. Initialize the physical grid based on the unspawned run
    fig_hist, grid_cells = build_fig_grid(bare_run, f"Multi-Isotope S2 Areas - {bare_run.run_id}")
    
    # 2. Iterate through each physical detector setting
    for set_idx, (bare_set, ax) in enumerate(grid_cells):
        
        # 3. Iterate through each isotope for this specific physical setting
        for iso, iso_run in spawned_runs.items():
            
            # The sets in spawned_runs are strictly ordered to match bare_run
            iso_set = iso_run.sets[set_idx]
            
            # Only plot if the fit succeeded for this specific isotope
            if getattr(iso_set, 'area_s2_fit_success', False):
                
                # Load the isolated data and its specific fit model
                s2_areas = load_s2areas(iso_set)
                fit_path = get_fit_path(iso_set, 's2_areas_hist_fit')
                fit_model = load_fit_result(fit_path, funcdefs={'v_crystalball_right': v_crystalball_right})
                
                # Plot onto the shared axis using the base hardware name for the title
                plot_s2areas_summary(ax=ax,
                                     set_name=bare_set.name, 
                                     s2_areas=s2_areas,
                                     bin_cuts=config.bin_cuts,
                                     fit_model=fit_model,
                                     lower_bound=None,
                                     target_isotope=iso)

    return bare_run, {"histograms_multiiso": fig_hist}
    # # 4. Explicit I/O Routing (Bypassing decorators for cross-run aggregation)
    # out_dir = get_output_root(bare_run.root_directory) / "plots" / "s2_areas" / "multi"
    # out_dir.mkdir(parents=True, exist_ok=True)
    
    # out_path = out_dir / f"{bare_run.run_id}_histograms_multiiso.png"
    # save_figure(fig_hist, out_path)