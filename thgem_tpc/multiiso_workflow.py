import numpy as np
from dataclasses import replace
import pandas as pd

from RaTag.core.datatypes import Run, SetPmt, SetAlpha
from RaTag.core.decorators import *
from RaTag.alphas.energy_map_reader import get_energies_for_uids
from RaTag.io import file_ops


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
        
        cloned_set = replace(set_pmt, target_isotope=isotope, multiiso=True)
        iso_arrays = {
            'uids': df_iso['uids'].values,
            's2_areas': df_iso['s2_areas'].values
        }
        
        spawned_data.append((cloned_set, iso_arrays))
        print(f"    ✓ Separated {isotope}: {len(df_iso)} events")
        
    return spawned_data


# ============================================================================
# 2. RUN-LEVEL ORCHESTRATOR
# ============================================================================

def map_multiiso_separation(run: Run, force: bool = False) -> Run:
    """
    Strictly zips PMT and Alpha sets together, separating the data and 
    flattening the spawned subset results back into run.sets.
    """
    print(f"\n" + "="*60 + f"\nSEPARATING MULTI-ISOTOPE SETS: {run.run_id}\n" + "="*60)
    
    all_spawned_sets = []
    
    # STRICT PAIRING BY DEFINITION
    for set_pmt, set_alpha in zip(run.sets, run.alpha_sets):
        print(f"\n  Mapping {set_pmt.source_dir.name}...")
        
        spawned_subsets = resolve_multiiso_separation(set_pmt, set_alpha, force=force)
        all_spawned_sets.extend(spawned_subsets)
        
    # Overwrite the run's sets with the flattened list of cloned, isotope-specific sets
    return replace(run, sets=all_spawned_sets)