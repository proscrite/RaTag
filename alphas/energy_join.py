# RaTag/core/energy_join.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pathlib import Path

from RaTag.alphas.energy_map_reader import load_energy_index, get_energies_for_uids  # returns (ids_sorted, Es_sorted)

def map_uids_to_energies(uids, map_dir, fmt='8b', scale=0.1):
    """
    uids: 1D numpy array uint32
    returns: energies array same shape (float32) with np.nan for missing entries
    """
    ids, Es = load_energy_index(map_dir, fmt=fmt, scale=scale)  # cached in memory
    i = np.searchsorted(ids, uids)
    out = np.full(uids.shape, np.nan, dtype=np.float32)
    # need to guard for i == len(ids) and mismatches
    mask = (i < len(ids)) & (ids[i] == uids)
    out[mask] = Es[i[mask]]
    return out

def assign_uids_to_isotopes(uids, map_dir, isotope_ranges, fmt='8b', scale=0.1):
    """
    Returns dict isotope -> indices (into uids) that belong to that isotope.
    Overlaps: if a uid's energy falls into multiple ranges, it will be included in multiple lists.
    """
    Es = map_uids_to_energies(uids, map_dir, fmt=fmt, scale=scale)
    
    assignments = {}
    for name, (emin, emax) in isotope_ranges.items():
        mask = (~np.isnan(Es)) & (Es >= emin) & (Es <= emax)
        assignments[name] = np.nonzero(mask)[0]   # indices into the uids array
    return assignments

def assign_isotope_by_energy(E: float, ranges: Dict[str, Tuple[float,float]], min_prob=0.5) -> Optional[str]:
    """
    Simple deterministic mapping: returns the first isotope whose energy range contains E.
    ranges: {"Th228": (Emin, Emax), "Ra226": (...)}
    Returns isotope name or None if none matched.
    """
    if E is None or (not (E==E)):  # nan check
        return None
    for name, (emin, emax) in ranges.items():
        if (E >= emin) and (E <= emax):
            return name
    return None

def map_results_to_isotopes(
        uids: np.ndarray,
        values: np.ndarray,
        chunk_dir: str,
        isotope_ranges: dict,
        value_columns: list[str],
    ) -> pd.DataFrame:
    """
    Generic UID → isotope → DataFrame assignment.

    Parameters
    ----------
    uids : np.ndarray
        Array of uint32 UIDs.
    values : np.ndarray
        1D or 2D array of measurement values.
    chunk_dir : str
        Directory containing energy .bin chunk files.
    isotope_ranges : dict
        {isotope_name: (Emin, Emax)}.
    value_columns : list[str]
        Column names that correspond to `values` columns.
    
    Returns
    -------
    pd.DataFrame with columns: ['uid', 'isotope'] + value_columns
    """
    # Get energies
    energies = get_energies_for_uids(uids, chunk_dir=chunk_dir)
    energies = np.array(energies)

    # Assign isotope labels
    isotopes = np.full_like(energies, fill_value="", dtype=object)
    for iso, (emin, emax) in isotope_ranges.items():
        mask = (energies >= emin) & (energies <= emax)
        isotopes[mask] = iso

    # Filter to only isotope-assigned entries BEFORE creating DataFrame
    assigned_mask = isotopes != ""
    filtered_uids = uids[assigned_mask]
    filtered_isotopes = isotopes[assigned_mask]
    
    # Filter values using the same mask
    if values.ndim == 1:
        filtered_values = values[assigned_mask]
    else:
        filtered_values = values[assigned_mask, :]

    # Build dataframe with filtered data
    df = pd.DataFrame({"uid": filtered_uids, "isotope": filtered_isotopes})

    # Add value columns
    if values.ndim == 1:
        df[value_columns[0]] = filtered_values
    else:
        for i, col in enumerate(value_columns):
            df[col] = filtered_values[:, i]

    return df


def generic_multiiso_workflow(set_pmt,
                              data_filename: str,
                              value_keys: list[str],
                              isotope_ranges: dict,
                              output_suffix: str,
                              plot_columns: list[str],
                              bins: int = 100):
    """
    Generic multi-isotope workflow: load NPZ → map → save → plot.
    
    Abstracts the common pattern in all workflow_xxx_multiiso functions.
    
    Args:
        set_pmt: SetPmt object
        data_filename: NPZ filename (e.g., "s1.npz", "s2.npz", "s2area.npz", "xray_areas.npz")
        value_keys: Keys in NPZ for values (e.g., ["t_s1"], ["areas"], ["t_s2_start", "t_s2_end"])
        isotope_ranges: {isotope: (Emin, Emax)}
        output_suffix: Output filename suffix (e.g., "s1_isotopes", "s2area_isotopes")
        plot_columns: Columns to plot in grouped histograms
        bins: Number of bins for histograms
        
    Returns:
        DataFrame with isotope assignments
        
    Example:
        >>> df = generic_multiiso_workflow(
        ...     set_pmt,
        ...     data_filename="s1.npz",
        ...     value_keys=["t_s1"],
        ...     isotope_ranges={"Th228": (5000, 7000)},
        ...     output_suffix="s1_isotopes",
        ...     plot_columns=["t_s1"]
        ... )
    """
    from RaTag.core.dataIO import store_isotope_df, save_figure
    from RaTag.plotting import plot_grouped_histograms
    
    # Setup directories
    data_dir = set_pmt.source_dir.parent / "processed_data"
    plots_dir = set_pmt.source_dir.parent / "plots" / "multiiso" / f"{output_suffix}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NPZ from all/ subdirectory
    all_dir = data_dir / "all"
    npz_path = all_dir / f"{set_pmt.source_dir.name}_{data_filename}"
    arr = np.load(npz_path, allow_pickle=True)
    print(f"  Loaded data from all/{npz_path.name}")
    # Extract values
    if len(value_keys) == 1:
        values = arr[value_keys[0]]
    else:
        values = np.column_stack([arr[key] for key in value_keys])
    
    # Map to isotopes using energy maps from energy_maps/ directory
    energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
    df = map_results_to_isotopes(uids=arr["uids"],
                                 values=values,
                                 chunk_dir=str(energy_maps_dir),
                                 isotope_ranges=isotope_ranges,
                                 value_columns=value_keys)
    
    # Save DataFrame to multiiso/ subdirectory
    multiiso_dir = data_dir / "multiiso"
    multiiso_dir.mkdir(parents=True, exist_ok=True)
    output_path = multiiso_dir / f"{set_pmt.source_dir.name}_{output_suffix}.parquet"
    store_isotope_df(df, output_path)
    print(f"  💾 Saved isotope data to multiiso/{output_path.name}")
    
    # Plot grouped histograms
    fig = plot_grouped_histograms(df, plot_columns, bins=bins)
    plot_path = plots_dir / f"{set_pmt.source_dir.name}_{output_suffix}.png"
    save_figure(fig, plot_path)
    plt.close(fig)
    print(f"  📊 Saved plot to {plot_path.name}")
    
    return df
