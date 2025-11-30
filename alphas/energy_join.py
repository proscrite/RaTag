# RaTag/core/energy_join.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

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

    # Build dataframe
    df = pd.DataFrame({"uid": uids, "isotope": isotopes})

    # If `values` is 1-column
    if values.ndim == 1:
        df[value_columns[0]] = values
    # If multi-column (S2 start & end)
    else:
        for i, col in enumerate(value_columns):
            df[col] = values[:, i]

    # Keep only isotope-assigned rows
    df = df[df["isotope"] != ""].reset_index(drop=True)

    return df
