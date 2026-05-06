import pandas as pd
import numpy as np
from pathlib import Path

def extract_hidex_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Reads a Hidex AMG Excel file and extracts both the vial information
    and the corresponding 2048-channel gamma spectra in a fully vectorized way.
    
    Returns a DataFrame where each valid vial has its metadata and a 'Spectrum' 
    column containing the 1D numpy array of counts.
    """
    
    # 1. Read the Results Block (starting at Excel row 27)
    results_df = pd.read_excel(filepath, sheet_name="Results", 
                               usecols="A:E", skiprows=26, header=None, engine="openpyxl")
    results_df.columns = ["ColA", "Datetime", "ParamC", "CountTime_mins", "ParamE"]

    # 2. Read the full Spectra sheet (no skiprows) so we can compute absolute
    # Excel row mappings robustly. We'll extract the C:BZV columns later.
    spectra_full = pd.read_excel(filepath, sheet_name="Spectra", header=None, engine="openpyxl")

    # Compute the Excel row numbers corresponding to each row we read from
    # the Results block. results_df.iloc[0] corresponds to Excel row 27.
    results_excel_row0 = 27
    results_count = len(results_df)
    results_excel_rows = np.arange(results_excel_row0, results_excel_row0 + results_count)

    # The Spectra block's corresponding Excel row for a given Results row is
    # (results_excel_row - 7) according to the instrument export layout
    # (so Results row 27 -> Spectra row 20). Convert to 0-based indices for
    # selecting from spectra_full.
    spectra_excel_rows = results_excel_rows - 7
    spectra_indices0 = spectra_excel_rows - 1  # 0-based

    # Keep only those Results rows whose 'ColA' appears numeric (vial numbers)
    # to exclude header-like text rows such as 'Results' or 'Rack'. Use
    # to_numeric with errors='coerce' so non-numeric become NaN.
    col_a_numeric = pd.to_numeric(results_df["ColA"], errors="coerce")
    numeric_mask = col_a_numeric.notna().to_numpy()

    # Ensure spectra indices are within bounds of the full spectra sheet
    spectra_in_bounds = (spectra_indices0 >= 0) & (spectra_indices0 < len(spectra_full))

    select_mask = numeric_mask & spectra_in_bounds
    if not select_mask.any():
        out_cols = list(results_df.columns) + ["Spectrum", "SourceFile"]
        return pd.DataFrame(columns=out_cols)

    # Filter Results rows and corresponding spectra rows
    sel_positions = np.flatnonzero(select_mask)
    valid_results = results_df.iloc[sel_positions].copy().reset_index(drop=True)
    selected_spectra_rows = spectra_indices0[sel_positions].astype(int)

    # Extract channels C (0-based col 2) through BZV (col 2049) from spectra_full
    # for the selected rows. If the full sheet has fewer than 2050 columns, trim
    # accordingly.
    max_col = min(spectra_full.shape[1], 2050)
    spectra_values = spectra_full.iloc[selected_spectra_rows, 2:max_col].astype(float).values

    # Drop any rows with NaNs in the spectra_values
    nan_mask = np.isnan(spectra_values).any(axis=1)
    if nan_mask.any():
        keep_idx = np.flatnonzero(~nan_mask)
        valid_results = valid_results.iloc[keep_idx].reset_index(drop=True)
        spectra_values = spectra_values[keep_idx]

    valid_results["Spectrum"] = [row for row in spectra_values]
    valid_results = valid_results.iloc[:-1]
    
    # Calculate count time in seconds
    valid_results["CountTime_sec"] = pd.to_numeric(valid_results["CountTime_mins"], errors="coerce") * 60.0

    return valid_results