import pandas as pd
import re
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from datetime import datetime

from .hidex_read import extract_hidex_raw_data
from .etl_hidex import transform_spectra_to_rates, fit_initial_populations, parse_timestamps, backcalculate_accumulation, LAMBDA_RA, LAMBDA_PB

# Th-228 Half-life in days
TH228_HALF_LIFE_DAYS = 1.9116 * 365.25
LAMBDA_TH228_DAYS = np.log(2) / TH228_HALF_LIFE_DAYS
LAMBDA_RA224_SEC = np.log(2) / (3.6319 * 24 * 3600)

def ingest_hidex_directory(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Scans a directory for Hidex files, extracts the raw arrays, and maps 
    them to a clean batch identifier.
    
    Returns:
        Dict[str, pd.DataFrame]: A collection mapping batch names (e.g., 'Run19_V1') 
                                 to their raw DataFrames containing the spectra.
    """
    raw_batches = {}
    files = sorted(data_dir.glob("HidexAMG-*.xlsx"))
    
    if not files:
        print(f"Warning: No HidexAMG files found in {data_dir}")
        return raw_batches

    for filepath in files:
        # 1. Extract the raw dataframe using the existing function. This is where we read the Excel and get the 'Spectrum' arrays.
        raw_df = extract_hidex_raw_data(filepath)
        if raw_df.empty:
            continue
            
        # 2. Parse out the ugly filename to get a clean batch ID
        clean_name = re.sub(r"^HidexAMG-", "", filepath.stem)
        match = re.search(r".*_.*_V\d+-\d+", clean_name)
        batch_id = match.group(0) if match else clean_name
        
        raw_batches[batch_id] = raw_df
        
    return raw_batches

def fit_batch_gamma_spectra(raw_batches: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Iterates over a collection of raw batches and maps the 240 keV peak 
    fitting algorithm across every spectrum.
    
    Returns:
        Dict[str, pd.DataFrame]: A new collection where the raw 'Spectrum' arrays 
                                 have been replaced by extracted physics metrics 
                                 (A, mu, sigma, RateError, etc.).
    """
    rate_batches = {}
    
    for batch_id, raw_df in raw_batches.items():
        # Apply the pure transformation function
        # This executes the curve_fit across the whole dataframe
        rate_batches[batch_id] = transform_spectra_to_rates(raw_df)
        
    return rate_batches

def export_batch_artifacts(batches: Dict[str, pd.DataFrame], out_path: Path, suffix: str = ""):
    """Handles the side-effect of saving the dataframes to disk.

    Parameters
    - batches: mapping of batch name -> DataFrame
    - out_path: directory to write CSVs into
    - suffix: optional suffix to append to each filename (e.g. "_SpectraFits")
    """
    for name, df in batches.items():

        filename = f"{name}{suffix}.csv" if suffix else f"{name}.csv"
        output_csv = out_path / filename
        df.to_csv(output_csv, index=False)
        print(f"  Saved batch artifact to {output_csv.name}")

def extract_bateman_populations(rate_batches: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """
    Workflow function that absorbs the loop. 
    Iterates over the collection of batches and applies the core physics fit.
    """
    population_results = {}
    
    for batch_name, df in rate_batches.items():
        # The variables times_sec, rates, errors are extracted here
        times_sec = parse_timestamps(df["Datetime"])
        rates = df["rate_cps"].values
        errors = df["rate_cps_error"].values
        
        # Call the core math function
        population_results[batch_name] = fit_initial_populations(times_sec, rates, errors)
        
    return population_results

def compute_recoil_accumulation_limits(population_fits: Dict[str, dict], accumulation_configs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Takes the fitted initial populations and applies the back-calculation to find 
    recoil accumulation limits. This is where you can also merge with alpha data in the future.
    """
    accumulation_results = {}

    # Helper to robustly obtain fit values from different possible key names
    def _get_fit_val(fp: dict, candidates: list, default: float = 0.0):
        for k in candidates:
            if k in fp:
                return fp[k]
        return default

    for batch_name, fit_params in population_fits.items():
        config = accumulation_configs.get(batch_name, {})
        delay_seconds = config.get("delay_seconds", 0)
        acc_seconds = config.get("acc_seconds", 0)

        # Read the measured fit values (supporting a couple of likely key names)
        n_ra_fit = _get_fit_val(fit_params, ["ra224_atoms_t0", "n_ra_0", "n_ra_fit"]) 
        n_ra_fit_err = _get_fit_val(fit_params, ["ra224_atoms_t0_err", "n_ra_0_err", "n_ra_fit_err"]) 
        n_pb_fit = _get_fit_val(fit_params, ["pb212_atoms_t0", "n_pb_0", "n_pb_fit"]) 
        n_pb_fit_err = _get_fit_val(fit_params, ["pb212_atoms_t0_err", "n_pb_0_err", "n_pb_fit_err"]) 

        backcalc = backcalculate_accumulation(n_ra_fit=n_ra_fit, n_pb_fit=n_pb_fit,
                                              delay_seconds=delay_seconds, acc_seconds=acc_seconds)

        accumulation_results[batch_name] = {
            "n_ra_fit": n_ra_fit,
            "n_ra_fit_err": n_ra_fit_err,
            "ra_activity_bq": n_ra_fit * LAMBDA_RA,
            "ra_activity_err_bq": n_ra_fit_err * LAMBDA_RA,
            "n_pb_fit": n_pb_fit,
            "n_pb_fit_err": n_pb_fit_err,
            "pb_activity_bq": n_pb_fit * LAMBDA_PB,
            "pb_activity_err_bq": n_pb_fit_err * LAMBDA_PB,
            "delay_minutes": float(delay_seconds) / 60.0,
            "n_ra_end_acc": backcalc.get("n_ra_end_acc"),
            "n_pb_end_acc": backcalc.get("n_pb_end_acc"),
            "max_ra_capacity": backcalc.get("max_ra_capacity"),
            "ratio_ra_to_pb": backcalc.get("ratio_ra_to_pb"),
            "saturation_pct": backcalc.get("saturation_pct"),
        }

    return accumulation_results

def export_accumulation_summary(accumulation_metrics: Dict[str, dict], output_csv: Path):
    """Exports the accumulation metrics to a CSV file.

    The CSV will contain the following columns in this order:
      Batch_ID, n_ra_fit, n_ra_fit_err, n_pb_fit, n_pb_fit_err,
      delay_minutes, n_ra_end_acc, n_pb_end_acc, max_ra_capacity, saturation_pct
    """
    # Build DataFrame, move the batch name from the index into a column
    summary_df = pd.DataFrame.from_dict(accumulation_metrics, orient='index')
    summary_df.index.name = 'Batch_ID'
    summary_df = summary_df.reset_index()

    # Ensure column order and presence (fill missing with NaN)
    cols = [
        'Batch_ID',
        'n_ra_fit', 'n_ra_fit_err', 'ra_activity_bq', 'ra_activity_err_bq',
        'n_pb_fit', 'n_pb_fit_err', 'pb_activity_bq', 'pb_activity_err_bq',
        'delay_minutes',
        'n_ra_end_acc', 'n_pb_end_acc',
        'max_ra_capacity', 'saturation_pct',
        'th228_true_bq', 'desorption_probability_pct',
    ]
    for c in cols:
        if c not in summary_df.columns:
            summary_df[c] = pd.NA

    summary_df = summary_df[cols]

    # Write numeric values in scientific notation where appropriate
    summary_df.to_csv(output_csv, index=False, float_format='%.2e')
    print(f"  Saved accumulation summary to {output_csv.name}")

def get_mca_results(mca_csv_path, mca_channels):
    mca_integration_time = float(pd.read_csv(mca_csv_path, skiprows=9, nrows=1).columns[0].split(' ')[0])
    mca_date_str = pd.read_csv(mca_csv_path, skiprows=7, nrows=1).columns[0].split(' ')[0]
    mca_date = datetime.strptime(mca_date_str, "%m/%d/%Y")

    dfmca = pd.read_csv(mca_csv_path, skiprows=11, nrows=2048)
    mca_counts = dfmca[mca_channels[0]: mca_channels[1]].sum().values[0]
    mca_activity = mca_counts / mca_integration_time
    return {'mca_activity': mca_activity, 'mca_date': mca_date}


def decay_correct_activity(a0: float, date0: str, date_target: str) -> float:
    """Decay corrects the Th-228 activity between two date strings (YYYY-MM-DD)."""
    t0 = datetime.strptime(date0, "%Y-%m-%d")
    t1 = datetime.strptime(date_target, "%Y-%m-%d")
    delta_days = (t1 - t0).days
    return a0 * np.exp(-LAMBDA_TH228_DAYS * delta_days)

def compute_desorption_probabilities(metrics: Dict, rate_batches: Dict[str, pd.DataFrame], mca_csv_path: str, mca_channels: list,
                                     foil_geometry_fraction: float = 0.5) -> Dict:
    """Unifies the Alpha and Gamma pipelines to find P_desorp."""
    
    # 1. Load MCA Data
    mca_results = get_mca_results(mca_csv_path=mca_csv_path, mca_channels=mca_channels)
    mca_date_str = mca_results['mca_date'].strftime("%Y-%m-%d")

    updated_metrics = {}
    for batch_name, batch_metrics in metrics.items():
        # Get gamma measurement date from the first row of Datetime
        first_dt = pd.to_datetime(rate_batches[batch_name]["Datetime"].iloc[0])
        gamma_date_str = first_dt.strftime("%Y-%m-%d")

        # 2. Decay correct the Th-228 source to the date of the Gamma measurement
        th228_true_bq = decay_correct_activity(mca_results['mca_activity'], mca_date_str, gamma_date_str)
        
        # Convert capacity atoms to Bq
        ra224_foil_capacity_bq = batch_metrics['max_ra_capacity'] * LAMBDA_RA224_SEC
        
        # P_desorp = Foil_Bq / (Source_Bq * Geometry)
        desorption_probability_pct = (ra224_foil_capacity_bq / (th228_true_bq * foil_geometry_fraction))
        
        new_metrics = batch_metrics.copy()
        new_metrics['th228_true_bq'] = th228_true_bq
        new_metrics['desorption_probability_pct'] = desorption_probability_pct
        updated_metrics[batch_name] = new_metrics
        
    return updated_metrics