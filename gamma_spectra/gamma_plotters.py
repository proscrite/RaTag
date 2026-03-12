import matplotlib.pyplot as plt
from matplotlib.pylab import yscale
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .etl_hidex import gaussian_linear, bateman_count_rate, parse_timestamps

# --- INDIVIDUAL PLOT GENERATORS (Pure-ish Functions) ---

def plot_spectrum_fit(x: np.ndarray, y: np.ndarray, popt: tuple, 
                      vial_number: str, spectra_row: int) -> plt.Figure:
    """Generates a plot of the raw spectrum and the fitted curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "o", ms=3, label="Data")
    
    if popt[0] != 0:
        # Plot the smooth fit
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        ax.plot(x_smooth, gaussian_linear(x_smooth, *popt), label="Fitted Curve", color='red')
        ax.legend()
        
    ax.set(xlabel="Channel (0–2047)", ylabel="Counts", title=f"Vial {vial_number} (Spectra row {spectra_row})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_bateman_decay(times_sec: np.ndarray, rates: np.ndarray, errors: np.ndarray, 
                       pop_fits: dict, batch_name: str) -> plt.Figure:
    """Generates the diagnostic decay curve plot (Ra224 and Pb212)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Convert seconds to days for plotting readability
    times_days = times_sec / (24 * 3600)
    
    # Plot experimental data
    ax.errorbar(times_days, rates, yerr=errors, fmt='o', markersize=4, label="Extracted Net Counts")
    
    # Plot theoretical fit
    t_smooth = np.linspace(0, times_sec.max() * 1.1, 500)
    rate_smooth = bateman_count_rate(t_smooth, pop_fits['ra224_atoms_t0'], pop_fits['pb212_atoms_t0'])
    ax.plot(t_smooth / (24 * 3600), rate_smooth, '-', color='red', label="Bateman Fit")
    
    ax.set(xlabel="Time since measurement start (Days)", ylabel="Net Count Rate", 
           title=f"Decay Diagnostics: {batch_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --- COLLECTION ORCHESTRATORS (The Shell) ---

def generate_debug_spectra(raw_batches: Dict[str, pd.DataFrame], 
                           rate_batches: Dict[str, pd.DataFrame], 
                           out_dir: Path,
                           max_spectra: int = 10):
    """Loops over collections and saves individual spectrum plots to a debug folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_id, raw_df in raw_batches.items():
        rate_df = rate_batches[batch_id]
        plotted = 0
        for i, raw_row in raw_df.iterrows():
            # Stop when we've saved max_spectra plots for this batch
            if max_spectra is not None and plotted >= max_spectra:
                break

            spectrum = np.array(raw_row['Spectrum'])
            if np.any(np.isnan(spectrum)):
                continue

            popt = (rate_df.loc[i, 'A'], rate_df.loc[i, 'mu'], rate_df.loc[i, 'sigma'],
                    rate_df.loc[i, 'b'], rate_df.loc[i, 'c'] )

            fig = plot_spectrum_fit(
                x=np.arange(len(spectrum)), y=spectrum, popt=popt,
                vial_number=raw_row['Datetime'], spectra_row=i+20
            )

            # Save and strictly close to prevent memory leaks!
            fig.savefig(out_dir / f"{batch_id}_Vial{raw_row['Datetime']}_row{i+20}.png", dpi=150)
            plt.close(fig)

            plotted += 1

def generate_decay_diagnostics(rate_batches: Dict[str, pd.DataFrame], 
                               population_fits: Dict[str, dict], 
                               out_dir: Path):
    """Generates and saves the batch-level Bateman decay curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_id, df in rate_batches.items():
        times_seconds = parse_timestamps(df["Datetime"])
        rates = df["net_counts"].values
        errors = df["net_counts_error"].values
        
        fig = plot_bateman_decay(times_seconds, rates, errors, population_fits[batch_id], batch_id)
        fig.savefig(out_dir / f"{batch_id}_DecayCurve.png", dpi=150)
        plt.close(fig)

def plot_accumulation_activities(accumulation_metrics: Dict[str, dict], out_path: Path):
    """Generates a bar plot comparing the fitted Ra and Pb activities across batches."""
    out_path.mkdir(parents=True, exist_ok=True)
    
    batch_names = list(accumulation_metrics.keys())
    ra_activities = [metrics['n_ra_fit'] for metrics in accumulation_metrics.values()]
    pb_activities = [metrics['n_pb_fit'] for metrics in accumulation_metrics.values()]
    ra_act_err = [metrics['n_ra_fit_err'] for metrics in accumulation_metrics.values()]
    pb_act_err = [metrics['n_pb_fit_err'] for metrics in accumulation_metrics.values()]

    x = np.arange(len(batch_names))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, ra_activities, yerr=ra_act_err, label='Ra-224 counts', color='blue', markersize=8, capsize=2, fmt='o')
    ax.errorbar(x, pb_activities, yerr=pb_act_err, label='Pb-212 counts', color='orange', markersize=8, capsize=2, fmt='o')
    
    ax.set(ylabel='Number of Atoms', title ='Fitted Activities by Batch', yscale='log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(str(out_path / "AccumulationActivities.png"), dpi=150)
    plt.close(fig)