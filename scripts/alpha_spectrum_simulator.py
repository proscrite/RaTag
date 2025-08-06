import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import crystalball
import pandas as pd

# from RaTag.scripts.unit_conversion import DEFAULT_TH228_CHAIN
from RaTag.scripts.crystalball_fitter import FitCbResult

# --- DecayMode and SpectrumSimulator classes for simulating decay spectra ---

class DecayMode:
    """
    Represents a single alpha decay mode with a Crystal Ball distribution.
    """
    def __init__(self, name, loc, beta, m, scale, branching_ratio):
        self.name = name
        self.loc = loc
        self.beta = beta
        self.m = m
        self.scale = scale
        self.branching_ratio = branching_ratio
        # Create normalized crystal ball distribution
        self.dist = crystalball(beta=self.beta, m=self.m, loc=self.loc, scale=self.scale)

    def sample(self, n_events, seed=None):
        """
        Generate samples for this decay mode via inverse transform sampling.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random
        u = rng.uniform(1e-9, 1 - 1e-9, n_events)
        return self.dist.ppf(u)

class SpectrumSimulator:
    """
    Simulates a combined alpha-decay energy spectrum from multiple decay modes.
    """
    def __init__(self, decay_modes):
        self.decay_modes = decay_modes

    def simulate(self, total_events, seed=None):
        """
        Generate a combined sample of total_events across all decay modes,
        allocating counts according to branching ratios.
        Returns a dict mapping mode names to individual samples, and an array of all samples.
        """
        rng = np.random.default_rng(seed)
        results = {}
        all_samples = []
        # Allocate events per mode
        for mode in self.decay_modes:
            n_mode = int(np.round(mode.branching_ratio * total_events))
            samples = mode.sample(n_mode, seed=rng.integers(1e9))
            results[mode.name] = samples
            all_samples.append(samples)
        combined = np.concatenate(all_samples)
        rng.shuffle(combined)
        return results, combined
    

def simulate_Th_Ra(fit_cb_result : FitCbResult,  N=1_000_000):
    """
    Simulate the alpha decay spectrum for Th-228 and Ra-224 using Crystal Ball distributions
    and plot the results.
    Parameters:
        cb1_x0, cb2_x0, cb3_x0, cb4_x0: Locations of the Crystal Ball distributions for Th-228 and Ra-224.
        beta, m, scale: Parameters for the Crystal Ball distribution.
    """
    
    beta = fit_cb_result.beta
    m = fit_cb_result.m
    scale = fit_cb_result.scale
    # Define decay modes for Th-228 and Ra-224 (hardcoded because they are fixed)
    # These values are based on the Th-228 decay chain and Ra-224 decay chain
    modes = [
        DecayMode(name="Th228_5423keV", loc=5423, branching_ratio=0.722, beta=beta, m=m, scale=scale),
        DecayMode(name="Th228_5340keV", loc=5340, branching_ratio=0.15, beta=beta, m=m, scale=scale),#0.278),
        DecayMode(name="Ra224_5686keV", loc=5686, branching_ratio=0.722, beta=beta, m=m, scale=scale),#0.949),
        DecayMode(name="Ra224_5449keV", loc=5449, branching_ratio=0.038, beta=beta, m=m, scale=scale), #0.051),
        ]

    simulator = SpectrumSimulator(modes)
    results, spectrum = simulator.simulate(total_events=N)
    return results, spectrum

# --- Functions for plotting and analyzing the simulated spectra ---

def histogram_decay(decays, bins=1000, xmin=5200, xmax=9000, col='blue', ax=None, label=None, total_events=None):
    """
    Plot a histogram for a decay mode using weights proportional to its contribution
    to the total spectrum based on branching ratio.

    If total_events is provided, the histogram will be scaled accordingly.
    """
    bin_width = (xmax - xmin) / bins
    if total_events is None:
        # Normalize independently
        weights = np.full_like(decays, 1 / (len(decays) * bin_width))
    else:
        # Normalize relative to total simulated events
        weights = np.full_like(decays, 1 / (total_events * bin_width))

    if ax is None:
        ax = plt.gca()

    n, bins, _ = ax.hist(decays, bins=bins, range=(xmin, xmax), weights=weights,
            alpha=0.9, color=col, label=label)
    ax.set(xlabel="Energy (channel)", ylabel="Normalized Counts", title="Decay Spectra")

    return n, bins

def plot_Th_Ra_simulation(results: dict, cutdown, cutup, ax=None):
    """
    Plot the simulated decay spectra for Th-228 and Ra-224.
    Parameters:
        results: Dictionary containing the simulated decay modes and their samples.
    """
    if ax is None:
        ax = plt.gca()
    total_events = sum(len(v) for v in results.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i,k in enumerate(results):
        print(f"{k}: {len(results[k])} events")
        n, bins = histogram_decay(results[k], xmin=cutdown, xmax=cutup, col=colors[i], bins=1000, ax=ax, label=k, total_events=total_events)
        if k == 'Th228_5423keV':
            norm_factor = np.max(n)

    ax.legend()
    plt.tight_layout()
    plt.gca().set(xlim=(cutdown, cutup), )
    
    return norm_factor, ax

def plot_simulation_full_spectrum(spectrum_ThRa, cutdown=5000, cutup=5800, ax=None):
    """
    Plot the full simulated spectrum for Th-228 and Ra-224.
    Parameters:
        spectrum_ThRa: Array containing the simulated decay spectrum.
        cutdown: Lower energy limit for the plot.
        cutup: Upper energy limit for the plot.
        ax: Matplotlib axis to plot on.
    """
    if ax is None:
        ax = plt.gca()
    
    # Normalize the histogram
    bin_width = (cutup - cutdown) / 1000
    weights = np.full_like(spectrum_ThRa, 1 / (len(spectrum_ThRa) * bin_width))
    n, bins, _ = ax.hist(spectrum_ThRa, bins=1000, range=(cutdown, cutup), weights=weights,
                         alpha=0.8, label='Simulated Spectrum', color='gray')
    ax.set(xlabel='Energy (keV)', ylabel='Normalized Counts', title='Decay Spectrum Simulation')
    ax.legend()
    ax.set(xlim=(cutdown, cutup), )
    return np.max(n), ax
    # n, bi, _ = ax.hist(spectrum_ThRa, bins=1000, range=(5000, 5800), weights=np.full_like(spectrum_ThRa, 1 / (len(spectrum_ThRa) * (1050 - 820) / 1000)), alpha=0.8, label='Simulated Spectrum', color='gray');

def plot_normalized_spectrum(df_roi: pd.DataFrame, norm_factor = 1, ax=None):
    """ Plot the normalized spectrum with the Th-228 peak.
    Parameters:
        df_roi: DataFrame containing the spectrum data.
        norm_factor: Normalization factor for the Th-228 peak.
        ax: Matplotlib axis to plot on.
    """
    if ax is None:
        ax = plt.gca()
    dff = df_roi.copy()
    # normalize and scale by the provided factor
    dff['counts_norm'] = dff['counts'] / dff['counts'].max() * norm_factor
    # choose x‐axis
    if 'energy' in dff.columns:
        x = dff['energy']
        ax.set_xlabel('Energy (keV)')
    else:
        x = dff.index
        ax.set_xlabel('Channel')
    ax.plot(x, dff['counts_norm'], 'r-', label='Measured spectrum')
    ax.legend()
    return ax

# --- Functions for computing S/B ratio and background rate ---

def compute_SB_ratio(E_low, E_high, simulation_ThRa, stat_sign=False):
    """
    Compute the signal-to-background ratio for the Th-Ra simulation data.
    Parameters:
    - E_low: Lower energy cut (keV)
    - E_high: Upper energy cut (keV)
    - simulation_ThRa: Dictionary containing the simulated data for Th and Ra isotopes.
    - stat_sign: If True, return the statistical significance instead of the ratio.
    Returns:
    - S/B ratio or statistical significance.
    """
    th228_5423 = simulation_ThRa['Th228_5423keV']
    th228_5340 = simulation_ThRa['Th228_5340keV']
    ra224_5686 = simulation_ThRa['Ra224_5686keV']
    ra224_5449 = simulation_ThRa['Ra224_5449keV']
    
    # Apply the cut to each isotope separately
    S = (
        np.sum((th228_5423 >= E_low) & (th228_5423 <= E_high)) +
        np.sum((th228_5340 >= E_low) & (th228_5340 <= E_high))
    )
    B = (
        np.sum((ra224_5686 >= E_low) & (ra224_5686 <= E_high)) +
        np.sum((ra224_5449 >= E_low) & (ra224_5449 <= E_high))
    )
    
    if stat_sign:
        return S / (np.sqrt(S) + np.sqrt(B)) if (S+B) > 0 else np.inf
    else:
        return S / B if B > 0 else np.inf

def contour_plot_SB_ratio(E_low_range, E_high_range, simulation_ThRa, ax=None):
    # Example scanning ranges (adjust for your energies)
    """Plot a contour of S/B ratio as a function of energy cuts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    

    # Prepare array to hold results
    S_over_B = np.full((len(E_low_range), len(E_high_range)), np.nan)

    # Loop over all cut combinations
    for i, E_low in enumerate(E_low_range):
        for j, E_high in enumerate(E_high_range):
            if E_high <= E_low:
                continue  # skip invalid cuts
            S_over_B[i, j] = compute_SB_ratio(E_low, E_high, simulation_ThRa, stat_sign=False)

    # Plot the contour
    c = ax.contourf(E_high_range, E_low_range, S_over_B, levels=50, cmap='viridis')
    ax.set_xlabel("Upper Energy Cut (keV)")
    ax.set_ylabel("Lower Energy Cut (keV)")
    ax.set_title("S/B Ratio as Function of Energy Cuts")
    fig.colorbar(c, label="S/B")
    return S_over_B, ax

def get_background_rate(results, low_cut, high_cut):
    """
    Calculate the background rate in the specified energy range.
    """
    ra_vals = np.concatenate([results['Ra224_5686keV'], results['Ra224_5449keV']])
    n_ra_in_cut = np.count_nonzero((ra_vals >= low_cut) & (ra_vals <= high_cut))
    background_rate = n_ra_in_cut / sum(len(v) for v in results.values()) * 100
    # background_rate = fraction_of_ra / (high_cut - low_cut)  # Approximate rate per channel
    return background_rate