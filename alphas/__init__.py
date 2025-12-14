"""
Alpha spectrum analysis package.

Modules:
- spectrum_fitting: Core fitting algorithms and energy calibration
- spectrum_plotting: Visualization functions for fits and calibrations
- energy_map_reader: Binary energy map file I/O
- energy_map_writer: Energy map generation from waveforms
- energy_join: Joining energy maps with event data
- wfm2spectra: Waveform to spectrum conversion
"""

# Core fitting and calibration
from RaTag.alphas.spectrum_fitting import (
    # Data types
    SpectrumData,
    EnergyCalibration,
    IsotopeRange,
    # Data loading
    load_spectrum_from_run,
    load_spectrum_from_energy_maps,
    # Fitting functions
    fit_single_crystalball,
    fit_po212_alpha_beta,
    fit_multi_crystalball_progressive,
    fit_full_spectrum_hierarchical,
    prepare_hierarchical_fit,
    # Calibration and range derivation
    derive_energy_calibration,
    derive_isotope_ranges,
    # Utilities
    ranges_to_dict,
)

# Visualization functions
from RaTag.alphas.spectrum_plotting import (
    plot_preliminary_fits,
    plot_energy_calibration,
    plot_isotope_ranges,
    plot_hierarchical_fit,
    plot_calibration_summary,
    plot_fit_peak,
    plot_residuals,
    mark_peak_position,
)

__all__ = [
    # Data types
    'SpectrumData',
    'EnergyCalibration',
    'IsotopeRange',
    # Data loading
    'load_spectrum_from_run',
    'load_spectrum_from_energy_maps',
    # Fitting
    'fit_single_crystalball',
    'fit_po212_alpha_beta',
    'fit_multi_crystalball_progressive',
    'fit_full_spectrum_hierarchical',
    'prepare_hierarchical_fit',
    # Calibration
    'derive_energy_calibration',
    'derive_isotope_ranges',
    'ranges_to_dict',
    # Plotting
    'plot_preliminary_fits',
    'plot_energy_calibration',
    'plot_isotope_ranges',
    'plot_hierarchical_fit',
    'plot_calibration_summary',
    'plot_fit_peak',
    'plot_residuals',
    'mark_peak_position',
]
