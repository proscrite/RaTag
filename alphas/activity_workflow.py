import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from RaTag.core.datatypes import Run
from RaTag.workflows.run_construction import initialize_run
from RaTag.alphas.spectrum_fitting import IsotopeRange, SpectrumData
from RaTag.alphas.activity_analysis import (load_all_timestamped_spectra, measure_activity_timeseries,
                                            TimeStampedSpectrum, ActivityMeasurement, BRANCHING_RATIOS)


def initialize_run_and_load_spectra(run_config: dict, energy_range: Tuple[float, float]) -> Tuple[Run, List[TimeStampedSpectrum]]:
    """Initialize the Run object and load all timestamped spectra."""
    path = Path(run_config['root_directory'])
    run = Run(
        run_id=run_config['run_id'],
        root_directory=path,
        target_isotope=run_config.get('target_isotope', 'Multi'),
        pressure=run_config.get('pressure', 2.0),
        temperature=run_config.get('temperature', 297.0),
        sampling_rate=run_config.get('sampling_rate', 5.0),
        drift_gap=run_config.get('drift_gap', 1.4),
        el_gap=run_config.get('el_gap', 0.8),
        el_field=run_config.get('el_field', 2375.0),
        sets=run_config.get('sets', [])
    )
    run = initialize_run(run)
    
    timestamped_spectra = load_all_timestamped_spectra(run, energy_range=energy_range)
    return run, timestamped_spectra


def apply_energy_calibration(timestamped_spectra: List[TimeStampedSpectrum], calib_file: str, energy_range: Tuple[float, float]) -> List[TimeStampedSpectrum]:
    """Applies a quadratic energy calibration to the spectra."""
    calib_data = np.load(calib_file)
    quad_a = calib_data['quad_a']
    quad_b = calib_data['quad_b']
    quad_c = calib_data['quad_c']

    def apply_quadratic_calibration(x):
        return quad_a * x**2 + quad_b * x + quad_c

    calibrated_spectra = []
    for spectrum_t in timestamped_spectra:
        spectrum = spectrum_t.spectrum
        energies_cal = apply_quadratic_calibration(np.array(spectrum.energies))
        spectrum_cal = SpectrumData(
            energies=energies_cal,
            energy_range=energy_range,  
            source=spectrum.source
        )
        spectrum_t_cal = TimeStampedSpectrum(
            spectrum=spectrum_cal,
            timestamp=spectrum_t.timestamp,
            set_name=spectrum_t.set_name,
            acquisition_time=spectrum_t.acquisition_time
        )
        calibrated_spectra.append(spectrum_t_cal)
    
    return calibrated_spectra


def measure_activities(timestamped_spectra: List[TimeStampedSpectrum], isotope_config: dict) -> List[ActivityMeasurement]:
    """Measures the activity for the specified isotope."""
    isotope_range = IsotopeRange(
        name=isotope_config['name'],
        E_min=isotope_config['E_min'],
        E_max=isotope_config['E_max'],
        E_peak=isotope_config['E_peak'],
        sigma=isotope_config['sigma'],
        purity=isotope_config.get('purity', 1.0)
    )
    
    efficiency = isotope_config.get('efficiency', 1.0)
    branching_ratio = BRANCHING_RATIOS.get(isotope_config['name'], isotope_config.get('branching_ratio', 1.0))

    measurements = measure_activity_timeseries(
        timestamped_spectra,
        isotope_range,
        efficiency=efficiency,
        branching_ratio=branching_ratio
    )
    return measurements


def export_activity_artifacts(measurements: List[ActivityMeasurement], out_path: Path, filename: str = "activity_measurements.csv"):
    """Exports activity measurements to a CSV file."""
    data = []
    for m in measurements:
        data.append({
            'time': m.acquisition_time.strftime('%Y-%m-%d %H:%M:%S'),
            'counts': m.counts,
            'live_time_h': m.live_time,
            'activity_bq': m.activity,
            'activity_err': m.activity_err
        })
    df = pd.DataFrame(data)
    df.to_csv(out_path / filename, index=False)
