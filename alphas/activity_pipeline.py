import yaml
from pathlib import Path

from .activity_workflow import (initialize_run_and_load_spectra, apply_energy_calibration,
                             measure_activities, export_activity_artifacts)
from .spectrum_fitting import IsotopeRange
from .activity_plotting import plot_activity_timeseries
from RaTag.plotting import plot_alpha_energy_spectrum

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def execute_alpha_pipeline(config_path: str, output_dir: str = "artifacts/alpha_spectroscopy"):
    """Explicit, top-down orchestrator for Alpha Yield analysis."""
    
    config = load_config(config_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Starting Alpha Pipeline...")
    
    # 1. Initialize run and load all time-stamped spectra
    print("Loading timestamped spectra...")
    run, spectra = initialize_run_and_load_spectra(config['run_params'],
                                                 energy_range=tuple(config['energy_range']))
    
    # 2. Apply quadratic calibration
    calib_file = config.get('calibration_file')
    try:
        print("Applying energy calibration...")
        spectra = apply_energy_calibration(spectra, calib_file, energy_range=tuple(config['calib_energy_range']))
        isotope_config = config['isotope']
        is_calibrated = True
    except FileNotFoundError as e:
        print(f"Calibration file missing or unprovided. Proceeding with instrumental energy scale.")
        isotope_config = config.get('isotope_instrumental', config['isotope'])
        is_calibrated = False
    
    # 3. Measure Th-228 activity for all time points
    print("Measuring activity timeseries...")
    measurements = measure_activities(spectra, isotope_config)

    # Output Activity plots
    print("Plotting activity timeseries...")
    fig = plot_activity_timeseries(measurements)
    fig.savefig(out_path / "activity_timeseries.png", dpi=150)
    
    # Optional: Plot the alpha spectrum with isotope ranges for verification
    if config.get('debug', False):
        print("Debug mode: Plotting alpha spectrum to verify ranges...")
        energy_array = [spectra[0].spectrum.energies]
        fig_debug, _ = plot_alpha_energy_spectrum(energies=energy_array, title="Debug Alpha Spectrum",
                                                nbins=100, energy_range=tuple(config['energy_range']),)
        fig_debug.savefig(out_path / "debug_alpha_spectrum.png", dpi=150)
    # 4. Final Export
    print("Exporting activity artifacts...")
    export_activity_artifacts(measurements, out_path, filename=config.get('output_filename', "activity_measurements.csv"))
    
    print(f"Pipeline execution complete. Artifacts saved to {out_path}")

if __name__ == "__main__":
    yaml_file = "configs/alpha_spectroscopy/th228_activity_analysis.yaml"
    outpath = "artifacts/alpha_spectroscopy"
    execute_alpha_pipeline(yaml_file, outpath)
