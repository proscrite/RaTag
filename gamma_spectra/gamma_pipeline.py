from pathlib import Path
from gamma_spectra.gamma_plotters import generate_debug_spectra, generate_decay_diagnostics, plot_accumulation_activities
import yaml

from .gamma_workflow import (
    ingest_hidex_directory,
    fit_batch_gamma_spectra,
    export_batch_artifacts,
    extract_bateman_populations,
    compute_recoil_accumulation_limits,
    export_accumulation_summary,
)

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# The Orchestrator
# ---------------------------------------------------------
def execute_gamma_pipeline(config_path: str, output_dir: str = "artifacts/gamma_spectroscopy"):
    """Explicit, top-down orchestrator for Hidex Gamma Yield analysis."""
    
    config = load_config(config_path)
    data_dir = Path(config['data_dir'])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Starting Gamma Pipeline...")
    
    # 1. Read all raw Excel files into a collection of DataFrames
    print("Ingesting raw Hidex data...")
    raw_batches = ingest_hidex_directory(data_dir)
    
    # 2. Fit the 240 keV peak for every spectrum in every batch
    print("Extracting count rates from gamma spectra...")
    rate_batches = fit_batch_gamma_spectra(raw_batches)
    
    # --- Debug Spectra ---
    if config.get('debug_mode', False):
        print("Debug Mode ON: Generating individual spectrum plots...")
        generate_debug_spectra(raw_batches, rate_batches, out_path / "debug_spectra", max_spectra=config.get('debug_max_spectra', 10))
    
    # 3. Export the spectrum fit results (A, mu, sigma, R_sq) to CSVs
    print("Exporting spectrum fit artifacts...")
    export_batch_artifacts(rate_batches, out_path, suffix="_SpectraFits")

    # 4. Fit the count rates to the coupled Bateman equations
    print("Extracting Ra-224 and Pb-212 populations at t=0...")
    population_fits = extract_bateman_populations(rate_batches)

    # --- Decay Diagnostics ---
    print("Generating batch decay diagnostics...")
    generate_decay_diagnostics(rate_batches, population_fits, out_path / "diagnostics")

    # 5. Back-calculate to account for vacuum delays and find saturation
    print("Computing recoil accumulation metrics...")
    accumulation_metrics = compute_recoil_accumulation_limits(population_fits, 
                                                              accumulation_configs=config['accumulation_params'])
    
    plot_accumulation_activities(accumulation_metrics, out_path )

    export_accumulation_summary(accumulation_metrics, out_path / "accumulation_summary.csv")
    for batch_name, acc_results in accumulation_metrics.items():
        print(f"  {batch_name}: N_Ra_acc={acc_results['n_ra_end_acc']:.2e}, N_Pb_acc={acc_results['n_pb_end_acc']:.2e}, "
                f"Ra/Pb ratio={acc_results['ratio_ra_to_pb']:.2f}")

    
    # 6. (Future) Combine with Alpha data
    # print("Merging with Th-228 alpha activity...")
    # desorption_metrics = compute_desorption_probabilities(accumulation_metrics, config['alpha_artifact_path'])
    
    # 7. Final Export
    # export_accumulation_summary(accumulation_metrics, out_path)

    print(f"Pipeline execution complete. Artifacts saved to {out_path}")

if __name__ == "__main__":
    yaml_file = "/Users/pabloherrero/sabat/RaTagging/configs/gamma_spectroscopy/gamma_spectroscopy.yaml"
    outpath = "/Users/pabloherrero/sabat/RaTagging/artifacts/gamma_spectroscopy"
    execute_gamma_pipeline(yaml_file, outpath)