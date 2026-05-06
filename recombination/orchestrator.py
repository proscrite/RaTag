# 1. Fix output of main pipeline
# 2. Apply X-ray calibration factor g_s2 to compute expected electrons
# 3. Implement transmission efficiency evaluation and correction
# 4. Compute Y (E_EL) and apply to get detector efficiency, include geometry factor
# 5. For Ra224, obtain E_R_mean from p_desorp

# 4. Compute recombination fraction and propagate uncertainties

"""
Pipeline for Multiphysics Statistical Integration.

This module acts as the orchestrator. It reads the integration YAML configuration,
fetches the necessary artifacts and run data, applies the designated physical
corrections (future step), and outputs the aggregated results.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt


from RaTag.recombination.recombination_plotters import plot_metric_vs_field

from RaTag.recombination.artifact_adapter import (load_integration_config, load_run_data, parse_el_trend_artifact, parse_geometric_artifact,
                                                   parse_gs2_artifact, parse_transmission_artifact, 
                                                   parse_desorption_artifact)

from RaTag.recombination.operators import (apply_el_yield_conversion, apply_gs2_conversion, calc_expected_electrons,
                                            apply_transmission_efficiency, 
                                            compute_recombination_fraction,
                                            get_isotope_recoil_energy, ALPHA_DECAY_KINEMATICS)

from RaTag.recombination.datatypes import CorrectionFactor, get_identity_factor


def plot_diagnostics(processed_runs: Dict[str, Any], config: Dict[str, Any]):
    """Utility function to generate diagnostic plots at each stage of the pipeline."""
    output_dir = Path(config.get('plot_path', 'artifacts/recombination/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_config = config.get('diagnostics_plots', [])

    if not plots_config:
        print("No diagnostic plots configured. Skipping plotting step.")
        return

    for p in plots_config:
        try:
            fig = plot_metric_vs_field(run_data=processed_runs, run_manifest=config['runs'], 
                                    y_col=p['y_col'], y_err_col=p['y_err_col'], ylabel=p['ylabel'], 
                                    title=p['title'])
            fig.savefig(output_dir / p['filename'], dpi=150)
            print(f"Saved plot: {output_dir / p['filename']}")
            plt.close(fig)
        except KeyError as e:
            print(f"Plot configuration error: Missing key {e} in plot config {p}. Skipping this plot.")

# ---  The Orchestrator (Main Execution) ---

def run_multiphysics_integration(config_path: str):
    """Main pipeline execution."""
    print("=" * 60)
    print("MULTIPHYSICS STATISTICAL INTEGRATION ENGINE")
    print("=" * 60)
    
    # 1. Load State
    print(f"Loading configuration from: {config_path}")
    config = load_integration_config(Path(config_path))
    raw_runs = load_run_data(config['runs'])
    
    print("Initial state loaded and plotted.")

    # Load Yield artifacts via Adapter
    gs2_artifact = parse_gs2_artifact(config['artifacts']['xray_factor'])
    g_ratio_artifact = parse_geometric_artifact(config['artifacts']['geometric_factor'])
    el_trend_model = parse_el_trend_artifact(config['artifacts']['el_trend_factor'])
    
    print(f"g_S2 = {gs2_artifact.value:.4f} ± {gs2_artifact.uncertainty:.4f} {gs2_artifact.units}")
    print(f"Geometric Ratio (X-ray to Recoil) = {g_ratio_artifact.value:.4f} {g_ratio_artifact.units}")

    # Load transmission efficiency model artifact
    trans_model = parse_transmission_artifact(config['artifacts']['transmission_efficiency'])

    # Load desorption probability artifact for Ra-224
    desorp_artifact = parse_desorption_artifact(config['artifacts']['gamma_spectroscopy'])
    print(f"Loaded desorption probability artifact: p_desorp = {desorp_artifact.value:.4f} ± {desorp_artifact.uncertainty:.4f} {desorp_artifact.units}")

    processed_runs = {}

    for run_id, df_raw in raw_runs.items():

        print(f"\nProcessing {run_id}...")
        run_meta = config['runs'][run_id]
        isotope = run_meta.get('isotope', 'Unknown')
        df_current = df_raw.copy()
        e_el = run_meta.get('EL_field_Vcm')
        
        # --- GLOBAL DETECTOR PHYSICS ---

        # 1. Correct by EL yield to get measured electrons
        
        print(f"  -> Applying Total EL Yield Y(E_EL) at {e_el} kV/cm")
        
        print(f"Relative EL Yield Trend f(E_EL) at {e_el} kV/cm = {el_trend_model.evaluate(e_el):.3f} (normalized to 1.0 at reference field)")
        df_gs2 = apply_el_yield_conversion(df_current, gs2_artifact, g_ratio_artifact, el_trend_model, e_el)
        processed_runs[run_id] = df_gs2

        # 2. Gate Transmission Correction (Requires transparency model and Run E_EL)
        
        df_drift = apply_transmission_efficiency(df_gs2, trans_model, e_el)

        w_value = gs2_artifact.metadata.get("W_value", 22.3)  # Default W-value in eV
        
        # --- ISOTOPE SPECIFIC PHYSICS ---

        recoil_energy_kev = get_isotope_recoil_energy(isotope)
        daughter_name = ALPHA_DECAY_KINEMATICS[isotope]["daughter"]
        
        print(f"  -> Recoil Energy ({daughter_name}): {recoil_energy_kev:.1f} keV")

        if isotope == "Th228":
            desorp_artifact = get_identity_factor("P_desorp_Th228_Identity")


        n_e_exp = calc_expected_electrons(recoil_energy_kev, w_value, desorp_artifact)
        print(f"  -> Expected electrons for {isotope}: {n_e_exp:.1f}")
            
        # 4. Final Recombination (Using the drifting electrons, not just measured)
        df_recomb = compute_recombination_fraction(df_drift, n_e_exp)
            
        # Lock in the fully processed state
        processed_runs[run_id] = df_recomb

    
    plot_diagnostics(processed_runs, config)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    run_multiphysics_integration("configs/recombination/recombination.yaml")
