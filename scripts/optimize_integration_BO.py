import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real

# --- RaTag Imports ---
from RaTag.io.bootstrap import bootstrap_from_config
from RaTag.el_tpc.baseline_workflow import map_baseline
from RaTag.el_tpc.drift_workflow import map_drift_physics
from RaTag.el_tpc.timing_workflow import map_time_windows
from RaTag.core.config import TimingConfig, IntegrationConfig
from RaTag.io.file_ops import iter_waveforms
from RaTag.waveform.preprocessing import standard_preprocessing
from RaTag.el_tpc.fit_s2_area import fit_s2_crystalball


def _compute_static_bounds(set_pmt, n_sigma_start: float, n_sigma_end: float) -> tuple[float, float]:
    """Helper to calculate the integration window based on trial n_sigma parameters."""
    t_start = float(set_pmt.t_s2_start)
    t_end = float(set_pmt.t_s2_end)
    std_start = float(set_pmt.t_s2_start_std or 0.0)
    std_end = float(set_pmt.t_s2_end_std or 0.0)
    
    return t_start - (n_sigma_start * std_start), t_end + (n_sigma_end * std_end)


def evaluate_recoil_resolution(params: list, run, base_config: IntegrationConfig, max_files: int = 10) -> float:
    """
    Objective function: Minimizes the relative uncertainty (ci95/mu) of the S2 fit
    by finding the perfect integration window margins.
    """
    n_sigma_start, n_sigma_end = params
    
    total_loss = 0.0
    valid_sets = 0
    
    for set_pmt in run.sets:
        if getattr(set_pmt, 't_s2_start', None) is None:
            continue
            
        static_start, static_end = _compute_static_bounds(set_pmt, n_sigma_start, n_sigma_end)
        
        # Guard against mathematically inverted windows
        if static_start >= static_end:
            total_loss += 100.0
            continue
            
        out_areas = []
        
        for wf in iter_waveforms(set_pmt, max_files=max_files):
            # We use the FIXED pre-processing params (optimized by the timing pipeline)
            wf = standard_preprocessing(wf, 
                                        n_pedestal=base_config.n_pedestal, 
                                        ma_window=base_config.ma_window, 
                                        threshold=base_config.bs_threshold)
            
            mask = (wf.t >= static_start) & (wf.t <= static_end)
            v_window = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
            
            dt = wf.t[1] - wf.t[0]
            chunk_areas = np.trapezoid(v_window, dx=dt, axis=1)
            out_areas.append(chunk_areas)
            
        if not out_areas:
            total_loss += 100.0
            continue
            
        areas = np.concatenate(out_areas)
        
        try:
            # We use our rigid Crystal Ball fitter to score the distribution
            result = fit_s2_crystalball(areas, bin_cuts=(0, 10), nbins=100)
            mu = result['peak_position']
            ci95 = result['ci95']
            
            if mu < 0.2:
                set_loss = 50.0 # Penalize grabbing only the noise peak
            else:
                set_loss = ci95 / mu # The optimal metric
                
        except Exception:
            set_loss = 100.0 # Penalize parameters that cause fit failures
            
        total_loss += set_loss
        valid_sets += 1
        
    return total_loss / max(1, valid_sets)


def generate_validation_plots(best_params: list, run, base_config: IntegrationConfig, run_id: str, max_files: int = 10):
    """Generates the visual proof for the optimal integration margins."""
    n_sigma_start, n_sigma_end = best_params
    
    test_sets = [run.sets[0], run.sets[-1]] if len(run.sets) > 1 else run.sets
    
    fig, axes = plt.subplots(1, len(test_sets), figsize=(6 * len(test_sets), 5))
    if len(test_sets) == 1: axes = [axes]
    
    fig.suptitle(f"Integration Window Validation - Run {run_id}\nσ_start={n_sigma_start:.2f}, σ_end={n_sigma_end:.2f}", fontsize=14)
    
    for i, set_pmt in enumerate(test_sets):
        ax = axes[i]
        
        if getattr(set_pmt, 't_s2_start', None) is None:
            continue
            
        static_start, static_end = _compute_static_bounds(set_pmt, n_sigma_start, n_sigma_end)
        out_areas = []
        
        for wf in iter_waveforms(set_pmt, max_files=max_files):
            wf = standard_preprocessing(wf, 
                                        n_pedestal=base_config.n_pedestal, 
                                        ma_window=base_config.ma_window, 
                                        threshold=base_config.bs_threshold)
            
            mask = (wf.t >= static_start) & (wf.t <= static_end)
            v_window = wf.v[:, mask] if wf.ff else wf.v[mask][np.newaxis, :]
            dt = wf.t[1] - wf.t[0]
            out_areas.append(np.trapezoid(v_window, dx=dt, axis=1))
            
        areas = np.concatenate(out_areas)
        filtered = areas[(areas >= 0) & (areas <= 10)]
        
        ax.hist(filtered, bins=100, alpha=0.5, color='orange', label='Data')
        
        try:
            result = fit_s2_crystalball(areas, bin_cuts=(0, 10), nbins=100)
            x_smooth = np.linspace(0, 10, 500)
            
            from RaTag.el_tpc.fit_s2_area import v_crystalball_right
            y_fit = result['result'].eval(x=x_smooth)
            
            ax.plot(x_smooth, y_fit, 'g-', lw=2, label=f"Fit (μ={result['peak_position']:.2f})")
            ax.axvline(result['peak_position'], color='green', linestyle=':')
            
        except Exception:
            ax.text(0.5, 0.5, "Fit Failed", ha='center')
            
        ax.set_title(f"{set_pmt.source_dir.name}\nWindow: [{static_start:.1f}, {static_end:.1f}] µs")
        ax.set_xlabel("S2 Area (mV·µs)")
        ax.legend()
        
    plt.tight_layout()
    plot_file = f"artifacts/recoil_BO/integration_bounds_validation_{run_id}.png"
    Path(plot_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    print(f"Validation plot saved to {plot_file}")


def main(run_id: str, config_path: str, max_files: int):
    # 1. Bootstrap the physics environment
    print(f"Bootstrapping Run {run_id}...")
    run = bootstrap_from_config(config_path)
    run = map_drift_physics(run)
    run = map_baseline(run, max_frames=None, n_points=200)
    
    print(f"Mapping Timing (Required for Recoil bounds)...")
    run = map_time_windows(run, max_frames=max_files*48, config=TimingConfig(), force=False)
    
    base_config = IntegrationConfig()
    
    # 2. Define the Search Space
    # We are now searching for the optimal multipliers for the standard deviation bounds!
    space = [
        Real(0.1, 5.0, name='n_sigma_start'),
        Real(0.5, 10.0, name='n_sigma_end') 
    ]
    
    # 3. Run Optimization
    print(f"Starting BO for Integration Margins over {len(run.sets)} sets...")
    res = gp_minimize(
        lambda p: evaluate_recoil_resolution(p, run=run, base_config=base_config, max_files=max_files), 
        space, 
        n_calls=20, 
        n_random_starts=10, 
        random_state=42,
        verbose=True
    )
    
    # 4. Save Configuration Data
    results = {
        "run_id": run_id,
        "best_score": res.fun,
        "parameters": {
            "n_sigma_start": float(res.x[0]),
            "n_sigma_end": float(res.x[1])
        }
    }
    
    json_file = f"artifacts/recoil_BO/integration_config_{run_id}.json"
    Path(json_file).parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nOptimization complete. Config saved to {json_file}")
    
    # 5. Generate Visual Proof
    generate_validation_plots(res.x, run, base_config, run_id, max_files=max_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize S2 Integration Window Boundaries.")
    parser.add_argument("--run", type=str, required=True, help="Run ID to optimize")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--files", type=int, default=10, help="Max files per set to process")
    args = parser.parse_args()
    
    main(args.run, args.config, args.files)