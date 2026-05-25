import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer

# --- RaTag Imports ---
from RaTag.io.bootstrap import bootstrap_from_config
from RaTag.el_tpc.baseline_workflow import map_run_baseline
from RaTag.el_tpc.drift_workflow import map_drift_physics
from RaTag.io.file_ops import iter_waveforms
from RaTag.el_tpc.waveform_features import find_s1, find_s2, compute_timing_statistics

def evaluate_timing_resolution(params: list, run, max_files: int = 10) -> float:
    """
    Flat objective function evaluating the timing parameter set.
    """
    N_s1, N_s2, window_size, threshold_bs, t_drift_margin = params
    
    total_loss = 0.0
    valid_sets = 0
    
    for set_pmt in run.sets:
        # Use the newly formalized physical property
        noise_sigma = getattr(set_pmt, 'baseline_std', 0.05) 
        if noise_sigma == 0.0:
            noise_sigma = 0.05 # Fallback protection
            
        threshold_s1 = noise_sigma * N_s1
        threshold_s2 = noise_sigma * N_s2
        t_drift_time = (set_pmt.time_drift or 0.0) * t_drift_margin
        
        out_s1, out_s2_start, out_s2_end = [], [], []
        s1_anchor = -5.0
        
        for wf in iter_waveforms(set_pmt, max_files=max_files):
            t_s1 = find_s1(wf, threshold=threshold_s1, t_max=-2.5)
            if not np.all(np.isnan(t_s1)):
                s1_anchor = np.nanmean(t_s1)
                
            t_s2_st, t_s2_ed = find_s2(
                wf, 
                threshold_s2=threshold_s2, 
                t_min=s1_anchor + t_drift_time, 
                window_size=int(window_size), 
                threshold_bs=threshold_bs
            )
            
            out_s1.append(t_s1)
            out_s2_start.append(t_s2_st)
            out_s2_end.append(t_s2_ed)
            
        # Compile arrays
        s1_arr = np.concatenate(out_s1)
        s2_starts = np.concatenate(out_s2_start)
        s2_ends = np.concatenate(out_s2_end)
        
        # Calculate widths securely
        valid_mask = ~np.isnan(s2_starts) & ~np.isnan(s2_ends)
        s2_widths = s2_ends[valid_mask] - s2_starts[valid_mask]
        
        # Statistics
        std_s1 = compute_timing_statistics(s1_arr, name="t_s1").get("t_s1_std", 0.0)
        std_s2_st = compute_timing_statistics(s2_starts, name="t_s2_start").get("t_s2_start_std", 0.0)
        std_s2_wd = compute_timing_statistics(s2_widths, name="s2_width").get("s2_width_std", 0.0)
        
        # Penalty Logic
        if std_s1 == 0.0 or std_s2_st == 0.0 or std_s2_wd == 0.0:
            set_loss = 100.0  # Massive penalty for finding nothing
        else:
            median_width = float(np.median(s2_widths)) if len(s2_widths) > 0 else 0.0
            # Ensure the S2 width is physically reasonable (not chopped off)
            width_penalty = 10.0 if median_width < 1.0 else 0.0 
            
            set_loss = std_s1 + std_s2_st + std_s2_wd + width_penalty
            valid_sets += 1
            
        total_loss += set_loss
        
    return total_loss / max(1, valid_sets)


def generate_validation_plots(best_params: list, run, run_id: str, max_files: int = 10):
    """
    Runs the best parameters on the first and last set to prove physical validity.
    """
    N_s1, N_s2, window_size, threshold_bs, t_drift_margin = best_params
    
    # Pick a high field and low field set (assuming sets are ordered)
    test_sets = run.sets[:3]
    if len(run.sets) < 3:
        test_sets = run.sets
    
    fig, ax = plt.subplots(len(test_sets), figsize=(12, 4 * len(test_sets)))
    fig.suptitle(f"Timing Parameter Validation - Run {run_id}", fontsize=16)
    
    for i, set_pmt in enumerate(test_sets):
        noise_sigma = getattr(set_pmt, 'baseline_std', 0.05)
        t_drift_time = (set_pmt.time_drift or 0.0) * t_drift_margin
        
        out_s1, out_s2_start, out_s2_end = [], [], []
        s1_anchor = -5.0
        
        for wf in iter_waveforms(set_pmt, max_files=max_files):
            t_s1 = find_s1(wf, threshold=noise_sigma * N_s1, t_max=-2.5)
            if not np.all(np.isnan(t_s1)):
                s1_anchor = np.nanmean(t_s1)
                
            t_s2_st, t_s2_ed = find_s2(
                wf, 
                threshold_s2=noise_sigma * N_s2, 
                t_min=s1_anchor + t_drift_time, 
                window_size=int(window_size), 
                threshold_bs=threshold_bs
            )
            
            out_s1.append(t_s1)
            out_s2_start.append(t_s2_st)
            out_s2_end.append(t_s2_ed)
            
        # Plotting
        s1_times = np.concatenate(out_s1)
        s2_times = np.concatenate(out_s2_start)
        s2_end = np.concatenate(out_s2_end)
        
        
        ax[i].hist(s1_times[~np.isnan(s1_times)], bins=50, alpha=0.6, label='S1 Times', color='blue')
        ax[i].hist(s2_times[~np.isnan(s2_times)], bins=50, alpha=0.6, label='S2 Starts', color='orange')
        ax[i].hist(s2_end[~np.isnan(s2_end)], bins=50, alpha=0.6, label='S2 Ends', color='green')
        ax[i].set_title(f"Set {set_pmt.source_dir.name} Timing")
        ax[i].set_xlabel("Time (µs)")
        ax[i].legend()
        
    plt.tight_layout()
    plot_file = f"/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/timing_validation_{run_id}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Validation plot saved to {plot_file}")


def main(run_id: str, config_path: str, max_files: int):
    # 1. Bootstrap the physics environment
    print(f"Bootstrapping Run {run_id}...")
    run = bootstrap_from_config(config_path)
    
    # Apply standard pipeline mapping up to baseline calculation
    run = map_drift_physics(run)
    run = map_run_baseline(run, max_frames=480, n_points=200)
    print(f"Run {run_id} baseline properties calculated. Set 0 baseline std: {getattr(run.sets[0], 'baseline_std', 'N/A')}")
    
    # 2. Define the Search Space
    # (Adjusted bounds to give window_size and t_drift_margin more room to breathe)
    space = [
        Real(2.0, 15.0, name='N_s1'),          
        Real(1.5, 12.0, name='N_s2'),           
        Integer(2, 20, name='window_size'),    
        Real(0.001, 0.05, name='threshold_bs'),
        Real(0.3, 0.9, name='t_drift_margin')  
    ]
    
    # 3. Run Optimization
    print(f"Starting Bayesian Optimization over {len(run.sets)} sets...")
    res = gp_minimize(
        lambda p: evaluate_timing_resolution(p, run=run, max_files=max_files), 
        space, 
        n_calls=20, 
        n_random_starts=10, 
        random_state=42,
        verbose=True
    )
    
    # 4. Save Configuration Data
    results = {
        "run_id": run_id,
        "el_field": getattr(run, 'el_field', None),
        "best_score": res.fun,
        "parameters": {
            "N_s1": res.x[0],
            "N_s2": res.x[1],
            "window_size": int(res.x[2]),
            "threshold_bs": res.x[3],
            "t_drift_margin": res.x[4]
        }
    }
    
    json_file = f"/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/timing_config_{run_id}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nOptimization complete. Config saved to {json_file}")
    
    # 5. Generate Visual Proof
    generate_validation_plots(res.x, run, run_id, max_files=10)

if __name__ == "__main__":

    """Command-line interface for optimizing timing parameters using Bayesian Optimization.
    Usage: python optimize_timing_BO.py --run <run_id> --config <config_path> --files <max_files>
    Example Run: python optimize_timing_BO.py --run 8 --config /path/to/run8_config.yaml --files 10
    """
    parser = argparse.ArgumentParser(description="Optimize S1/S2 timing parameters.")
    parser.add_argument("--run", type=str, required=True, help="Run ID to optimize")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--files", type=int, default=10, help="Max files per set to process")
    args = parser.parse_args()
    
    main(args.run, args.config, args.files)