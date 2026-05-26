import numpy as np
from dataclasses import replace
from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.functional import map_over
from RaTag.core.decorators import disk_cache, limit_frames
from RaTag.io.file_ops import iter_waveforms

# Import the low-level math function
from RaTag.el_tpc.waveform_features import compute_waveform_baseline

# ============================================================================
# SET-LEVEL WORKFLOW (One-Pass Logic)
# ============================================================================
@disk_cache(target_attr='baseline_std')
@limit_frames
def resolve_set_baseline(set_pmt: SetPmt, max_files: int, n_points: int = 200, force: bool = False) -> SetPmt:
    """
    Resolves the baseline median and standard deviation for a single set 
    by sampling the pre-trigger region of the first few waveforms.
    """
    medians, stds = [], []
    
    for wf in iter_waveforms(set_pmt, max_files=max_files):
        b_med, b_std = compute_waveform_baseline(wf, n_points=n_points)
        medians.append(b_med)
        stds.append(b_std)
        
    if not medians:
        print(f"Warning: No waveforms found for set {set_pmt.source_dir.name}. Defaulting baselines to 0.0")
        return replace(set_pmt, baseline_median=0.0, baseline_std=0.0)
        
    # Aggregate robustly using the median of our samples
    set_baseline_median = round(float(np.median(medians)), 5)
    set_baseline_std = round(float(np.median(stds)), 5)
    
    print(f"Set {set_pmt.source_dir.name} baseline resolved: median={set_baseline_median}, std={set_baseline_std} (from {len(medians)} files)")
    
    return replace(set_pmt, baseline_median=set_baseline_median, baseline_std=set_baseline_std)

# ============================================================================
# PUBLIC API (The "map" functions called by pipeline.py)
# ============================================================================

def map_run_baseline(run: Run, max_frames: int = 480, n_points: int = 200, force: bool = False) -> Run:
    """
    Entry point: Maps the baseline calculation workflow over all sets in the Run.
    """
    bound_baseline = lambda s: resolve_set_baseline(s, max_frames=max_frames, n_points=n_points, force=force)
    
    new_sets = map_over(run.sets, bound_baseline, catch_errors=True)
    return replace(run, sets=new_sets)