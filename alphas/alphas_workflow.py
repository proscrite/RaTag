import numpy as np
from scipy.signal import savgol_filter
from typing import Optional, Any
from dataclasses import replace

from RaTag.core.config import AlphaCalibrationConfig
from RaTag.core.datatypes import Run, SetAlpha, Waveform
from RaTag.core.decorators import *
from RaTag.core.functional import map_over

from RaTag.waveform.preprocessing import *
from RaTag.io import file_ops

# ============================================================================
# 1. VECTORIZED WAVEFORM PROCESSING
# ============================================================================
def _compute_alpha_energies(wf: Waveform,
                            threshold_bs: float = 0.3, 
                            window_ma: int = 1000) -> np.ndarray:
    """
    Vectorized extraction of alpha peak energies using Savitzky-Golay filtering.
    """

    wf_sub = subtract_thresholded_pedestal(wf, threshold_bs=threshold_bs)
    wf_smooth = moving_average(wf_sub, window=window_ma)

    # 4. Find maximum for each waveform
    v_smooth = wf_smooth.v if wf_smooth.ff else wf_smooth.v[np.newaxis, :]
    peak_values = np.max(v_smooth, axis=1) if wf_smooth.ff else np.max(v_smooth, axis=0)

    # 5. Apply instrumental calibration factor
    energies = peak_values / 1.058
    return energies

@allow_force
@load_cached_metadata(target_attr='n_alpha_energies')
@load_cached_npz(signal_type='alpha_energies')
@write_metadata(target_attr='n_alpha_energies')
@write_npz_arrays(signal_type='alpha_energies')
@limit_frames
def resolve_alpha_energies(set_alpha: SetAlpha, 
                            max_files: Optional[int] = None, 
                            config: AlphaCalibrationConfig = AlphaCalibrationConfig(),
                            force: bool = False) -> tuple[Any, dict]:
    """Executes reconstruction and formats the standard .npz arrays."""

    print("  🔹 Resolving alpha energies for:", set_alpha.source_dir.name)
    out_uids, out_energies = [], []
    
    for wf in file_ops.iter_alpha_waveforms(set_alpha, max_files=max_files,  show_progress=True):

        energies = _compute_alpha_energies(wf, threshold_bs=config.threshold_bs, 
                                            window_ma=config.window_ma)
        
        out_uids.append(wf.uids)
        out_energies.append(energies)
        
    if not out_energies:
        raise ValueError(f"No alpha waveforms processed for {set_alpha.source_dir.name}.")
        
    energies = np.concatenate(out_energies)
    uids = np.concatenate(out_uids)
    
    updated_set = replace(set_alpha, n_alpha_energies=len(energies))
    
    arrays = {
        "energies": energies,
        "uids": uids
    }
    
    return updated_set, arrays

# ============================================================================
# 2. ORCHESTRATOR
# ============================================================================

def map_alpha_events(run: Run, 
                     max_frames: Optional[int] = None,
                     config: Optional[Any] = AlphaCalibrationConfig() or None, 
                     force: bool = False) -> Run:
    """Entry point: Maps energy reconstruction over independent alpha sets."""
    print("\n" + "="*60 + f"\nRECONSTRUCTING ALPHA ENERGIES: {run.run_id}\n" + "="*60)
    
    if not getattr(run, 'alpha_sets', None):
        print("  ⚠ No alpha_sets found in Run. Bootstrap the run first.")
        return run
    print(f"Force flag: {force}")
    

    bound_alphas = lambda s: resolve_alpha_energies(s, max_frames=max_frames, 
                                                    config=config, force=force)
    
    # Map over the independent alpha sets.
    updated_alpha_sets = map_over(run.alpha_sets, bound_alphas, catch_errors=True)
    
    return replace(run, alpha_sets=updated_alpha_sets)