import numpy as np
from scipy.signal import savgol_filter
from typing import Optional, Any
from dataclasses import replace

from RaTag.core.config import AlphaCalibrationConfig
from RaTag.core.datatypes import Run, SetAlpha
from RaTag.core.decorators import *
from RaTag.core.functional import map_over
from RaTag.io import file_ops

# ============================================================================
# 1. VECTORIZED WAVEFORM PROCESSING
# ============================================================================
def _compute_alpha_energies(v_batch: np.ndarray, 
                            threshold_bs: float = 0.3, 
                            dither_amplitude: float = 0.02, 
                            savgol_window: int = 501, 
                            savgol_order: int = 3) -> np.ndarray:
    """
    Vectorized extraction of alpha peak energies using Savitzky-Golay filtering.
    """
    n_wfms, _ = v_batch.shape
    
    # 1. Vectorized baseline correction
    baselines = np.zeros(n_wfms, dtype=np.float32)
    for i in range(n_wfms):
        v_bs = v_batch[i, v_batch[i] < threshold_bs]
        baselines[i] = np.mean(v_bs) if len(v_bs) >= 10 else np.median(v_batch[i, :200])
    
    v_corrected = v_batch - baselines[:, np.newaxis]
    
    # 2. Vectorized dithering
    if dither_amplitude > 0:
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, size=v_batch.shape)
        v_dithered = v_corrected + dither
    else:
        v_dithered = v_corrected
    
    # 3. Apply Savitzky-Golay filter along time axis (axis=1)
    v_smooth = savgol_filter(v_dithered, savgol_window, savgol_order, axis=1)
    
    # 4. Find maximum for each waveform
    peak_values = v_smooth.max(axis=1)
    
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
                            savgol_window: int = 501,
                            force: bool = False) -> tuple[Any, dict]:
    """Executes reconstruction and formats the standard .npz arrays."""

    print("  🔹 Resolving alpha energies for:", set_alpha.source_dir.name)
    out_uids, out_energies = [], []
    
    for wf in file_ops.iter_alpha_waveforms(set_alpha, max_files=max_files,  show_progress=True):
        v_2d = wf.v if wf.ff else wf.v[np.newaxis, :]

        energies = _compute_alpha_energies(v_2d, savgol_window=savgol_window)
        
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
    savgol_window = getattr(config, 'savgol_window', 501) if config else 501

    bound_alphas = lambda s: resolve_alpha_energies(s, max_frames=max_frames, 
                                                    savgol_window=savgol_window, force=force)
    
    # Map over the independent alpha sets.
    updated_alpha_sets = map_over(run.alpha_sets, bound_alphas, catch_errors=True)
    
    return replace(run, alpha_sets=updated_alpha_sets)