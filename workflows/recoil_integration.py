"""Recoil (S2) signal analysis."""
from typing import Dict, Callable

from core.config import IntegrationConfig,  FitConfig 
from core.datatypes import PMTWaveform, SetPmt, S2Areas, Run
from core.dataIO import iter_waveforms
from waveform.preprocessing import subtract_pedestal, moving_average, threshold_clip
from waveform.s1s2_detection import extract_window
from waveform.integration import integrate_trapz

# ============================================================================
# WAVEFORM LEVEL
# ============================================================================

def integrate_s2_frame(wf: PMTWaveform, s2_start: float, s2_end: float, 
                      config: IntegrationConfig) -> float:
    """Integrate S2 signal in a single frame."""
    # Preprocess
    wf = subtract_pedestal(wf, n_points=config.n_pedestal)
    wf = moving_average(wf, window=config.ma_window)
    wf = threshold_clip(wf, threshold=config.bs_threshold)
    
    # Extract window and integrate
    wf_s2 = extract_window(wf, s2_start, s2_end)
    areas = config.integrator(wf_s2, config.dt)
    return areas[-1]  # Total area

# ============================================================================
# SET LEVEL
# ============================================================================

def integrate_s2_set(set_pmt: SetPmt,
                     t_window: tuple[float, float],
                     integration_config: IntegrationConfig) -> S2Areas:
    
    """
    Apply the S2 area pipeline to all waveforms in a set.

    Args:
        set_pmt: Measurement set (lazy list of waveforms).
        t_window: (t_start, t_end) defining S2 window in seconds.
        n_pedestal: number of samples to average for pedestal subtraction.
        ma_window: moving average window length (samples).
        bs_threshold: threshold for clipping voltages above baseline.
        dt: time step [µs] for Riemann integration.

    Returns:
        S2Areas object with raw integration results.
    """
    areas = []

    for idx, wf in enumerate(iter_waveforms(set_pmt)):
        try:
            area = integrate_s2_frame(wf, s2_start=t_window[0], s2_end=t_window[1],
                                        config=integration_config)
            areas.append(area)
        except Exception as e:
            # Optionally, handle bad waveforms gracefully
            # (e.g., append np.nan to keep indexing aligned)
            areas.append(np.nan)

    areas = np.array(areas)
    areas = areas.flatten()  # Ensure 1D array (for FastFrame, etc.)
    s2areas = S2Areas(source_dir=set_pmt.source_dir,
                      areas=areas,
                      method="s2_area_pipeline",
                      params={
                        "t_window": t_window,
                        "width_s2": t_window[1] - t_window[0],
                        "n_pedestal": integration_config.n_pedestal,
                        "ma_window": integration_config.ma_window,
                        "bs_threshold": integration_config.bs_threshold,
                        "dt": integration_config.dt,
                        "set_metadata": set_pmt.metadata,
                    }
            )
    return s2areas

# ============================================================================
# RUN LEVEL
# ============================================================================

def integrate_s2_run(run: Run, ts2_tol = -2.7, range_sets: slice = None,
                     integration_config: IntegrationConfig = IntegrationConfig(),
                     use_estimated_s2_windows: bool = True) -> Dict[str, S2Areas]:
    """
    Integrate S2 areas for all sets in a Run.

    Args:
        run: Run object with measurements populated.
        ts2_tol: Time tolerance before S2 window start (µs). Ignored if use_estimated_s2_windows=True.
        range_sets: Optional slice to select subset of sets.
        integration_config: Integration configuration parameters.
        use_estimated_s2_windows: If True, use S2 window statistics from metadata 
                                  (from s2_variance_run). If False, use t_drift + ts2_tol.

    Returns:
        Dict mapping set_id -> S2Areas.
    """
    results = {}
    sets_to_process = run.sets[range_sets] if range_sets is not None else run.sets

    for set_pmt in sets_to_process:
        # Preconditions: set must already have t_s1 and time_drift
        if "t_s1" not in set_pmt.metadata or set_pmt.time_drift is None:
            raise ValueError(f"Set {set_pmt.source_dir} missing t_s1 or time_drift")

        # Determine S2 integration window
        if use_estimated_s2_windows and 't_s2_start_mean' in set_pmt.metadata:
            # Use estimated S2 timing from metadata
            t_start = set_pmt.metadata['t_s2_start_mean']
            t_end = set_pmt.metadata['t_s2_end_mean']
            t_window = (t_start, t_end)
            print(f"Processing {set_pmt.source_dir.name} with estimated S2 window: {t_window}")
        else:
            # Fallback to original method: t_drift + ts2_tol
            t_s1 = set_pmt.metadata["t_s1"]   # µs
            t_drift = set_pmt.time_drift      # µs
            t_start = t_s1 + t_drift + ts2_tol
            t_end = t_start + run.width_s2  
            t_window = (t_start, t_end)
            
            if use_estimated_s2_windows:
                print(f"⚠ Warning: {set_pmt.source_dir.name} missing S2 window estimates, using fallback: {t_window}")
            else:
                print(f"Processing {set_pmt.source_dir.name} with calculated t_window: {t_window}")
        
        results[set_pmt.source_dir.name] = integrate_s2_set(set_pmt, t_window, 
                                                            integration_config=integration_config)

    return results


# ============================================================================
# Fits (move out to fitting.py?)
# ============================================================================

def fit_s2_run(run: Run, s2_areas: Dict[str, S2Areas],
              fit_config: FitConfig) -> Dict[str, FitResult]:
    """Fit S2 distributions for all sets."""
    ...




