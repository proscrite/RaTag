from scipy.signal import find_peaks, peak_widths
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict
from .datatypes import PMTWaveform, SetPmt, Run, XRayEvent, XRayResults
from .config import IntegrationConfig
from .dataIO import iter_waveforms, store_xrayset
from .units import s_to_us, V_to_mV
from .transformations import *

# -----
# --- Signal classification functions for single waveforms

def excessive_s2_check(wf: PMTWaveform, s2_start: float, bs_threshold: float, max_area_s2: float = 1e5, debug: bool = False ) -> tuple[bool, str]:
    """Check if waveform baseline before S2 is acceptable."""
    mask = (wf.t >= s2_start)

    Vs2 = wf.v[mask]
    if Vs2[Vs2 > bs_threshold].sum() > max_area_s2:
        if debug:
            print(f'Area above baseline before S2: {Vs2[Vs2 > bs_threshold].sum():.2e} (max allowed {max_area_s2})')
        return False, "excessive S2 signal above baseline"
    return True, "ok"

def separation_check(wf: PMTWaveform, t_s1: float, s2_start: float,
                     min_s2_sep: float, min_s1_sep: float) -> tuple[bool, str]:
    """Check if waveform has sufficient separation between S1 and S2."""
    t_thresh = wf.t[wf.v > 0] if np.any(wf.v > 0) else None
    if t_thresh is None:
        return False, "no signal above baseline"
    
    t_pre_s2 = s2_start - t_thresh[-1]
    t_post_s1 = t_thresh[0] - t_s1
    
    if not (t_pre_s2 > min_s2_sep and t_post_s1 > min_s1_sep):
        return False, f"insufficient separation (t_pre_s2={t_pre_s2:.2f}), (t_post_s1={t_post_s1:.2f})"
    return True, "ok"

def xray_event_pipeline(wf: PMTWaveform,
                        t_s1: float,
                        s2_start: float,
                        bs_threshold: float = 0.5,
                        max_area_s2: float = 1e5,
                        min_s2_sep: float = 1.0,
                        min_s1_sep: float = 0.5,
                        n_pedestal: int = 200,
                        ma_window: int = 10,
                        dt: float = 2e-4,
                        integrator: Callable[[PMTWaveform, float], np.ndarray] = integrate_trapz,
                        debug: bool = False
                        ) -> list[XRayEvent]:
    """Classify X-ray-like events in a waveform (supports FastFrame)."""

    wf_id = Path(wf.source).stem
    wf = t_in_us(wf)
    wf = v_in_mV(wf)
    wf = subtract_pedestal(wf, n_points=n_pedestal)

    events = []

    # If FastFrame, iterate over frames individually
    frames = enumerate(wf.v) if wf.ff else [(0, wf.v)]

    for frame_idx, v_frame in frames:
        # Build a single-frame view for processing
        frame_wf = replace(wf, v=v_frame, ff=False)

        # --- baseline check ---
        passed, reason = excessive_s2_check(frame_wf, s2_start, bs_threshold,
                                            max_area_s2=max_area_s2, debug=debug)
        if not passed:
            events.append(XRayEvent(wf_id=f"{wf_id}_ff{frame_idx}",
                                    accepted=False, reason=reason))
            continue

        # --- preprocessing for integration ---
        frame_wf = extract_window(frame_wf, t_s1, s2_start)
        frame_wf = moving_average(frame_wf, window=ma_window)
        frame_wf = threshold_clip(frame_wf, threshold=bs_threshold)

        # --- separation check ---
        passed, reason = separation_check(frame_wf, t_s1, s2_start,
                                          min_s2_sep, min_s1_sep)
        if not passed:
            events.append(XRayEvent(wf_id=f"{wf_id}_ff{frame_idx}",
                                    accepted=False, reason=reason))
            continue

        # --- integrate ---
        area = float(integrator(frame_wf, dt=dt))
        events.append(XRayEvent(wf_id=f"{wf_id}_ff{frame_idx}",
                                accepted=True, reason="ok", area=area))

    return events

# -----
# --- Set-level S1 estimation and X-ray classification
# -------------------------------------------------
def classify_xrays_set(set_pmt: SetPmt,
                       t_s1: float,
                       s2_start: float,
                       bs_threshold: float = 0.5,
                       max_area_s2: float = 1e5,
                       min_s2_sep: float = 1.0,
                       min_s1_sep: float = 0.5,
                       n_pedestal: int = 200,
                       ma_window: int = 10,
                       dt: float = 2e-4,
                       integrator: Callable[[PMTWaveform, float], np.ndarray] = integrate_trapz,
                       ) -> XRayResults:
    """
    Apply the X-ray event pipeline to all waveforms in a set.

    Args:
        set_pmt: Measurement set (lazy list of waveforms).
        t_s1: S1 time [µs].
        s2_start: expected S2 start [µs].
        bs_threshold: baseline threshold [mV].
        min_s2_sep: minimum required separation before S2 [µs].
        min_s1_sep: minimum required separation after S1 [µs].
        n_pedestal: number of samples for pedestal subtraction.
        ma_window: moving average window length (samples).
        dt: integration timestep [µs].
        integrator: function to perform integration (default trapezoidal).

    Returns:
        XRayResults with classification results for each waveform.
    """
    events: list[XRayEvent] = []

    for idx, wf in enumerate(iter_waveforms(set_pmt)):
        try:
            event = xray_event_pipeline(
                wf,
                t_s1=t_s1,
                s2_start=s2_start,
                bs_threshold=bs_threshold,
                max_area_s2=max_area_s2,
                min_s2_sep=min_s2_sep,
                min_s1_sep=min_s1_sep,
                n_pedestal=n_pedestal,
                ma_window=ma_window,
                dt=dt,
                integrator=integrator,
            )
            events.append(event)
        except Exception as e:
            # Mark waveform as failed classification
            events.append(XRayEvent(
                wf_id=str(idx),
                accepted=False,
                reason=f"error: {e}",
            ))

    xresults = XRayResults(
        set_id=set_pmt.source_dir,
        events=events,
        params={
            "t_s1": t_s1,
            "s2_start": s2_start,
            "bs_threshold": bs_threshold,
            "min_s2_sep": min_s2_sep,
            "min_s1_sep": min_s1_sep,
            "n_pedestal": n_pedestal,
            "ma_window": ma_window,
            "dt": dt,
            "integrator": integrator.__name__,
            "set_metadata": set_pmt.metadata,
        }
    )
    store_xrayset(xresults)
    return xresults
# -------------------------------------------------
# ---- Run-level x-ray integration 
# -------------------------------------------------

def classify_xrays_run(run: Run, ts2_tol = -2.7, range_sets: slice = None,
                     config: IntegrationConfig = IntegrationConfig() ) -> Dict[str, XRayResults]:
    """
    Classify X-ray events for all sets in a Run.
    Args:
        run: Run object with measurements populated.
        range_sets: slice to select subset of sets to process.
        kwargs: passed to classify_xrays_set.
    Returns:
        Dict mapping set_id -> XRayResults.
    """
    results = {}
    sets_to_process = run.sets[range_sets] if range_sets is not None else run.sets

    for set_pmt in sets_to_process:
        # Preconditions: set must already have t_s1 and time_drift
        if "t_s1" not in set_pmt.metadata or set_pmt.time_drift is None:
            raise ValueError(f"Set {set_pmt.source_dir} missing t_s1 or time_drift")

        t_s1 = set_pmt.metadata["t_s1"]   # µs
        t_drift = set_pmt.time_drift      # µs
        t_start = t_s1 + t_drift + ts2_tol # ad-hoc offset
        # t_end = t_start + run.width_s2   # Not needed in the x-ray pipeline here

        print(f"Processing x-rays in set {set_pmt.source_dir} in drift window: {(t_s1, t_start)}")


        
        results[set_pmt.source_dir.name] = classify_xrays_set(set_pmt, t_s1=t_s1, s2_start=t_start,
                                                             bs_threshold=config.bs_threshold,
                                                             max_area_s2=config.max_area_s2,
                                                             min_s2_sep=config.min_s2_sep,
                                                             min_s1_sep=config.min_s1_sep,
                                                             n_pedestal=config.n_pedestal,
                                                            ma_window=config.ma_window,
                                                             dt=config.dt,
                                                             integrator=config.integrator)
    return results
