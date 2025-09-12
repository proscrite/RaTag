# RaTag/cuts.py
import numpy as np
from typing import Callable, List, Union
from dataclasses import replace, reduce
from .datatypes import PMTWaveform, SetPmt, RejectionLog

# -------------------------------
# Cut factories
# -------------------------------
def make_time_amplitude_cut(t_start: float, t_end: float = None,
                            threshold: float = 0.02, nmax_grass: int = 5):
    def cut_fn(wf):
        t1 = wf.t[-1] if t_end is None else t_end
        mask = (wf.t >= t_start) & (wf.t <= t1)
        tsel, Vsel = wf.t[mask], wf.v[mask]
        ok = int((Vsel > threshold).sum()) <= nmax_grass
        return ok, tsel, Vsel
    return cut_fn


def make_vertical_cut(vmin: float):
    """Reject if max(V) < vmin."""
    def cut_fn(wf):
        ok = wf.v.max() >= vmin
        return ok, wf.t[ok], wf.v[ok]
    return cut_fn


def moving_average(x, window_size=9):
    """Simple moving average filter"""
    return np.convolve(x, np.ones(window_size)/window_size, mode="same")

def make_smoothed_vertical_cut(vmin: float, t_start: float, t_end: float, smooth_window: int = 9):
    """Accept only values of smoothed wf over vmin"""
    def cut_fn(wf):
        s2_mask = (wf.t >= t_start) & (wf.t <= t_end)
        V_s2 = wf.v[s2_mask]
        t_s2 = wf.t[s2_mask]
        V_smooth = moving_average(V_s2, smooth_window)
        ok = V_smooth >= vmin
        return ok.any(), t_s2[ok], V_smooth[ok]
    return cut_fn



def drift_region_cut(set_pmt: SetPmt, t_tol: float = 1e-6, threshold: float = 0.02, nmax_grass: int = 5):
    t_s1 = set_pmt.metadata["t_s1"]
    t_drift = set_pmt.time_drift
    return make_time_amplitude_cut(t_start = t_s1 + t_tol, t_end = t_s1 + t_drift - t_tol, threshold=threshold, nmax_grass=nmax_grass)

def post_s2_cut(set_pmt: SetPmt, t_tol: float = 1e-6, width_s2: float = 1e-5, threshold: float = 0.02, nmax_grass: int = 5):
    t_s1 = set_pmt.metadata["t_s1"]
    t_drift = set_pmt.time_drift
    # width_s2 = set_pmt.metadata["width_s2"]
    t2_end = t_s1 + t_drift + width_s2 + t_tol

    return make_time_amplitude_cut(t_start=t2_end, t_end=None, threshold=threshold, nmax_grass=nmax_grass)

# -------------------------------
# Cut application and logging
# -------------------------------

def apply_cut(set_pmt: SetPmt,
              cut_fn: Callable[[PMTWaveform], bool],
              cut_name: str,
              reason: str = "") -> RejectionLog:
    """
    Apply a cut function to all waveforms in a SetPmt.

    Args:
        set_pmt: The set of waveforms to evaluate.
        cut_fn: A function that takes a PMTWaveform and returns True (pass) or False (reject).
        cut_name: A human-readable name for the cut.
        reason: Optional description of the physics motivation.

    Returns:
        RejectionLog with passed/rejected indices.
    """
    passed, rejected = [], []
    for idx, wf in enumerate(set_pmt.iter_waveforms()):
        ok, _, _ = cut_fn(wf)
        (passed if ok else rejected).append(idx)
    return RejectionLog(cut_name, cut_fn, passed, rejected)

def filter_set(set_pmt: SetPmt, logs: Union[RejectionLog, List[RejectionLog]]) -> SetPmt:
    """
    Return a new SetPmt containing only filenames that passed the cut(s).
    Accepts a single RejectionLog or a list of them.
    If a list is given, waveforms must pass ALL cuts to be included.
    """
    if isinstance(logs, RejectionLog):
        passed = set(logs.passed)
    else:  # list of logs
        passed_sets = [set(log.passed) for log in logs]
        passed = set.intersection(*passed_sets) if passed_sets else set()

    new_files = [fn for idx, fn in enumerate(set_pmt.filenames) if idx in passed]
    return replace(set_pmt, filenames=new_files)

# def filter_set(set_pmt: SetPmt, log: RejectionLog) -> SetPmt:
#     keep = [set_pmt.filenames[i] for i in log.passed]
#     return replace(set_pmt, filenames=keep)

def combine_logs(logs: List[RejectionLog]) -> RejectionLog:
    if not logs:
        return RejectionLog("all_cuts", [], [], lambda wf: True, reason="No cuts")

    passed = sorted(set.intersection(*(set(log.passed) for log in logs)))
    rejected = sorted(set.union(*(set(log.rejected) for log in logs)))

    # Combined cut: apply all cuts in sequence
    def combined_cut(wf: PMTWaveform) -> bool:
        return all(log.cut_fn(wf) for log in logs)

    return RejectionLog(
        cut_name="all_cuts",
        passed=passed,
        rejected=rejected,
        cut_fn=combined_cut,
        reason="Passed all cuts"
    )
# -----------------------------------------------
# S2 area integration pipeline
# -----------------------------------------------


# For reference, previous version of evaluate_cuts:

# def evaluate_cuts(set_pmt: SetPmt,
#                   bs_window: tuple[float, float],
#                   t_tol: float = 1e-6,
#                   threshold: float = 0.02,
#                   nmax_grass: int = 5,
#                   width_s2: float = 1e-5) -> list[RejectionLog]:
#     """Apply all defined cuts and return their logs."""
#     baseline = make_baseline_cut(bs_window)
#     drift_cut = drift_region_cut(set_pmt, t_tol, threshold, nmax_grass)
#     post_s2 = post_s2_cut(set_pmt, t_tol, width_s2, threshold, nmax_grass)

#     logs = [
#         apply_cut(set_pmt, baseline, "baseline", "Stable baseline"),
#         apply_cut(set_pmt, drift_cut, "drift_region", "Clean drift region"),
#         apply_cut(set_pmt, post_s2, "post_s2", "No noise after S2"),
#     ]
#     return logs


# -------------------------------
# High-level pipeline function
# -------------------------------

def pipe(value, *funcs):
    """Apply functions sequentially to a value."""
    return reduce(lambda v, f: f(v), funcs, value)
