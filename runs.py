from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple
from .measurement import SetPmt

@dataclass(frozen=True)
class Run:
    root_directory: Path
    run_id: str
    el_field: float
    target_isotope: str = "Th228"
    pressure: float = 2.0
    temperature: float = 293.0
    sampling_rate: float = 1e9
    el_gap: float = 0.8
    drift_gap: float = 1.4
    measurements: List[SetPmt] = None

    # Orchestrate cut params here
    bs_window: tuple[float, float] = (-1.5e-5, -1.0e-5)
    width_s2: float = 1e-5
    t_s1: float = 0.0  # can be refined by batch analysis

def run_from_root(
    root_directory: Path,
    run_id: str,
    el_field: float,
    pressure: float,
    sampling_rate: float,
    el_gap: float,
    drift_gap: float,
    t_s1: float,
    width_s2: float,
    bs_window: Tuple[float, float],
    discover_measurements: callable,  # pure: Path -> List[SetPmt]
) -> Run:
    sets = discover_measurements(root_directory)
    # push down run-level defaults to each SetPmt metadata if you want:
    sets2 = [replace(s, metadata={**s.metadata, "t_s1": t_s1, "width_s2": width_s2})
             for s in sets]
    return Run(root_directory, run_id, el_field, pressure, "Th228",
               sampling_rate, el_gap, drift_gap, sets2, width_s2, bs_window, t_s1)