from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple
from lmfit.models import GaussianModel  # type: ignore
from .measurement import integrate_set_s2, set_fields, set_transport_properties, estimate_s1_from_batches
from .fits import fit_s2area
from .datatypes import S2Areas, SetPmt



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
    sets: List[SetPmt] = None

    # Orchestrate cut params here
    bs_window: tuple[float, float] = (-1.5e-5, -1.0e-5)
    width_s2: float = 1e-5
    t_s1: float = 0.0  # can be refined by batch analysis

    def sets_from_dir(self) -> "Run":
        """
        Populate measurements with SetPmt objects from root_directory.
        Returns a new Run with measurements filled.
        """
        from .measurement import SetPmt
        sets = []
        for subdir in sorted(self.root_directory.iterdir()):
            if subdir.is_dir() and subdir.name.startswith(f"FieldScan"):
                try:
                    sets.append(SetPmt.from_directory(subdir))
                except Exception as e:
                    print(f"Skipping {subdir}: {e}")

        return replace(self, sets=sets)
    
    def with_transport_and_s1(self,
                          n_batches: int = 5,
                          batch_size: int = 20) -> "Run":
        """
        Return a new Run where each SetPmt has:
        - drift_field, EL_field, reduced fields
        - drift speed, drift time, diffusion coeff
        - estimated t_s1 from batches
        """
        updated_sets = []
        for s in self.sets:
            s1 = estimate_s1_from_batches(s, n_batches=n_batches, batch_size=batch_size)
            s1 = set_fields(s1, drift_gap=self.drift_gap, el_gap=self.el_gap, gas_density=4.91e19)
            s1 = set_transport_properties(s1, drift_gap=self.drift_gap, transport=None)
            updated_sets.append(s1)

        return replace(self, sets=updated_sets)


def integrate_run_s2(
    run: Run,
    n_pedestal: int = 2000,
    ma_window: int = 9,
    ts2_tol = 2.7e-6,
    threshold: float = 0.8,
) -> dict[str, S2Areas]:
    """
    Integrate S2 for all sets in a run, using pipeline parameters.

    Args:
        run: Run object (physics-level config).
        n_pedestal, ma_window, threshold: analysis parameters.

    Returns:
        dict keyed by set directory name -> S2Areas
    """
    results = {}

    for set_pmt in run.sets:
        # Preconditions: set must already have t_s1 and time_drift
        if "t_s1" not in set_pmt.metadata or set_pmt.time_drift is None:
            raise ValueError(f"Set {set_pmt.source_dir} missing t_s1 or time_drift")

        t_s1 = set_pmt.metadata["t_s1"]   # s
        t_drift = set_pmt.time_drift      # s
        t_start = (t_s1 + t_drift + ts2_tol) * 1e6  # µs
        t_end = t_start + run.width_s2 * 1e6  # µs
        t_window = (t_start, t_end) 
        print(f"Processing set {set_pmt.source_dir} with t_window: {t_window}")

        areas = integrate_set_s2(
            set_pmt,
            t_window=t_window,
            n_pedestal=n_pedestal,
            ma_window=ma_window,
            threshold=threshold,
            
        )

        results[set_pmt.source_dir.name] = S2Areas(
            set_id=set_pmt.source_dir.name,
            areas=areas,
            indices=list(range(len(areas))),
            method="s2_area_pipeline",
            params={
                "t_window": t_window,
                "n_pedestal": n_pedestal,
                "ma_window": ma_window,
                "threshold": threshold,
                "width_s2": run.width_s2,
                "set_metadata": set_pmt.metadata,
            },
        )

    return results


def fit_run_s2(
    run: Run, 
    s2areas: dict[str, S2Areas], 
    bin_cuts: tuple[float, float] = (0, 4),
    nbins: int = 100,
    exclude_index: int = 1,
    flag_plot: bool = False
) -> dict[str, S2Areas]:
    """
    Apply Gaussian fits to all S2Area results in a run.
    
    Args:
        run: Run object with SetPmt measurements.
        s2areas: dict of {set_id: S2Area} with raw integration results.
        bin_cuts: (min, max) range for histogram.
        nbins: number of bins.
        exclude_index: skip first bins if pedestal leak.
        flag_plot: whether to show diagnostic plots.
    
    Returns:
        dict of {set_id: S2Area} with fit results populated.
    """
    return {
        sid: fit_s2area(
            s2, 
            bin_cuts=bin_cuts,
            nbins=nbins,
            exclude_index=exclude_index,
            flag_plot=flag_plot
        )
        for sid, s2 in s2areas.items()
    }
