from typing import Dict
from dataclasses import replace
import numpy as np
from lmfit.models import GaussianModel # type: ignore
import matplotlib.pyplot as plt # type: ignore

from .dataIO import iter_waveforms
from .datatypes import SetPmt, S2Areas, Run
from .transformations import s2_area_pipeline
from .config import IntegrationConfig, FitConfig

def integrate_set_s2(set_pmt: SetPmt,
                     t_window: tuple[float, float],
                     n_pedestal: int = 2000,
                     ma_window: int = 9,
                     dt: float = 2e-4,
                     threshold: float = 0.8) -> S2Areas:
    
    """
    Apply the S2 area pipeline to all waveforms in a set.

    Args:
        set_pmt: Measurement set (lazy list of waveforms).
        t_window: (t_start, t_end) defining S2 window in seconds.
        n_pedestal: number of samples to average for pedestal subtraction.
        ma_window: moving average window length (samples).
        threshold: threshold for clipping voltages above baseline.
        dt: time step [µs] for Riemann integration.

    Returns:
        S2Areas object with raw integration results.
    """
    areas = []

    for idx, wf in enumerate(iter_waveforms(set_pmt)):
        try:
            area = s2_area_pipeline(wf, t_window,
                                    n_pedestal=n_pedestal,
                                    ma_window=ma_window,
                                    threshold=threshold,
                                    dt=dt)
            areas.append(area)
        except Exception as e:
            # Optionally, handle bad waveforms gracefully
            # (e.g., append np.nan to keep indexing aligned)
            areas.append(np.nan)

    return S2Areas(
        source_dir=set_pmt.source_dir,
        areas=np.array(areas),
        method="s2_area_pipeline",
        params={
            "t_window": t_window,
            "n_pedestal": n_pedestal,
            "ma_window": ma_window,
            "threshold": threshold,
            "dt": dt,
            "width_s2": t_window[1] - t_window[0],
            "set_metadata": set_pmt.metadata,
            }
        )

# -------------------------------------------------
# Run-level integration
# -------------------------------------------------
def integrate_run_s2(run: Run, ts2_tol = 2.7,
                     config: IntegrationConfig = IntegrationConfig() ) -> Dict[str, S2Areas]:
    """
    Integrate S2 areas for all sets in a Run.

    Args:
        run: Run object with measurements populated.
        kwargs: passed to integrate_set_s2.

    Returns:
        Dict mapping set_id -> S2Areas.
    """
    results = {}

    for set_pmt in run.sets:
        # Preconditions: set must already have t_s1 and time_drift
        if "t_s1" not in set_pmt.metadata or set_pmt.time_drift is None:
            raise ValueError(f"Set {set_pmt.source_dir} missing t_s1 or time_drift")

        t_s1 = set_pmt.metadata["t_s1"]   # µs
        t_drift = set_pmt.time_drift      # µs
        t_start = t_s1 + t_drift + ts2_tol
        t_end = t_start + run.width_s2  
        t_window = (t_start, t_end) 

        print(f"Processing set {set_pmt.source_dir} with t_window: {t_window}")

        
        results[set_pmt.source_dir.name] = integrate_set_s2(set_pmt, t_window, 
                                                            n_pedestal=config.n_pedestal,
                                                           ma_window=config.ma_window,
                                                           threshold=config.threshold,
                                                           dt=config.dt)

    return results


# -------------------------------------------------
# Set-level Gaussian fit
# -------------------------------------------------
def fit_set_s2(s2: S2Areas,
               bin_cuts: tuple[float, float] = (0, 4),
               nbins: int = 100,
               exclude_index: int = 1,
               flag_plot: bool = False) -> S2Areas:
    """
    Fit Gaussian to S2Areas distribution of a single set.

    Args:
        s2: S2Areas object with raw areas.
        bin_cuts: (min, max) for histogram.
        nbins: number of bins.
        exclude_index: skip first bins if pedestal leak.
        flag_plot: if True, show histogram and fit.

    Returns:
        New S2Areas with fit results populated.
    """
    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        return replace(s2, fit_success=False)

    n, bins = np.histogram(area_vec, bins=nbins)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    n, cbins = n[exclude_index:], cbins[exclude_index:]

    model = GaussianModel()
    params = model.make_params(amplitude=n.max(),
                               center=np.mean(cbins),
                               sigma=np.std(cbins))
    result = model.fit(n, params, x=cbins)

    mean = result.params["center"].value
    sigma = result.params["sigma"].value
    stderr = result.params["center"].stderr
    ci95 = 1.96 * stderr if stderr else None

    if flag_plot:
        n, bins, _ = plt.hist(area_vec, bins=nbins, alpha=0.6, color='g', label="Data")
        plt.plot(cbins, model.eval(x=cbins, params=result.params), 'r--', label='fit')
        plt.gca().set(xlabel = 'Area (mV·ns)', ylabel = 'Counts')
        plt.legend()

    return replace(s2,
                   mean=mean,
                   sigma=sigma,
                   ci95=ci95,
                   fit_result=result,
                   fit_success=result.success)


# -------------------------------------------------
# Run-level Gaussian fit
# -------------------------------------------------
def fit_run_s2(areas: Dict[str, S2Areas], config: FitConfig = FitConfig()) -> Dict[str, S2Areas]:
    """
    Apply Gaussian fits across all sets in a run.

    Args:
        areas: Dict of {set_id: S2Areas} with raw areas.
        kwargs: passed to fit_set_s2.

    Returns:
        Dict of {set_id: S2Areas} with fit results.
    """
    return {sid: fit_set_s2(s2, 
                            bin_cuts=config.bin_cuts,
                            nbins=config.nbins,
                            exclude_index=config.exclude_index)
                              for sid, s2 in areas.items()}