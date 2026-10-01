import numpy as np
import matplotlib.pyplot as plt

from RaTag.core.datatypes import SetPmt, Run
from RaTag.core.functional import map_over
from RaTag.io.file_ops import load_npz_arrays
from RaTag.core.decorators import *
from RaTag.el_tpc.fit_s2_area import _find_dynamic_lower_bound

@write_plots(subfolder="width_filter_qa")
def plot_width_filter_diagnostics(set_pmt: SetPmt, s2_widths: np.ndarray, mid_valley: float,
                                  s2_areas: np.ndarray, filtered_areas: np.ndarray, one_minus_areas: np.ndarray):
    """ Plot diagnostics for S2 width filtering. """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    ax[0].hist(s2_widths, bins=100);
    ax[0].axvline(x=mid_valley, color='red', linestyle='--', label=f'Mid-Valley Width Threshold ({mid_valley:.2f} us)')
    ax[0].set(xlabel='S2 Width (us)', ylabel='Counts', title='S2 Width Distribution, E_drift = 2000 V/cm')
    ax[0].legend()

    ax[1].hist(s2_areas, bins=100, alpha=0.3, range=(0, np.mean(filtered_areas) * 3), label='All Areas')
    ax[1].hist(filtered_areas, bins=100, alpha=0.5, range=(0, np.mean(filtered_areas) * 3), label=f'Filtered Areas (<{mid_valley:.2f} us)')
    ax[1].hist(one_minus_areas, bins=100, alpha=0.5, range=(0, np.mean(filtered_areas) * 3), label=f'Excluded Areas (>{mid_valley:.2f} us)')
    ax[1].set(xlabel='S2 Area (mV*us)', ylabel='Counts', title=f'S2 Area Distribution by Width Category {set_pmt.name}')
    ax[1].legend()

    return s2_widths, fig       # The first return value is a placeholder to satisfy the @write_plots decorator, which expects a tuple of (data, figure).

@allow_force
@require_attributes('t_s2_start')
@write_metadata(target_attr='n_areas_recoil')
@write_npz_arrays(signal_type='s2_areas')
def filter_by_s2_width(set_pmt: SetPmt , max_lower_bound: float = 3) -> tuple[SetPmt, dict]:
    """
    Filter S2 events based on their widths using a dynamic lower bound.
    """
    s2_areas_file = load_npz_arrays(set_pmt, 's2_areas')
    s2_times = load_npz_arrays(set_pmt, 'timing')
    s2_widths = s2_times['t_s2_end'] - s2_times['t_s2_start']
    counts, bins = np.histogram(s2_widths, bins=100)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    mid_valley = _find_dynamic_lower_bound(cbins, counts, max_lower_bound)
    peak_pos = cbins[np.argmax(counts)]

    if mid_valley <= peak_pos:
        raise ValueError("Dynamic lower bound should be greater than the peak position.")

    area_uids = s2_areas_file['uids']
    time_uids = s2_times['uids']
    filtered_widths = s2_widths[s2_widths < mid_valley]
    filtered_uids = time_uids[s2_widths < mid_valley]
    filtered_areas = s2_areas_file['s2_areas'][np.isin(area_uids, filtered_uids)]
    one_minus_areas = s2_areas_file['s2_areas'][~np.isin(area_uids, filtered_uids)]

    plot_width_filter_diagnostics(set_pmt, s2_widths, mid_valley, s2_areas_file['s2_areas'], filtered_areas, one_minus_areas)

    n_areas_recoil = len(filtered_areas)
    stats = s2_areas_file['stats']
    stats['pass_width_filter'] = n_areas_recoil
    area_arrays = {
                "s2_areas": filtered_areas,
                "uids": filtered_uids,
                "stats": stats
            }
    updated_set = replace(set_pmt, n_areas_recoil=n_areas_recoil)
    return updated_set, area_arrays

def map_width_filter(run: Run, max_lower_bound: float = 3, force: bool = False) -> Run:
    """
    Apply S2 width filtering to all sets in the run.
    """
    print("\n" + "="*60)
    print(f"FILTERING BY S2 WIDTHS: {run.run_id}")
    print("="*60)

    bound_filter = lambda set_pmt: filter_by_s2_width(set_pmt, max_lower_bound=max_lower_bound, force=force)
    updated_sets = map_over(run.sets, bound_filter, catch_errors=True)
    
    return replace(run, sets=updated_sets)