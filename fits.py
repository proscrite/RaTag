import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace 
from lmfit.models import GaussianModel # type: ignore
from .datatypes import S2Areas


def fit_s2area(s2: S2Areas,
               bin_cuts: tuple[float, float] = (0, 4),
               nbins: int = 100,
               exclude_index: int = 1,
               flag_plot = False) -> S2Areas:
    """
    Fit Gaussian to the S2 area distribution.
    
    Args:
        s2: S2Area object with raw areas.
        bin_cuts: (min, max) range for histogram.
        nbins: number of bins.
        exclude_index: skip first bins if pedestal leak.
    
    Returns:
        New S2Area with fit results populated.
    """
    area_vec = s2.areas[(s2.areas > bin_cuts[0]) & (s2.areas < bin_cuts[1])]
    if len(area_vec) == 0:
        return replace(s2,
                       mean=None,
                       sigma=None,
                       ci95=None,
                       fit_success=False)

    n, bins = np.histogram(area_vec, bins=nbins)
    
    # plt.close()  # avoid showing histogram here
    cbins = 0.5 * (bins[1:] + bins[:-1])
    n = n[exclude_index:]
    cbins = cbins[exclude_index:]

    model = GaussianModel()
    guessed = model.guess(n, x=cbins)
    params = model.make_params(amplitude=guessed['amplitude'].value, 
                               center=np.mean(cbins), sigma=np.std(cbins))
    # params = model.make_params(amplitude=n.max(),
    #                            center=np.mean(cbins),
    #                            sigma=np.std(cbins))
    result = model.fit(n, params, x=cbins)

    mean  = result.params['center'].value
    sigma = result.params['sigma'].value
    stderr = result.params['center'].stderr
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
                   fit_success=result.success,
                   fit_result=result)  # Store the fit result