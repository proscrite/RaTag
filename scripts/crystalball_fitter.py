import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import crystalball
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from RaTag.scripts.unit_conversion import load_mca_data

# --- existing single‐peak helpers --------------------------------------------

def _cb_pdf(x, A, beta, m, loc, scale):
    return A * crystalball.pdf(x, beta, m, loc, scale)

@dataclass
class FitCbResult:
    A: float
    beta: float
    m: float
    loc: float
    scale: float
    popt: np.ndarray
    pcov: np.ndarray

    @property
    def distribution(self):
        return crystalball(beta=self.beta,
                           m=self.m,
                           loc=self.loc,
                           scale=self.scale)

# --- new: vectorized CB for lmfit -------------------------------------------

import lmfit

def v_crystalball(x, N, beta, m, x0, sigma):
    """Vectorized Crystal Ball for lmfit (prefixes: N, beta, m, x0, sigma)."""
    absb = np.abs(beta)
    z    = (x - x0) / sigma
    gauss = np.exp(-0.5 * z**2)
    A_tail = (m/absb)**m * np.exp(-0.5*absb**2)
    B = m/absb - absb
    tail = A_tail / (B - z)**m
    return N * np.where(z > -absb, gauss, tail)

class CrystalBallFitter:
    """
    1-peak fit via scipy.curve_fit  OR
    multi-peak fit via lmfit.Model(v_crystalball)
    """
    def __init__(self, file_path: str, cutdown: int, cutup: int):
        self.file_path = file_path
        self.cutdown    = cutdown
        self.cutup     = cutup

        self.df_full  = None
        self.df_roi   = None
        self.fitres   = None    # FitCbResult
        self.multi_res= None    # lmfit.ModelResult

    def load(self):
        df = load_mca_data(self.file_path)
        df.index.name = 'channel'
        self.df_full = df
        return df

    def select_roi(self):
        if self.df_full is None:
            raise RuntimeError("Call .load() first")
        m = (self.df_full.index >= self.cutdown) & (self.df_full.index <= self.cutup)
        self.df_roi = self.df_full.loc[m].copy()
        return self.df_roi

    # --- single‐peak curve_fit ----------------------------------------------

    def fit(self, p0=None, bounds=None):
        if self.df_roi is None:
            raise RuntimeError("Call .select_roi() first")
        x = self.df_roi.index.values
        y = self.df_roi['counts'].values.astype(float)
        y /= y.max()

        p0     = p0 or [y.max(), 1.5, 1.5, x.mean(), (self.cutup - self.cutdown)/10]
        bounds = bounds or ([0,0,0, self.cutdown, 0],
                            [np.inf,np.inf,np.inf, self.cutup, np.inf])

        popt, pcov = curve_fit(_cb_pdf, x, y, p0=p0, bounds=bounds)
        self.fitres = FitCbResult(*popt, popt=popt, pcov=pcov)
        return self.fitres

    def plot_fit(self, ax=None):
        if self.df_roi is None or self.fitres is None:
            raise RuntimeError("Need .select_roi() and .fit() first")
        ax = ax or plt.gca()
        x = self.df_roi.index.values
        y = self.df_roi['counts'].values.astype(float);  y /= y.max()

        ax.plot(x, y, 'b-', label='Data (norm)')
        ax.plot(x, _cb_pdf(x, *self.fitres.popt), 'r--', label='1-peak fit')
        ax.set(xlabel='Channel', ylabel='Normalized Counts',
               title=f'ROI {self.cutdown}–{self.cutup}')
        ax.legend()
        return ax

    # --- new: multi-peak via lmfit -----------------------------------------

    def fit_multi(self,
                  n_peaks: int = 2,
                  p0: dict = None,
                  shared_params: bool = True):
        """
        Fit a sum of `n_peaks` CrystalBall peaks with lmfit.
        Returns and stores an lmfit.ModelResult in self.multi_res.
        """
        if self.df_roi is None:
            raise RuntimeError("Call .load() and .select_roi() first")
        x = self.df_roi.index.values
        y = self.df_roi['counts'].values.astype(float);  y /= y.max()

        # build composite lmfit model
        model = lmfit.Model(v_crystalball, prefix='cb1_')
        for i in range(2, n_peaks+1):
            model += lmfit.Model(v_crystalball, prefix=f'cb{i}_')

        # parameters & defaults
        params = model.make_params()
        # default gu e ss: spread centers evenly, unit amplitude, σ~(ROI/20)
        roi_width = self.cutup - self.cutdown
        locs = np.linspace(self.cutdown, self.cutup, n_peaks+2)[1:-1]
        for i, loc in enumerate(locs, 1):
            params[f'cb{i}_N'].set(value=1, min=0)
            params[f'cb{i}_x0'].set(value=loc, min=self.cutdown, max=self.cutup)
            params[f'cb{i}_beta'].set(value=1.150, min=0)
            params[f'cb{i}_m'].set(value=1.728, min=1)
            params[f'cb{i}_sigma'].set(value=7.329, min=1e-6)

        # override with user‐supplied p0 dict
        if p0:
            for k, v in p0.items():
                if k in params:
                    params[k].set(value=v)

        # tie β, m, σ together if requested
        if shared_params:
            for i in range(2, n_peaks+1):
                for par in ('beta','m','sigma'):
                    params[f'cb{i}_{par}'].expr = f'cb1_{par}'

        # do the fit
        self.multi_res = model.fit(y, params=params, x=x)
        return self.multi_res

    def plot_multi_fit(self, ax=None):
        """
        Overlay data + total lmfit fit + each crystalball component.
        """
        if self.df_roi is None or self.multi_res is None:
            raise RuntimeError("Need .select_roi() and .fit_multi() first")
        ax = ax or plt.gca()
        x = self.df_roi.index.values
        y = self.df_roi['counts'].values.astype(float);  y /= y.max()

        ax.plot(x, y, 'k-', label='Data')
        ax.plot(x, self.multi_res.eval(x=x), 'r--', label='Total fit')
        # each component
        comps = self.multi_res.eval_components(x=x)
        for prefix, comp in comps.items():
            ax.plot(x, comp, lw=1, label=f'{prefix}')
        ax.set(xlabel='Channel', ylabel='Normalized Counts',
               title=f'Multi-peak fit ({len(comps)} peaks)')
        ax.legend()
        return ax