"""
Visualization functions for activity analysis.
"""
from typing import List, Tuple
from matplotlib.pylab import ylabel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from RaTag.alphas.activity_analysis import ActivityMeasurement

def plot_activity_timeseries(measurements: List[ActivityMeasurement],
                             figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Plot counts and activity vs time since first measurement on twin axes.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    t0 = measurements[0].timestamp
    t = np.array([(m.timestamp - t0) / 3600.0 for m in measurements])  # Hours since start
    
    counts = np.array([m.counts for m in measurements])
    A = np.array([m.activity for m in measurements])
    A_err = np.array([m.activity_err for m in measurements])
    isotope = measurements[0].isotope
    
    color1 = "tab:blue"
    ax1.errorbar(t, counts, fmt='o', color=color1, label='Net Counts')
    ax1.set(xlabel="Time since first measurement [hours]", ylabel="Net Counts", title=f"Activity Time Series for {isotope}")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.errorbar(t, A, yerr=A_err, fmt='s', color=color2, label='Activity (Bq)')
    ax2.set(ylabel=f"Activity of {isotope} [Bq]")
    ax2.tick_params(axis='y', labelcolor=color2)

    return fig
