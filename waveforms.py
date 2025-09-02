# RaTag/waveforms.py
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

@dataclass
class Waveform:
    """Generic waveform with common properties."""
    t: np.ndarray          # time axis
    v: np.ndarray          # voltage (signal)
    source: Optional[str]  # filename or run ID (for traceability)

    def __len__(self):
        return len(self.t)

    def slice(self, t_min: float, t_max: float) -> "Waveform":
        """Return a sub-waveform within [t_min, t_max]."""
        mask = (self.t >= t_min) & (self.t <= t_max)
        return Waveform(self.t[mask], self.v[mask], source=self.source)

    def baseline(self, t_min: float, t_max: float) -> float:
        """Compute baseline average in [t_min, t_max]."""
        return self.slice(t_min, t_max).v.mean()

    def area(self, t_min: float, t_max: float) -> float:
        """Compute area under the curve (trapezoid)."""
        w = self.slice(t_min, t_max)
        return np.trapz(w.v, w.t)
    
    def plot(self, ax=None, **kwargs):
        """Plot waveform."""
        if ax is None:
            ax = plt.gca()
        ax.plot(self.t, self.v, **kwargs)
        ax.set(xlabel='Time (µs)', ylabel='Signal (V)', title=Path(self.source).name if self.source else 'Waveform')
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-6, -6))
        return ax


@dataclass
class PMTWaveform(Waveform):
    """PMT-specific waveform with PMT calibration info."""
    gain: Optional[float] = None

    def to_photoelectrons(self, t_min: float, t_max: float) -> float:
        """Convert integrated charge to photoelectrons (if gain set)."""
        area = self.area(t_min, t_max)
        return area / self.gain if self.gain else area


@dataclass
class SiliconWaveform(Waveform):
    """PIN diode waveform."""
    sensitivity: Optional[float] = None

    def to_energy(self, t_min: float, t_max: float) -> float:
        """Convert signal to energy (if sensitivity set)."""
        area = self.area(t_min, t_max)
        return area * self.sensitivity if self.sensitivity else area
