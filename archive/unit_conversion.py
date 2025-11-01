## Functions for Energy/Voltage/Channel conversion 
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# -- Functions for converting between energy, voltage, and channel numbers --

def get_total_gain(dial, coarse_gain = 50):
    """
    Calculate the total gain based on the fine gain dial and coarse gain setting.
    :param dial: Fine gain dial setting (0-100)
    :param coarse_gain: Coarse gain setting (default 50)
    :return: Total gain in mV/keV
    """
    Gmin = 0.5
    Gmax = 1.5
    dmin = 0
    dmax = 100
    Gfine = Gmin + (Gmax - Gmin) * (dial - dmin) / (dmax - dmin)
    Gtotal = Gfine * coarse_gain
    return Gtotal

def channel_to_voltage(channel):
    """
    Convert a channel number to voltage and energy peak.
    :param channel: Channel number (0-2047)
    :return: event voltage in mV
    """
    if not (0 <= channel < 2048):
        raise ValueError("Channel must be between 0 and 2047")
    
    V = channel * 10 / 2048 # V
    V_mV = V * 1000 # mV

    # print(f"V = {V:.2f} V")
    return V_mV

def voltage_to_energy(voltage, fine_gain = 63, coarse_gain = 50):
    """
    Convert a voltage in mV to energy peak in eV.
    :param voltage: Voltage in mV
    :param fine_gain: Fine gain setting (0-63)
    :param coarse_gain: Coarse gain setting (default 50)
    :return: Energy peak in eV
    """
    gain_amp = get_total_gain(fine_gain, coarse_gain)
    gain_preamp = 45 / 1000 # mV / keV preamp gain

    E_peak = voltage / (gain_amp * gain_preamp)# keV
    print(f"E_peak = {E_peak:.2f} KeV")
    return E_peak

def energy_to_voltage(energy, fine_gain = 63, coarse_gain = 50):
    """
    Convert an energy peak in eV to voltage in mV.
    :param energy: Energy peak in eV
    :param fine_gain: Fine gain setting (0-63)
    :param coarse_gain: Coarse gain setting (default 50)
    :return: Voltage in mV
    """
    gain_amp = get_total_gain(fine_gain, coarse_gain)
    gain_preamp = 45 / 1000 # mV / keV preamp gain

    voltage = energy * gain_amp * gain_preamp # mV
    # print(f"Voltage = {voltage:.2f} mV")
    return voltage

# -- Functions for reading and processing waveform files --

def load_mca_data(file_path):
    """
    Load MCA data from a MAESTRO .csv file.
    """
    df_mca = pd.read_csv(file_path, sep='\t', header=0, skiprows=11, nrows=2048, names=['counts'])
    return df_mca

def linear(x, a, b):
    return a * x + b

# reference chain energies as a list so order is guaranteed
DEFAULT_TH228_CHAIN = [
    ("Th228b", 5340.36),
    ("Th228a", 5423.15),
    ("Ra224a", 5685.37),
    ("Bi212", 6200.0),
    ("Rn220", 6405.0),
    ("Po216", 6906.0),
    ("Po212", 8954.0),
]

# -- Functions for simulating and calibrating spectra --
@dataclass
class Nuclide:
    name: str
    energy: float        # keV
    channel: float = None
    voltage: float = None  # mV

    def __repr__(self):
        return f"<Nuclide {self.name}: E={self.energy} keV → ch={self.channel:.1f}>"

class PeakFinder:
    def __init__(self, height=1000, distance=10, prominence=100):
        self.height = height
        self.distance = distance
        self.prominence = prominence

    def find(self, counts: np.ndarray):
        """
        Returns (indices, properties) from scipy.signal.find_peaks
        """
        return find_peaks(counts,
                          height=self.height,
                          distance=self.distance,
                          prominence=self.prominence)

class EnergyCalibrator:
    def __init__(self):
        self.coefficients = None  # (a, b) in channel = a*E + b

    def fit(self, energies: np.ndarray, channels: np.ndarray):
        popt, _ = curve_fit(linear, energies, channels, p0=[5.8, -100])
        self.coefficients = popt
        return popt

    def predict_channel(self, energies: np.ndarray):
        if self.coefficients is None:
            raise RuntimeError("Must call .fit() before predicting")
        return linear(energies, *self.coefficients)

    def predict_energy(self, channels: np.ndarray):
        if self.coefficients is None:
            raise RuntimeError("Must call .fit() before predicting")
        a, b = self.coefficients
        return (channels - b) / a

class SpectrumCalibrator:
    """
    Orchestrates loading, peak‐finding and calibration.
    """
    def __init__(self,
                 path: str,
                 nuclides=DEFAULT_TH228_CHAIN,
                 peak_finder: PeakFinder=None,
                 calibrator: EnergyCalibrator=None):
        self.path = path

        self.nuclides = { name: Nuclide(name, E) for name, E in nuclides }
        self.peak_finder = peak_finder or PeakFinder()
        self.calibrator = calibrator or EnergyCalibrator()
        self.data = None
        self.counts = None
        self.peak_indices = None
        self.peak_props = None

    def load(self):
        df = load_mca_data(self.path)
        self.data = df
        self.counts = df["counts"].values

    def find_peaks(self):
        self.peak_indices, self.peak_props = self.peak_finder.find(self.counts)
        if len(self.peak_indices) == 0:
            raise ValueError("No peaks found in the spectrum")
        
        if len(self.peak_indices) != len(self.nuclides):
            raise ValueError(f"Found {len(self.peak_indices)} peaks, but expected {len(self.nuclides)}, modify the peak_finder parameters")

    def calibrate(self):
        # pick as many top‐prominence peaks as nuclides, sorted by channel
        sorted_peaks = np.sort(self.peak_indices[:len(self.nuclides)])

        # keep the same insertion order as DEFAULT_TH228_CHAIN
        names    = list(self.nuclides.keys())
        energies = np.array([self.nuclides[n].energy for n in names])
        self.calibrator.fit(energies, sorted_peaks)
        predicted_chs = self.calibrator.predict_channel(energies)
        for name, ch in zip(names, predicted_chs):
            chf = float(ch)
            self.nuclides[name].channel = chf
            # compute the corresponding voltage (mV) on the MCA
            self.nuclides[name].voltage = channel_to_voltage(chf)

    def run(self):
        self.load()
        self.find_peaks()
        self.calibrate()
        return self.nuclides

    def get_energy_spectrum(self):
        """
        Return the full spectrum in energy‐keV vs counts.
        Must have called .run() (or at least .calibrate()) first.
        """
        import numpy as np
        if self.data is None or self.calibrator.coefficients is None:
            raise RuntimeError("Call .run() before getting energy spectrum")
        # channel axis
        channels = np.arange(len(self.data))
        # convert to energy [keV]
        energies = self.calibrator.predict_energy(channels)
        counts   = self.data['counts'].values
        return energies, counts

    ##########################################
    ### Monitoring and plotting methods    ###
    ##########################################

    def plot_spectrum(self, ax=None, peak_marker='x', **plot_kwargs):
        """
        Plot the raw MCA spectrum with detected peaks.
        """
        
        ax = ax or plt.gca()
        ax.plot(self.counts, **plot_kwargs)
        ax.plot(self.peak_indices,
                self.counts[self.peak_indices],
                peak_marker, label='Peaks')
        # annotate each peak with its nuclide name
        names = list(self.nuclides.keys())
        for name, idx in zip(names, self.peak_indices[:len(names)]):
            ax.text(x=idx, y=self.counts[idx], s=name, fontsize=12, 
                    ha='center', va='bottom')

        ax.set(xlabel='Channel',
               ylabel='Counts',
               title='Spectrum with Peaks')
        ax.legend()
        # add top x-axis in energy if calibrated
        if self.calibrator.coefficients is not None:
            a, b = self.calibrator.coefficients
            # channel → energy, inverse energy → channel
            to_energy  = lambda ch: (ch - b) / a
            to_channel = lambda E: a * E + b
            ax2 = ax.secondary_xaxis('top',
                                     functions=(to_energy, to_channel))
            ax2.set_xlabel('Energy (keV)')
        return ax

    def plot_calibration(self, ax=None, marker='o', line_kwargs=None, text_offset=(0.02, -0.02)):
        """
        Plot energy vs. channel reference points and overlay the linear fit.
        """
        ax = ax or plt.gca()
        names    = list(self.nuclides.keys())
        energies = np.array([self.nuclides[n].energy for n in names])
        channels = np.array([self.nuclides[n].channel for n in names])

        # plot reference points
        ax.plot(energies, channels, marker, label='Reference peaks')
        # annotate each point
        dx, dy = text_offset
        xspan  = energies.max() - energies.min()
        yspan  = channels.max() - channels.min()
        for name, e, ch in zip(names, energies, channels):
            ax.text(e + dx*xspan, ch + dy*yspan, name)

        # draw fit line
        xs = np.linspace(energies.min(), energies.max(), 200)
        ys = linear(xs, *self.calibrator.coefficients)
        ax.plot(xs, ys, **(line_kwargs or {'ls':'--','color':'b'}), label='Linear Fit')
 
        # add secondary y-axis: channel → voltage (V)
        # forward: channel_to_voltage, inverse: channel = voltage*2048/(10*1000)
        to_voltage = lambda ch: ch * 10.0 / 2048    # vectorized
        to_channel = lambda V: V * 2048 / 10.0
        ax2 = ax.secondary_yaxis('right', functions=(to_voltage, to_channel))
        ax2.set_ylabel('Voltage (V)')

        ax.set(xlabel='Energy (keV)',
               ylabel='Channel',
               title='Energy Calibration')
        ax.legend()
        return ax


