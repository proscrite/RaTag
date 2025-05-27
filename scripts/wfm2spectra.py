from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
# from .wfm2read_fast import wfm2read
from RaTag.scripts.wfm2read_fast import wfm2read

def get_all_peaks(path):
    """Function to read waveforms and plot histogram of peaks."""
    
    files = glob(path + '/*.wfm')

    peaks = []
    for f in files:
        y, t, peak, info, ind_over, ind_under = wfm2read(f)
        peaks.append(peak)
    peaks = np.array(peaks)
    return peaks

def histogram_peaks(peaks):
    """Function to plot histogram of peaks."""
    plt.hist(peaks, bins=4, edgecolor='black')
    plt.xlabel('Peak Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peaks from Waveform Files')
    plt.show()


def get_total_gain(dial, coarse_gain = 50):
    Gmin = 0.5
    Gmax = 1.5
    dmin = 0
    dmax = 100
    Gfine = Gmin + (Gmax - Gmin) * (dial - dmin) / (dmax - dmin)
    Gtotal = Gfine * coarse_gain
    return Gtotal

def channel_to_voltage(channel, fine_gain = 63, coarse_gain = 50):
    """
    Convert a channel number to voltage and energy peak.
    :param channel: Channel number (0-2047)
    :param fine_gain: Fine gain setting (0-63)
    :param coarse_gain: Coarse gain setting (default 50)
    :return: Tuple of voltage in mV and energy peak in eV
    """
    if not (0 <= channel < 2048):
        raise ValueError("Channel must be between 0 and 2047")
    
    gain_amp = get_total_gain(fine_gain, coarse_gain)
    gain_preamp = 45 # mV / MeV preamp gain

    V = channel * 10 / 2048 # V
    V_mV = V * 1000 # mV

    E_peak = V_mV / (gain_amp * gain_preamp) # MeV

    print(f"V = {V:.2f} V")
    print(f"E_peak = {E_peak:.2f} MeV")
    return V_mV, E_peak

if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print("Usage: python wfm2spectra.py <path_to_wfm_files>")
        sys.exit(1)

    path = sys.argv[1]
    peaks = get_all_peaks(path)
    print(f"Found {len(peaks)} peaks in {path}")
    print(f"Peaks: {peaks}")
    histogram_peaks(peaks)