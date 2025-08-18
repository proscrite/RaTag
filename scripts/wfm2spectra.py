from glob import glob
import numpy as np
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# from .wfm2read_fast import wfm2read
from RaTag.scripts.wfm2read_fast import wfm2read

def get_peak_volts(file_path):
    """
    Function to read all waveform files in a directory and extract peaks.
    :param file_path: Path to the directory containing waveform files.
    :return: List of peaks found in the waveform files.
    """
    y, t, info, ind_over, ind_under = wfm2read(file_path, step=0.001)

    peaks, props = find_peaks(y, height=1, distance=1, prominence=1)
    volts = y[peaks]
    return volts

def histogram_voltages(volts):
    """Function to plot histogram of peaks."""
    plt.hist(volts, bins=100, edgecolor='black')
    plt.xlabel('Peak Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peaks from Waveform Files')
    plt.show()


if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print("Usage: python wfm2spectra.py <path_to_wfm_files>")
        sys.exit(1)

    path = sys.argv[1]
    peaks = get_peak_volts(path)
    print(f"Found {len(peaks)} peaks in {path}")
    print(f"Peaks: {peaks}")
    histogram_voltages(peaks)