from glob import glob
import numpy as np
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# from .wfm2read_fast import wfm2read
from RaTag.scripts.wfm2read_fast import wfm2read

def get_peak_volts(y):
    """
    Function to read all waveform files in a directory and extract peaks.
    :param file_path: Path to the directory containing waveform files.
    :return: List of peaks found in the waveform files.
    """
    peaks, props = find_peaks(y, height=1, distance=1000, prominence=1)
    volts = y[peaks]
    return volts

def histogram_voltages(volts):
    """Function to plot histogram of peaks."""
    plt.hist(volts, bins=100, edgecolor='black')
    plt.xlabel('Peak Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peaks from Waveform Files')
    plt.show()

def get_baseline(V, npoints = 200):
    """Estimate the baseline of the waveform."""
    return np.mean(V[:npoints])

def alpha_peak(V, npoints_bs = 200):
    bs = get_baseline(V, npoints_bs)
    V = V - bs
    return V.max() / 1.058

def analyze_file_source(file):
    wf = wfm2read(file)
    V, t = wf[0], wf[1]
    peaks = []
    if len(V.shape) > 1:
        for v in V:
            peaks.append(alpha_peak(v))
    else:   
        peaks.append(alpha_peak(V))
    peaks = np.array(peaks)
    return peaks

if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print("Usage: python wfm2spectra.py <path_to_wfm_files>")
        sys.exit(1)

    path = sys.argv[1]
    peaks = get_peak_volts(path)
    print(f"Found {len(peaks)} peaks in {path}")
    print(f"Peaks: {peaks}")
    histogram_voltages(peaks)