from glob import glob
import numpy as np
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# from .wfm2read_fast import wfm2read
from RaTag.scripts.wfm2read_fast import wfm2read

def get_peak_volts(V):
    """
    Function to read all waveform files in a directory and extract peaks.
    :param file_path: Path to the directory containing waveform files.
    :return: List of peaks found in the waveform files.
    """
    peaks, props = find_peaks(V, height=1, distance=1000, prominence=1)
    
    if len(peaks) >1:
        # Keep only the highest peak
        highest_peak_index = np.argmax(props['peak_heights'])
        peaks = [peaks[highest_peak_index]]
    volts = V[peaks]
    return volts

def histogram_voltages(volts):
    """Function to plot histogram of peaks."""
    plt.hist(volts, bins=100, edgecolor='black')
    plt.xlabel('Peak Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peaks from Waveform Files')
    plt.show()

def analyse_analog_file(file):
    wf = wfm2read(file, verbose=False)
    V, t = wf[0], wf[1]  # in mV and ns
    peaks = []
    if len(V.shape) > 1:
        for v in V:
            peaks.append(alpha_peaks(v))
    else:
        peaks.append(alpha_peaks(V))
    return np.array(peaks)

def compute_analog_path(path):
    files = sorted(glob(path + '/*.wfm') )
    all_peaks = []
    for file in files:
        peaks = analyse_analog_file(file)
        all_peaks.append(peaks)
    return np.concatenate(all_peaks)

def get_baseline(V):
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

def process_alpha_waveforms(path_sca: str) -> np.ndarray:
    """
    Process a list of waveform files and extract peak voltages.
    
    Parameters:
    -----------
    path_sca : str
        Path to the directory containing waveform files

    Returns:
    --------
    numpy.ndarray
        Array of peak voltages (baseline subtracted)
    """
    peak_volts = []
    
    file_list = sorted(glob(path_sca + '/*.wfm'))
    for f in file_list:
        wf = wfm2read(f, verbose=False)
        V, t = wf[0], wf[1]
        
        if len(V.shape) > 1:
            # Multiple waveforms in the file
            for v in V:
                bs = get_baseline(v)
                peak = v.max() - bs
                peak_volts.append(peak)
        else:
            # Single waveform in the file
            peak_volts.append(alpha_peak(f))
    
    return np.array(peak_volts)

if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print("Usage: python wfm2spectra.py <path_to_wfm_files>")
        sys.exit(1)

    path = sys.argv[1]
    peaks = get_peak_volts(path)
    print(f"Found {len(peaks)} peaks in {path}")
    print(f"Peaks: {peaks}")
    histogram_voltages(peaks)