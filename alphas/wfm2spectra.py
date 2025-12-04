from glob import glob
import numpy as np
import sys
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# from .wfm2read_fast import wfm2read
from RaTag.core.wfm2read_fast import wfm2read

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

def get_baseline(V, npoints: int = 200) -> float:
    """Estimate the baseline of the waveform."""
    return np.median(V[:npoints])

def find_peak_interpolated(V):
    """Find peak using parabolic interpolation"""
    max_idx = V.argmax()
    if max_idx == 0 or max_idx == len(V)-1:
        return V[max_idx]
    
    # Use 3 points around maximum for parabolic fit
    y0, y1, y2 = V[max_idx-1], V[max_idx], V[max_idx+1]
    
    # Parabolic interpolation
    denom = 2*(2*y1 - y0 - y2)
    if abs(denom) < 1e-10:
        return y1
    
    offset = (y0 - y2) / denom
    peak = y1 - 0.25 * (y0 - y2) * offset
    
    return peak


def alpha_peak(V, threshold_bs=0.3, dither_amplitude=0.02):
    """
    Peak detection with dithering to break ADC quantization.
    
    Parameters:
    -----------
    V : numpy.ndarray
        Raw voltage waveform
    threshold_bs : float
        Voltage threshold for baseline estimation
    smooth_window : int
        Median filter window size
    dither_amplitude : float
        Amplitude of uniform dither noise (default: 0.02V, ~1/4 ADC step)
    
    Returns:
    --------
    float
        Peak energy in MeV
    """
    # Threshold-based baseline
    Vbs = V[V < threshold_bs]
    if len(Vbs) < 10:
        baseline = np.median(V[:200])
    else:
        baseline = np.mean(Vbs)
    
    # Baseline-corrected waveform
    V_corrected = V - baseline

    
    # **KEY**: Add uniform dither noise BEFORE filtering
    # This breaks the quantization and allows interpolation to work
    if dither_amplitude > 0:
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, size=V_corrected.shape)
        V_dithered = V_corrected + dither
    else:
        V_dithered = V_corrected
    # Parabolic interpolation on filtered peak
    peak_value = find_peak_interpolated(V_dithered)
    
    # Convert to MeV
    energy = peak_value / 1.058
    
    return energy

def alpha_peak_vectorized(V_batch, threshold_bs=0.3, smooth_window=21, dither_amplitude=0.02):
    """
    Vectorized peak detection for multiple waveforms.
    
    Parameters:
    -----------
    V_batch : numpy.ndarray
        2D array of waveforms (n_waveforms, n_samples)
    
    Returns:
    --------
    numpy.ndarray
        Array of energies for each waveform
    """
    n_wfms = V_batch.shape[0]
    energies = np.zeros(n_wfms, dtype=np.float32)
    
    for i in range(n_wfms):
        energies[i] = alpha_peak(V_batch[i, :], threshold_bs, smooth_window, dither_amplitude)
    
    return energies

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