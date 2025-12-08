from glob import glob
import numpy as np
import sys
from scipy.signal import find_peaks, savgol_filter
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


def alpha_peak(V, threshold_bs=0.3, dither_amplitude=0.02, savgol_window=501, savgol_order=3):
    """
    Optimized alpha peak energy extraction using Savitzky-Golay filter.
    
    This method produces smooth energy spectra without artificial clustering
    by applying polynomial smoothing to break ADC quantization artifacts.
    
    Parameters:
    -----------
    V : numpy.ndarray
        Raw voltage waveform (typically 35,000 samples at 5 GS/s)
    threshold_bs : float
        Voltage threshold for baseline point selection (default: 0.3V)
    dither_amplitude : float
        Uniform dither amplitude in volts (default: 0.02V)
    savgol_window : int
        Savitzky-Golay filter window size (default: 501 samples ≈ 100 ns)
        Must be odd. Larger = more smoothing, closer to MCA analog shaping
    savgol_order : int
        Polynomial order for Savitzky-Golay (default: 3)
        
    Returns:
    --------
    float
        Peak energy in MeV (calibrated with factor 1.058)
        
    Notes:
    ------
    - Uses threshold-based baseline for robustness
    - Applies dithering to break ADC quantization
    - Savitzky-Golay filter approximates analog Gaussian shaping (2-6 μs)
    - No iterative fitting = robust and fast
    - Window size 501 ≈ 100 ns approaches lower end of MCA shaping times
    """
    # 1. Threshold-based baseline
    Vbs = V[V < threshold_bs]
    if len(Vbs) >= 10:
        baseline = np.mean(Vbs)
    else:
        baseline = np.median(V[:200])
    
    V_corrected = V - baseline
    
    # 2. Add dither to break ADC quantization
    if dither_amplitude > 0:
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, size=V_corrected.shape)
        V_dithered = V_corrected + dither
    else:
        V_dithered = V_corrected
    
    # 3. Apply Savitzky-Golay filter (polynomial smoothing)
    V_smooth = savgol_filter(V_dithered, savgol_window, savgol_order)
    
    # 4. Simple maximum on smoothed waveform
    peak_value = V_smooth.max()
    
    # 5. Apply calibration factor
    energy = peak_value / 1.058
    
    return energy

def alpha_peak_vectorized(V_batch, threshold_bs=0.3, dither_amplitude=0.02, savgol_window=501, savgol_order=3):
    """
    Truly vectorized peak detection for multiple waveforms using Savitzky-Golay filter.
    
    This function processes a batch of waveforms efficiently by vectorizing baseline
    correction, dithering, and applying Savitzky-Golay filter along the time axis.
    
    Parameters:
    -----------
    V_batch : numpy.ndarray
        2D array of waveforms (n_waveforms, n_samples)
    threshold_bs : float
        Voltage threshold for baseline point selection (default: 0.3V)
    dither_amplitude : float
        Uniform dither amplitude in volts (default: 0.02V)
    savgol_window : int
        Savitzky-Golay filter window size (default: 501)
    savgol_order : int
        Polynomial order for Savitzky-Golay (default: 3)
    
    Returns:
    --------
    numpy.ndarray
        Array of energies (in MeV) for each waveform
        
    Notes:
    ------
    - Uses vectorized operations for significant speedup on large batches
    - savgol_filter is applied along axis=1 (time axis) for each waveform
    - Baseline is computed per-waveform but vectorized across batch
    """
    n_wfms, n_samples = V_batch.shape
    
    # 1. Vectorized baseline correction
    baselines = np.zeros(n_wfms, dtype=np.float32)
    for i in range(n_wfms):
        Vbs = V_batch[i, V_batch[i] < threshold_bs]
        if len(Vbs) >= 10:
            baselines[i] = np.mean(Vbs)
        else:
            baselines[i] = np.median(V_batch[i, :200])
    
    # Subtract baseline (broadcasting)
    V_corrected = V_batch - baselines[:, np.newaxis]
    
    # 2. Vectorized dithering
    if dither_amplitude > 0:
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, size=V_batch.shape)
        V_dithered = V_corrected + dither
    else:
        V_dithered = V_corrected
    
    # 3. Apply Savitzky-Golay filter along time axis (axis=1)
    # This is the key vectorization - scipy applies filter to each row
    V_smooth = savgol_filter(V_dithered, savgol_window, savgol_order, axis=1)
    
    # 4. Find maximum for each waveform (vectorized)
    peak_values = V_smooth.max(axis=1)
    
    # 5. Apply calibration factor
    energies = peak_values / 1.058
    
    return energies

def analyze_file_source(file, savgol_window=501):
    """
    Analyze a single waveform file and extract peak energies.
    
    Parameters:
    -----------
    file : str
        Path to waveform file
    savgol_window : int
        Savitzky-Golay window size (default: 501)
    
    Returns:
    --------
    numpy.ndarray
        Array of peak energies from all frames in the file
    """
    wf = wfm2read(file)
    V, t = wf[0], wf[1]
    peaks = []
    if len(V.shape) > 1:
        for v in V:
            peaks.append(alpha_peak(v, savgol_window=savgol_window))
    else:   
        peaks.append(alpha_peak(V, savgol_window=savgol_window))
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