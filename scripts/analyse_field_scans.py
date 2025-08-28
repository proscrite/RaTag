import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import re
from RaTag.scripts.wfm2read_fast import wfm2read  # type: ignore

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def extract_voltage_pairs(base_dir, pattern=None):
    """Extract (anode_voltage, gate_voltage) pairs from subdirectory names."""
    voltage_pairs = []
    if pattern is None:
        pattern = 'FieldScan_1GSsec_Anode{}_Gate{}'
    subdirs = sorted(os.listdir(base_dir))
    for subdir in subdirs:
        try:
            gate_voltage = int(re.search(r'Gate(\d+)', subdir).group(1))
            anode_voltage = int(re.search(r'Anode(\d+)', subdir).group(1))
            # print(f"Anode voltage : {anode_voltage}, Gate voltage : {gate_voltage}")
            voltage_pairs.append((anode_voltage, gate_voltage))
        except AttributeError:
            continue
    return np.array(voltage_pairs)

def compute_average_waveform(files, bs_t_window=(-1.5e-5, -1.0e-5), threshold_bs=0.05):
    w = wfm2read(files[0], verbose=False)
    t, V = w[1], -w[0]

    V_avg = np.zeros_like(V)

    for f in files:
        t, V = load_waveform(f)

        # 1) Reject on "too large signal in baseline window"
        baseline_mask = (t > bs_t_window[0]) & (t < bs_t_window[1])
        V_w = V[baseline_mask]
        if (V_w > threshold_bs).sum() > 5:
            continue
        
        if V.shape != V_avg.shape:
            print(f'Warning: waveform in {f} has different shape. Skipping.')
            continue
        V_avg += V
    V_avg /= len(files)
    return V_avg, t

def compute_drift_time(t, V, height_S1=0.001):
    
    indS1 = find_peaks(V[t < 0], height=height_S1, distance=200)[0]

    if len(indS1) > 1:
        indS1 = indS1[np.argmax(V[indS1])]
    else:
        indS1 = indS1[0]

    indS2 = find_peaks(V[t > 0], height=V[indS1], distance=500)[0]

    if len(indS2) > 1:
        indS2 = indS2[np.argmin(t[indS2])]
    else:
        indS2 = indS2[0]
    indS2 += len(V[t < 0])
    return t[indS2] - t[indS1], indS1, indS2


def get_fields(voltage_pairs, EL_gap=0.8, drift_gap=1.4):
    """Calculate electric fields from voltage pairs and gaps."""
    E_drift = np.array([round(vg/drift_gap, 3) for _, vg in voltage_pairs])
    E_el = np.array([round( (va-vg)/EL_gap , 3) for va, vg in voltage_pairs])
    return E_drift, E_el

def check_fields(voltage_pairs, E_el, E_drift):
    """Check if E_el is constant and if E_el/E_drift > 2."""
    if np.diff(E_el).all() != 0:
        raise ValueError ("Warning: EL field is not constant across all voltage pairs")
    if (E_el / E_drift <= 2).any():
        print(f"Warning: E_EL/E_drift < 2 for voltage pairs = {voltage_pairs[np.where(E_el/E_drift / 2 <= 2)[0]]} V")
        return voltage_pairs[np.where(E_el/E_drift / 2 <= 2)[0]]

def load_waveform(file):
    wf = wfm2read(file)
    t, V = wf[1], -wf[0]
    return t, V
    
def subtract_baseline(t, V, t_window=(-1.5e-5, -1.0e-5), v_window=(-0.002, 0.002)):
    baseline_window = (t > t_window[0]) & (t < t_window[1])
    if v_window is not None:
        baseline_window &= (V > v_window[0]) & (V < v_window[1])
    baseline = np.min(V[baseline_window])
    V -= baseline
    return V

def integrate_s2(t, V, s2_lowcut=0.5e-5, s2_upcut=1.5e-5):
    s2_window = (t > s2_lowcut) & (t < s2_upcut)
    area_s2 = np.trapz(V[s2_window], t[s2_window]) * 1e6 / 1e-3  # mV/us

    return area_s2

def extract_s2_areas(files, threshold_bs = 0.05, integ_window=(0.5e-5, 1.5e-5), bs_t_window=(-1.5e-5, -1.0e-5), bs_v_window=None):
    areas = []
    for f in files:
        t, V = load_waveform(f)

        # 1) Reject on “too large signal in baseline window”
        baseline_mask = (t > bs_t_window[0]) & (t < bs_t_window[1])
        V_w = V[baseline_mask]
        if (V_w > threshold_bs).sum() > 5:
            continue

        # 2) Baseline‐subtract
        V_corr = subtract_baseline(t, V, bs_t_window, bs_v_window)

        # 3) Integrate S2
        area = integrate_s2(t, V_corr, integ_window[0], integ_window[1])
        areas.append(area)

    return np.array(areas)

def apply_hist_cuts(arr, lowcut, upcut):
    return arr[(arr > lowcut) & (arr < upcut)]

def save_histogram(data, bins, out_path, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def find_files_by_voltage(base_dir, pattern, voltages):
    files_dict = {}
    for v in voltages:
        path = os.path.join(base_dir, pattern.format(v))
        files = sorted(glob(path))
        files_dict[v] = files
    return files_dict

def save_area_array(area_array, out_path):
    np.save(out_path, area_array)

def drop_voltage_pairs(voltage_pairs, to_drop):
        """Remove specified (anode_voltage, gate_voltage) pairs from voltage_pairs.
        Args:
            voltage_pairs (np.ndarray or list): Array/list of (anode_voltage, gate_voltage) pairs.
            to_drop (list): List of pairs to remove.
        Returns:
            np.ndarray: Filtered voltage_pairs.
        """
        filtered = [pair for pair in voltage_pairs if tuple(pair) not in to_drop]
        return np.array(filtered)

class GaussianFitter:
    """Class to fit Gaussian to histogram data.
    1) Fit Gaussian to histogram data.
    2) Plot histogram and Gaussian fit.
    3) Extract fit parameters and confidence intervals.
    """
    def __init__(self, hist_data, bins, label=None, color=None, mask_range=(-4, 5)):
        """Initialize with histogram data and parameters.
        Args:
            hist_data (np.array): Histogram data.
            bins (np.array): Histogram bin edges.
            label (str): Label for plot.
            color (str): Color for plot.
            mask_range (tuple): Range to mask for fitting.
        """
        self.hist_data = hist_data
        self.bins = bins
        self.label = label
        self.color = color
        self.mask_range = mask_range

        self.bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.mask = (self.bin_centers > self.mask_range[0]) & (self.bin_centers < self.mask_range[1])
    

    def fit(self):
        """Fit Gaussian to histogram data."""
        popt, pcov = curve_fit(gaussian, self.bin_centers[self.mask], self.hist_data[self.mask], p0=[np.max(self.hist_data), np.mean(self.bin_centers[self.mask]), np.std(self.bin_centers[self.mask])])
        self.popt = popt
        self.pcov = pcov
        return popt, pcov

    def plot(self, ax=None, flag_label=True):
        """Plot histogram and Gaussian fit."""
        if ax is None:
            ax = plt.gca()
        bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        if flag_label and self.label is not None:
            la = self.label
        else:
            la = None
        ax.bar(self.bins[:-1], self.hist_data, width=np.diff(self.bins), alpha=0.5, label=la, color=self.color)
        if not hasattr(self, 'popt'):
            self.fit()
        ax.plot(bin_centers, gaussian(bin_centers, *self.popt), 'r--')
        ax.legend(fontsize=10)
        return ax
    
    def get_fit_params(self):
        """Get fit parameters and 95% confidence intervals.
        Returns:
            tuple: (mean, ci95)
        """
        if not hasattr(self, 'popt') or not hasattr(self, 'pcov'):
            self.fit()
        mean = self.popt[1]
        rms = self.popt[2]
        std_error = np.sqrt(self.pcov[1, 1])
        ci95 = 1.96 * std_error
        return mean, ci95
    
class S2Window:
    def __init__(self, path_subdir, batch_size=20, pattern='/*.wfm'):
        self.path_subdir = path_subdir
        self.batch_size = batch_size
        self.pattern = pattern
        self.error_count = 0
        
    def batch_files(self):
        n_all_files = len(glob(self.path_subdir+self.pattern))
        files_batch = [sorted(glob(self.path_subdir+self.pattern))[self.batch_size*i:self.batch_size*(i+1)] for i in range(n_all_files//self.batch_size)]
        return files_batch
    
    def compute_average_batch(self, fi):
        v_avg, t_avg = compute_average_waveform(fi)
        v_avg = subtract_baseline(t_avg, v_avg, (-1.5e-5, -1.0e-5))
        t_avg = t_avg
        return v_avg, t_avg
    
    def find_s1(self, t, V, height_S1=0.001, min_distance=200):
        indS1 = find_peaks(V[t < 0], height=height_S1, distance=min_distance)[0]
        if len(indS1) > 1:
            indS1 = indS1[np.argmax(V[indS1])]
        else:
            indS1 = indS1[0]
        return indS1

    def find_s2_start(self, t, V, indS1, height_frac=0.8, min_distance=50):
        indS2 = find_peaks(V[t > 0], height=V[indS1]*height_frac, distance=min_distance)[0]
        if len(indS2) > 1:
            indS2 = indS2[np.argmin(t[indS2])]
        else:
            indS2 = indS2[0]
        indS2 += len(V[t < 0])
        return indS2

    def find_s2_width(self, t, V, indS1, height_frac=0.8, min_distance=50):
        indS2 = self.find_s2_start(t, V, indS1, height_frac, min_distance)
        indS2_end = find_peaks(V[t > 0], height=V[indS1]*height_frac, distance=min_distance)[0] + len(V[t < 0])
        indS2_end = indS2_end[np.where(np.diff(indS2_end) < 4000)[0]]
        if len(indS2_end) == 0:
            raise ValueError("No valid S2 end found within max distance")
        indS2_end = indS2_end[-1]
        if indS2_end >= len(t):
            raise ValueError("S2 window exceeds waveform length")
        return indS2, indS2_end
    
    def compute_average_s2_window(self):
        
        files_batch = self.batch_files()
        s1_times = []
        s2_starts = []
        s2_ends = []
        
        for fi in files_batch:
            v_avg, t_avg = self.compute_average_batch(fi)
            try:
                indS1 = self.find_s1(t_avg, v_avg)
                indS2, indS2_end = self.find_s2_width(t_avg, v_avg, indS1)
            except (IndexError, ValueError):
                self.error_count += 1
                continue
            s1_times.append(t_avg[indS1])
            s2_starts.append(t_avg[indS2])
            s2_ends.append(t_avg[indS2_end])
        self.s1_avg = np.mean(s1_times)
        self.s2_start_avg = np.mean(s2_starts)
        self.s2_end_avg = np.mean(s2_ends)
        self.duration = self.s2_end_avg - self.s2_start_avg
        return self.s1_avg, self.s2_start_avg, self.s2_end_avg, self.duration
    
    def compute_average_s2_start(self):
        files_batch = self.batch_files()
        s1_times = []
        s2_starts = []
        
        for fi in files_batch:
            v_avg, t_avg = self.compute_average_batch(fi)
            try:
                indS1 = self.find_s1(t_avg, v_avg)
                indS2 = self.find_s2_start(t_avg, v_avg, indS1)
            except (IndexError, ValueError):
                self.error_count += 1
                continue
            s1_times.append(t_avg[indS1])
            s2_starts.append(t_avg[indS2])
        self.s1_avg = np.mean(s1_times)
        self.s2_start_avg = np.mean(s2_starts)
        return self.s1_avg, self.s2_start_avg
    
    def plot_avg_values(self):
        allFiles = sorted(glob(self.path_subdir+self.pattern))
        v_avg, t_avg = compute_average_waveform(allFiles[:self.batch_size])
        v_avg = subtract_baseline(t_avg, v_avg, (-1.5e-5, -1.0e-5))
        t_avg = t_avg * 1e6
        if not hasattr(self, 's1_avg') or not hasattr(self, 's2_start_avg'):
            raise AttributeError("Average S1 and S2 start times not computed. Run compute_average_s2_start() or compute_average_s2_window() first.")
        plt.plot(t_avg, v_avg)
        plt.axvline(self.s1_avg*1e6, color='r', linestyle='--', label='Avg S1 time')
        plt.axvline(self.s2_start_avg*1e6, color='g', linestyle='--', label='Avg S2 start')
        if hasattr(self, 'duration'):
            plt.axvline(self.s2_start_avg*1e6 + self.duration*1e6, color='m', linestyle='--', label='Avg S2 end time')
        # plt.axvline(self.s1_avg*1e6 + (self.s2_start_avg - self.s1_avg)*1e6, color='b', linestyle='--', label='Avg drift time (S2 start)')
        plt.gca().set(xlabel='Time (us)', ylabel='Voltage (V)', title='Average waveform with average S1 and S2 times')
        plt.legend()

class S2AreaExtractor:
    """Class to extract S2 areas from waveform files.
    1) Reject on "too large signal in baseline window"
    2) Baseline-subtract
    3) Integrate S2
    """
    def __init__(self, files, threshold_bs=0.05, s2_start=1e-6, s2_duration=2e-5, bs_t_window=(-1.5e-5, -1.0e-5), bs_v_window=None):
        """Initialize with waveform files and parameters.
        Args:
            files (list): List of waveform file paths.
            s2_start (float): Time start of S2 window.
            s2_duration (float): Duration of S2 window. (Fixed for a Field Scan)
            threshold_bs (float): Threshold for baseline rejection.
            bs_t_window (tuple): Time window for baseline subtraction.
            bs_v_window (tuple): Voltage window for baseline subtraction.
        """
        self.files = files
        self.s2_start = s2_start
        self.s2_duration = s2_duration
        self.threshold_bs = threshold_bs
        self.bs_t_window = bs_t_window
        self.bs_v_window = bs_v_window

    def extract_areas(self):
        """Extract S2 areas from waveform files.
        Returns:
            np.array: Array of extracted S2 areas.
        """
        areas = []
        for f in self.files:
            t, V = load_waveform(f)

            # 1) Reject on "too large signal in baseline window"
            baseline_mask = (t > self.bs_t_window[0]) & (t < self.bs_t_window[1])
            V_w = V[baseline_mask]
            if (V_w > self.threshold_bs).sum() > 5:
                continue

            # 2) Baseline-subtract
            V_corr = subtract_baseline(t, V, self.bs_t_window, self.bs_v_window)

            # 3) Integrate S2
            area = integrate_s2(t, V_corr, self.s2_start, self.s2_start + self.s2_duration)
            areas.append(area)

        return np.array(areas)

class FieldScanAnalyzer:
    """Class to analyze field scan data.
    1) Extract voltage pairs from subdirectory names.
    2) Calculate electric fields from voltage pairs and gaps.
    3) Check if E_el is constant and if E_el/E_drift > 2.
    4) Extract S2 areas from waveform files.
    5) Apply histogram cuts.
    6) Plot histograms.
    """
    def __init__(self, base_dir, pattern=None, to_drop = None, n_files=None,
                areaExtractor=S2AreaExtractor, fitter=GaussianFitter, WindowS2=S2Window,
                hist_cuts=(-5, 120), nbins=120, EL_gap=0.8, drift_gap=1.4):
        """Initialize with base directory and parameters.
        Args:
            base_dir (str): Base directory containing field scan subdirectories.
            pattern (str): Pattern for subdirectory names.
            areaExtractor (class): Class to extract S2 areas from waveform files.
            hist_cuts (tuple): Histogram cuts for S2 areas.
            nbins (int): Number of bins for histograms.
            EL_gap (float): Gap distance for EL field calculation.
            drift_gap (float): Gap distance for drift field calculation.
        """
        self.base_dir = base_dir
        self.pattern = pattern
        self.to_drop = to_drop 
        self.n_files = n_files

        self.hist_cuts = hist_cuts
        self.nbins = nbins
        
        self.areaExtractor = areaExtractor
        self.fitter = fitter
        self.WindowS2 = WindowS2

        self.EL_gap = EL_gap
        self.drift_gap = drift_gap
        self.setup()


    def setup(self):
        """Setup by extracting voltage pairs and calculating fields."""
        self.voltage_pairs = extract_voltage_pairs(self.base_dir, self.pattern)
        if self.to_drop is not None:
            self.voltage_pairs = drop_voltage_pairs(self.voltage_pairs, to_drop=self.to_drop)
            
        self.v_gate = self.voltage_pairs[:, 1]
        self.v_anode = self.voltage_pairs[:, 0]
        self.E_drift, self.E_el = get_fields(self.voltage_pairs, self.EL_gap, self.drift_gap)
        self.E_dr_dict, self.E_el_dict = dict(zip(self.v_gate, self.E_drift)), dict(zip(self.v_gate, self.E_el))
        self.voltage_over_lim = check_fields(self.voltage_pairs, self.E_el, self.E_drift)

    def find_s2(self, path):
        """Find average S2 start time in a given subdirectory."""
        s2_window_finder = self.WindowS2(path)
        batches = s2_window_finder.batch_files()
        print(f'Found {len(batches)} batches of files in {path}')
        s1_avg, s2_start_avg = s2_window_finder.compute_average_s2_start()
        return s1_avg, s2_start_avg

    def analyze(self):
        """Analyze field scan data and plot histograms."""
        hist_lowcut, hist_upcut = self.hist_cuts

        s2_areas = {}
        s2_areas_cuts = {}
        glob_pattern = self.pattern + '/*.wfm' if self.pattern else 'FieldScan_1GSsec_EL{}_Gate{}' + '/*.wfm'
        for anode_v, gate_v in self.voltage_pairs:
            path = os.path.join(self.base_dir, glob_pattern.format(anode_v, gate_v))
            files = sorted(glob(path))
            if self.n_files is not None:
                files = files[:self.n_files]
            print(f'Integrating {len(files)} files for EL {anode_v} V, Gate {gate_v} V')

            _, s2_start = self.find_s2(os.path.join(self.base_dir, self.pattern.format(anode_v, gate_v)))
            print(f'Found S2 start at {s2_start*1e6:.2f} us for EL {anode_v} V, Gate {gate_v} V')
            
            extractor = self.areaExtractor(files, s2_start=s2_start)
            arr_areas = extractor.extract_areas()
            s2_areas[gate_v] = arr_areas
            s2_areas_cuts[gate_v] = apply_hist_cuts(arr_areas, hist_lowcut, hist_upcut)
            area_out_path = os.path.join(self.base_dir, self.pattern.format(anode_v, gate_v), f'area_s2_EL{anode_v}_Gate{gate_v}.npy')
            save_area_array(arr_areas, area_out_path)
        self.s2_areas = s2_areas
        self.s2_areas_cuts = s2_areas_cuts
        return s2_areas_cuts, s2_areas
    
    def plot_histograms(self, ax = None, label_anode = False, label_gate = False):
        """Plot histograms of S2 areas for all voltage pairs."""
        if ax is None:
            ax = plt.gca()
        for (anode_v, gate_v) in self.voltage_pairs:
            label = f"$E_{{drift}}$: {round(gate_v/self.drift_gap)} V/cm"
            if label_anode: label += " $V_{{anode}}$: {anode_v} V"
            if label_gate: label += " $V_{{gate}}$: {gate_v} V"
            ax.hist(self.s2_areas_cuts[gate_v], bins=self.nbins, alpha=0.5, label=label)
        ax.set(xlabel = 'S2 Area (mV*us)', ylabel = 'Counts', title = f'Drift field scan, EL field: {self.E_el[0]} kV/cm')

        plt.legend(fontsize=10)
        return ax
    
    def fit_histograms_2peaks(self, nbins=100, sep_A = 70, flag_plot:bool = True, ax=None):
        """Fit Gaussian to histograms of S2 areas for all voltage pairs."""
         
        mask_range_p1=(self.hist_cuts[0], sep_A)
        mask_range_p2=(sep_A, self.hist_cuts[1])
        if ax is None:
            ax = plt.gca()
        fit_results_p1 = {}
        fit_results_p2 = {}
        for gate_v, a1 in self.s2_areas_cuts.items():
            n, bins = np.histogram(a1, bins=nbins)
            fitter_p1 = self.fitter(n, bins, label=str(gate_v), mask_range=mask_range_p1)
            fitter_p2 = self.fitter(n, bins, label=str(gate_v), mask_range=mask_range_p2)
            try:
                fitter_p1.fit()
                fitter_p2.fit()
            except (ValueError, RuntimeError):
                print(f"Could not fit Gaussian for gate voltage {gate_v} V")
                continue
            if flag_plot:
                fitter_p1.plot(ax=ax)
                fitter_p2.plot(ax=ax, flag_label=False)
            mean_p1, ci95_p1 = fitter_p1.get_fit_params()
            mean_p2, ci95_p2 = fitter_p2.get_fit_params()
            fit_results_p1[gate_v] = (mean_p1, ci95_p1)
            fit_results_p2[gate_v] = (mean_p2, ci95_p2)
            print(f'{gate_v} V: amplitude={fitter_p1.popt[0]:.2f}, mean={fitter_p1.popt[1]:.2f}, sigma={fitter_p1.popt[2]:.2f})')
            print(f'{gate_v} V: amplitude={fitter_p2.popt[0]:.2f}, mean={fitter_p2.popt[1]:.2f}, sigma={fitter_p2.popt[2]:.2f})')
        self.fit_results_p1 = fit_results_p1
        self.fit_results_p2 = fit_results_p2
        return fit_results_p1, fit_results_p2
    
    def plot_s2_vs_field(self):

        """Plot fit results of Gaussian fits to histograms of S2 areas."""
        fig, ax = plt.subplots(2, figsize=(8, 10), sharex=True)
            
        mean_p1, ci95_p1 = zip(*self.fit_results_p1.values())
        mean_p2, ci95_p2 = zip(*self.fit_results_p2.values())
        E_d = [self.E_dr_dict[v] for v in self.fit_results_p1.keys()]
        ax[0].errorbar(E_d, mean_p1, yerr=ci95_p1, fmt='o', color='blue', capsize=10, capthick=2, label='High S2 area peak')
        ax[1].errorbar(E_d, mean_p2, yerr=ci95_p2, fmt='o', color='orange', capsize=10, capthick=2, label='Low S2 area peak')

        for a in ax:
            a.legend()
            a.set(xlabel='$E_{drift}$ (V/cm)', ylabel='Mean S2 Area (mV$\cdot$us)', 
                title='Mean S2 Area vs $E_{drift}$ (95% CI)')
        return ax

