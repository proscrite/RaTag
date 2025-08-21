import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from RaTag.scripts.wfm2read_fast import wfm2read  # type: ignore

def integrate_s2(file, s2_lowcut=0.5e-5, s2_upcut=1.5e-5, bs_pedestal = -0.000772466):
    wf = wfm2read(file)
    t, V = wf[1], -wf[0]
    # if bs_pedestal is not None:
    #     V -= bs_pedestal
    bs_lowcut = 2e-5
    bs_upcut = 4e-5
    baseline_window = (t > bs_lowcut) & (t < bs_upcut) & (V < 0.002) & (V > -0.002)
    baseline = np.mean(V[baseline_window])
    V -= baseline
    s2_window = (t > s2_lowcut) & (t < s2_upcut)
    area_s2 = np.trapz(V[s2_window], t[s2_window]) * 1e6 / 1e-3  # mV/us
    return area_s2

def extract_s2_areas(files, s2_lowcut=0.5e-5, s2_upcut=1.5e-5):
    return np.array([integrate_s2(f, s2_lowcut, s2_upcut) for f in files])

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

def analyse_voltage_scan():
    # Anode voltage scan (Gate fixed at 700 V)
    base_dir = r'E:\Pablos_Mighty_measurements\RUN1\VoltageScans'
    anode_voltages = [2100, 2600, 3100, 3600]
    pattern = r'VoltageScan_1GSsec_EL{}_Gate700\*.wfm'
    files_dict = find_files_by_voltage(base_dir, pattern, anode_voltages)
    s2_lowcut, s2_upcut = 0.5e-5, 1.5e-5
    hist_lowcut, hist_upcut = -10, 20

    s2_areas = {}
    s2_areas_cuts = {}
    for v, files in files_dict.items():
        arr = extract_s2_areas(files, s2_lowcut, s2_upcut)
        s2_areas[v] = arr
        s2_areas_cuts[v] = apply_hist_cuts(arr, hist_lowcut, hist_upcut)
        out_path = os.path.join(base_dir, f'VoltageScan_EL{v}_Gate700', f'histogram_EL{v}_Gate700.png')
        save_histogram(s2_areas_cuts[v], bins=150, out_path=out_path,
                       xlabel='S2 Area (mV*us)', ylabel='Counts',
                       title=f'Anode {v} V, Gate 700 V')
        area_out_path = os.path.join(base_dir, f'VoltageScan_EL{v}_Gate700', f'area_s2_EL{v}_Gate700.npy')
        save_area_array(arr, area_out_path)

    # Plot all histograms together
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green', 'red']
    for v, color in zip(anode_voltages, colors):
        plt.hist(s2_areas_cuts[v], bins=150, alpha=0.5, label=f'Anode {v} V', color=color)
    plt.xlabel('S2 Area (mV*us)')
    plt.ylabel('Counts')
    plt.title('Anode scan, Gate at 700 V')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'anode_scan_all.png'))
    plt.close()

    # Gate voltage scan (Anode fixed at 3100 V)
    gate_voltages = [700, 1200, 1700, 2200, 2700]
    pattern = r'VoltageScan_1GSsec_EL3100_Gate{}\*.wfm'
    files_dict = find_files_by_voltage(base_dir, pattern, gate_voltages)
    hist_lowcut, hist_upcut = -5, 20

    s2_areas = {}
    s2_areas_cuts = {}
    for v, files in files_dict.items():
        arr = extract_s2_areas(files, s2_lowcut, s2_upcut)
        s2_areas[v] = arr
        s2_areas_cuts[v] = apply_hist_cuts(arr, hist_lowcut, hist_upcut)
        out_path = os.path.join(base_dir, f'VoltageScan_EL3100_Gate{v}', f'histogram_EL3100_Gate{v}.png')
        save_histogram(s2_areas_cuts[v], bins=150, out_path=out_path,
                       xlabel='S2 Area (mV*us)', ylabel='Counts',
                       title=f'Gate {v} V, Anode 3100 V')
        area_out_path = os.path.join(base_dir, f'VoltageScan_EL3100_Gate{v}', f'area_s2_EL3100_Gate{v}.npy')
        save_area_array(arr, area_out_path)

    # Plot all histograms together
    plt.figure(figsize=(8, 6))
    colors = ['purple', 'red', 'green', 'orange', 'blue']
    for v, color in zip(gate_voltages[::-1], colors):
        plt.hist(s2_areas_cuts[v], bins=150, alpha=0.5, label=f'Gate {v} V', color=color)
    plt.xlabel('S2 Area (mV*us)')
    plt.ylabel('Counts')
    plt.title('Gate Voltage scan, Anode 3.1 kV')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'gate_scan_all.png'))
    plt.close()

def analyse_field_scan():
    # Drift field scan (vary both voltages together)
    base_dir = r'E:\Pablos_Mighty_measurements\RUN1\FieldScans'
    voltage_pairs = [(2100, 700), (2600, 1200), (3100, 1700), (3600, 2200)]
    pattern = r'FieldScan_1GSsec_EL{}_Gate{}\*.wfm'
    s2_lowcut, s2_upcut = 0.5e-5, 1.5e-5
    hist_lowcut, hist_upcut = -5, 20

    s2_areas = {}
    s2_areas_cuts = {}
    for el_v, gate_v in voltage_pairs:
        path = os.path.join(base_dir, pattern.format(el_v, gate_v))
        files = sorted(glob(path))
        if not files:
            print(f'No files found for EL {el_v} V, Gate {gate_v} V')
            continue
        arr = extract_s2_areas(files, s2_lowcut, s2_upcut)
        s2_areas[(el_v, gate_v)] = arr
        s2_areas_cuts[(el_v, gate_v)] = apply_hist_cuts(arr, hist_lowcut, hist_upcut)
        # out_path = os.path.join(base_dir, f'FieldScan_EL{el_v}_Gate{gate_v}', f'histogram_EL{el_v}_Gate{gate_v}.png')
        # save_histogram(s2_areas_cuts[(el_v, gate_v)], bins=150, out_path=out_path,
        #                xlabel='S2 Area (mV*us)', ylabel='Counts',
        #                title=f'EL {el_v} V, Gate {gate_v} V')
        area_out_path = os.path.join(base_dir, f'FieldScan_EL{el_v}_Gate{gate_v}', f'area_s2_EL{el_v}_Gate{gate_v}.npy')
        save_area_array(arr, area_out_path)

    # Plot all histograms together
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green', 'red']
    for (el_v, gate_v), color in zip(voltage_pairs, colors):
        label = f"$E_{{drift}}$: {round(gate_v/1.4)} V/cm, $V_{{EL}}$: {el_v} V"
        plt.hist(s2_areas_cuts[(el_v, gate_v)], bins=150, alpha=0.5, label=label, color=color)
    plt.xlabel('S2 Area (mV*us)')
    plt.ylabel('Counts')
    plt.title('Drift field scan, EL field: 1.75 kV/cm')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'field_scan_all.png'))
    plt.close()

# Example usage:
# analyse_voltage_scan()
# analyse_field_scan()