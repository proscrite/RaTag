import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from RaTag.scripts.wfm2read_fast import wfm2read  # type: ignore 

# --- Parameters ---
dt0 = 0.2  # ns
dt1 = 6    # ns
dt2 = 11   # ns
ymin = 0.4

area_vec = []
plot_switch = 0

dir_base = "/Users/pabloherrero/sabat/RaTagging/scope_data/waveforms/RUN5_EL2375Vcm_5GSsec/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode1950V_Gate50V/"
fname_base = dir_base + "FieldScan_5GSsec_Anode2000V_Gate100V/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode2100V_Gate200V/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode2500V_Gate600V/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode3000V_Gate1100V/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode3500V_Gate1600V/"
# fname_base = dir_base + "FieldScan_5GSsec_Anode4000V_Gate2100V/"

def moving_average(x, window_size=9):
    """Simple moving average filter (like MATLAB smooth with default)."""
    return np.convolve(x, np.ones(window_size)/window_size, mode="same")
print("Looking into directory:")
print(fname_base)

for Nfile, file in enumerate(sorted(glob(fname_base + "*.wfm"))[:5]):
    print(Nfile, ': ', file)

input("Press Enter to continue...")
# --- Loop over files ---
for Nfile, file in enumerate(sorted(glob(fname_base + "*.wfm"))):

    # fname = f"{fname_base}{Nfile}Wfm_Ch1.wfm"

    # Replace this with your actual waveform reader
    # Expected output: sig_PMT (array), t (array, in seconds), info_sig...
    wf = wfm2read(file)  
    sig_PMT, t = wf[0], wf[1]

    t = t * 1e9        # convert to ns
    sig_PMT = sig_PMT * 1e3  # convert to mV

    print(f"Nfile = {Nfile}")

    iped = np.where(t < t[0] + 6e3)[0]
    pedestal = np.mean(sig_PMT[iped])
    y = -(sig_PMT - pedestal)

    # --- Define S2 window ---
        # Wide S2 window
    # --------------
    # t1 = 1.5e4; # gate 50 V, anode 1950 V 
    # t2 = 3.5e4; # gate 50 V, anode 1950 V 
    t1 = 1.1e4; # gate 100V, anode 2000 V
    t2 = 1.8e4; # gate 100V, anode 2000 V
    # t1 = 1.0e4; # gate 200 V, anode 2100 V (filename says "Gate400")
    # t2 = 1.6e4; # gate 200 V, anode 2100 V (filename says "Gate400")
    # t1 = 0.8e4; # gate 600V, anode 2500 V
    # t2 = 1.4e4; # gate 600V, anode 2500 V
    # t1 = 0.75e4; # gate 1100V, anode 3000 V
    # t2 = 1.3e4; # gate 1100V, anode 3000 V
    # t1 = 0.65e4; # gate 1600V, anode 3500 V
    # t2 = 1.1e4; # gate 1600V, anode 3500 V
    # t1 = 0.55e4; # gate 2100V, anode 4000 V
    # t2 = 1.05e4; # gate 2100V, anode 4000 V


    i = np.where((t >= t1) & (t <= t2))[0]
    y_S2_window = moving_average(y[i], 9)
    t_S2_window = t[i]

    # area calculation
    imin = np.where(y_S2_window > 2 * ymin)[0]
    area_i = np.sum(y_S2_window[imin]) * dt0
    area_vec.append(area_i)

    # --- Optional plotting ---
    if plot_switch == 1:
        plt.figure(1)
        plt.clf()
        plt.plot(t, y, label="Signal")
        plt.plot(t_S2_window[imin], y_S2_window[imin], ".r", label="Selected points")
        plt.axvline(t1, color="k")
        plt.axvline(t2, color="k")
        plt.title(f"Nfile = {Nfile}, area = {int(area_i)}")
        plt.legend()
        plt.show(block=False)
        input("Press Enter to continue...")

# --- Save results ---
fname_out = f"{fname_base}area_vec.txt"
np.savetxt(fname_out, area_vec)