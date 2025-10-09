from scipy.signal import find_peaks, peak_widths
# -----
# --- Signal classification functions for single waveforms

def detect_s2_candidates(V, t, baseline_region=None, peak_thresh_factor=5):
    # V: 1D array, t in microseconds per sample
    if baseline_region is None:
        # pick a baseline region early in V (or compute global)
        baseline = V[:int(0.2*len(V))]
    else:
        baseline = V[baseline_region[0]:baseline_region[1]]
    b_mean = baseline.mean()
    b_std  = baseline.std()
    thresh = b_mean + peak_thresh_factor * b_std

    peaks, props = find_peaks(V, height=thresh, distance=3)  # distance in samples
    heights = props['peak_heights']
    widths_res = peak_widths(V, peaks, rel_height=0.5)
    # widths_res[0] is width in samples
    s2_list = []
    for i, p in enumerate(peaks):
        start = int(max(0, p - widths_res[0][i]*1.2))
        end   = int(min(len(V)-1, p + widths_res[0][i]*1.2))
        q = V[start:end].sum() * t   # integrated charge (V*us)
        width_us = widths_res[0][i] * t
        s2_list.append({'peak_index':p, 'amp':heights[i], 'start':start, 'end':end, 'width_us':width_us, 'charge':q})
    return s2_list, b_mean, b_std

def classify_event_v1(V, t, s2_list, s2_start_idx, s1_idx, params=None, flag_plot=False):
    # params: dict of thresholds
    if params is None:
        params = {}
    
    after_s1 = [s for s in s2_list if s['peak_index'] > s1_idx]
    if not after_s1: return False, "no S2 after S1"
    # identify main S2: the first peak after s2_start_idx
    after_drift = [s for s in after_s1 if s2_start_idx is None or s['peak_index'] > s2_start_idx]
    if not after_drift: return False, "no S2 after S1"
    main = max(after_drift, key=lambda s: s['charge'].sum()) # earliest main
    # find early S2: any S2 between S1 and main
    early_candidates = [s for s in after_s1 if s['peak_index'] < s2_start_idx]
    # early_candidates = [s for s in s2_list if s['peak_index'] < main['peak_index']]  # Alternatively, consider all before main

    if not early_candidates:
        return False, "no early S2"
    # early = max(early_candidates, key=lambda s: s['charge'].sum())  # largest early # No, this only works if the peaks are well joined
    early = max(early_candidates, key=lambda s: s['end'])  # latest early
    # compute inter-window metrics
    inter_start = early['end']
    inter_end   = s2_start_idx
    # if inter_end <= inter_start:
    #     return False, "overlapping pulses"   # Irrelevant, this can never happen with the new definition of early
    print(f"Early S2 at {t[early['peak_index']]:.2f} us, main S2 at {t[main['peak_index']]:.2f} us")
    inter = V[inter_start:inter_end]
    if not inter.size:
        return False, "no inter-window samples"
    print(f"Inter-window: {t[inter_start]:.2f} to {t[inter_end]:.2f} us")
    rms = inter.std()
    maxabs = np.max(inter)
    first_peak_drift = after_s1[0]
    width_xray = t[inter_start] - t[first_peak_drift['start']]

    # thresholds (tune on data)
    RMS_thresh = params.get('RMS_thresh', 3 * np.std(V[:100])) 
    MAXabs_thresh = params.get('MAXabs_thresh', np.max(V[:100]) + 2 * np.std(V[:100]))

    width_thresh = params.get('width_thresh_us', 8.0) # example
    if flag_plot:
        
        plt.figure()
        t_early_candidates = [t[s['peak_index']] for s in early_candidates]
        v_early_candidates = [V[s['peak_index']] for s in early_candidates]
        t_after_drift = [t[s['peak_index']] for s in after_drift]
        v_after_drift = [V[s['peak_index']] for s in after_drift]

        plt.plot(t_after_drift, v_after_drift, 'ro', label='after drift candidates')
        plt.plot(t_early_candidates, v_early_candidates, 'mo', label='early S2 candidates')
        plt.plot(t, V, label="Waveform")
        plt.axvline(t[s1_idx], color='k', label="S1")
        plt.axvline(t[first_peak_drift['peak_index']], color='r', label="First after S1")
        # plt.axvline(t[early['peak_index']], color='g', label="Early S2")
        # plt.axvline(t[main['peak_index']], color='r', label="Main S2")
        plt.axvline(t[inter_start], color='m', ls='--', label="Inter start")
        plt.axvline(t[inter_end], color='m', ls='--', label="Inter end")
        plt.gcf().legend()

    # apply cuts
    if rms >  RMS_thresh: return False, f"inter RMS too large {rms:.3e}"
    if maxabs >  MAXabs_thresh: return False, f"inter maxabs too large {maxabs:.3e}"
    
    if width_xray > width_thresh: return False, "early pulse too wide"
    
    # multiplicity check
    # n_extra_peaks = sum(1 for s in s2_list if s['peak_index']>inter_start and s['peak_index']<inter_end)
    # if n_extra_peaks > 0:
    #     return False, "extra peaks in inter-window"
    return True, "passed cuts"

def classify_event_v2(V, t, t_s1, s2_start, bs_threshold=5e-4,
                       min_s2_sep=3.0, min_s1_sep = 2.0, flag_plot=False):
    """Classify event based on inter-window noise between S1 and S2.
    Args:
        V: waveform values (1D array)
        t: time values (1D array, same length as V), in microseconds
        t_s1: time of S1 pulse (float, in microseconds)
        t_s2_start: time of start of S2 pulse (float, in microseconds)
        bs_threshold: baseline threshold to consider signal significant (float)
        min_t_inter: minimum required inter-window time (float, in microseconds)
        flag_plot: if True, generate diagnostic plot (bool)
    Returns:
        (bool, str): (True, reason) if event passes, (False, reason) if rejected
    """
    
    Vdrift = V[(t > t_s1) & (t < s2_start)]
    tdrift = t[(t > t_s1) & (t < s2_start)]

    Vdrift_bs = Vdrift[Vdrift > bs_threshold]
    tdrift_bs = tdrift[Vdrift > bs_threshold]
    if len(tdrift_bs) == 0:
        # print("False: No drift samples above baseline")
        return False, "no drift samples above baseline", (tdrift, Vdrift)

    Vs2 = V[t >= s2_start]
    Vs2_bs = Vs2[Vs2 > bs_threshold]
    # print(f"S2 signal above baseline: {Vs2_bs.sum():.3e}")
    if Vs2_bs.sum() > 1e5:
        # print('Detected alpha-like S2 signal above baseline, possible S1 in drift')
        return False, "excessive S2 signal above baseline", (tdrift, Vdrift)
    
    # time before S2
    t_pre_s2 = s2_start - tdrift_bs[-1]
    t_post_s1 = tdrift_bs[0] - t_s1
    if flag_plot:
        plt.figure()
        plt.plot(t, V, label="Waveform")
        plt.axvline(t_s1, color='k', label="S1")
        plt.axvline(s2_start, color='r', label="S2 start")
        plt.plot(tdrift_bs, Vdrift_bs, lw=2, label="Drift above baseline")
        # plt.axvline(tdrift_bs[-1], color='m', ls='--', label="Last drift above baseline")
        plt.gcf().legend()
        plt.title(f"t_pre_s2 = {t_pre_s2:.2f} us")

    if (t_pre_s2 > min_s2_sep) & (t_post_s1 > min_s1_sep):
        # print('Ok')
        return True, f"t_pre_s2 = {t_pre_s2:.2f} us > {min_s2_sep} us", (tdrift, Vdrift)
    else:
        # print(f"False: t_pre_s2 = {t_pre_s2:.2f} us <= {min_t_pre_s2} us")
        return False, f"t_pre_s2 = {t_pre_s2:.2f} us <= {min_s2_sep} us", (tdrift, Vdrift)

    