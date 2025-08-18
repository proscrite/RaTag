#!/usr/bin/env python3
# filepath: process_wfm.py

import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from RaTag.scripts.wfm2read_fast import wfm2read


def moving_average(y, window):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def compute_rise_times(V_s, t, peaks, window):
    """Return array of 10–90% rise times for each peak index in peaks."""
    rt = []
    for pi in peaks:
        seg, tseg = V_s[:pi+1], t[:pi+1]
        base = np.median(seg[:window])
        amp  = seg[pi] - base

        thr10, thr90 = base + 0.1*amp, base + 0.9*amp
        i10 = np.where(seg >= thr10)[0][0]
        i90 = np.where(seg >= thr90)[0][0]

        t10 = np.interp(thr10, seg[[i10-1, i10]], tseg[[i10-1, i10]])
        t90 = np.interp(thr90, seg[[i90-1, i90]], tseg[[i90-1, i90]])
        rt.append(t90 - t10)
    return np.array(rt)


def classify_s1_s2(peaks, rise_times):
    """Return indices of S1 (min rise) and S2 (max rise) peaks."""
    s1_idx = peaks[np.argmin(rise_times)]
    s2_idx = peaks[np.argmax(rise_times)]
    return s1_idx, s2_idx


def compute_s2_area(V, V_s, t, idx_s2, window):
    """Integrate raw V under S2 between 10% thresholds (uV·s)."""
    base = np.median(V_s[:idx_s2])
    amp  = V_s[idx_s2] - base
    thr10 = base + 0.1*amp

    left = np.where(V_s[:idx_s2] < thr10)[0][-1]
    down = np.where(V_s[idx_s2:] < thr10)[0]
    right = idx_s2 + (down[0] if down.size else len(V_s)-1)

    area = np.trapz(V[left:right], t[left:right]) * 1e6
    return area


def process_file(path, trig_t, trig_v,
                 window=50, height_frac=0.2, dist_frac=1/20):
    wf = wfm2read(path)
    t, V = wf[1], -wf[0]

    V_s = moving_average(V, window)
    peaks, props = find_peaks(
        V_s,
        height=np.max(V_s)*height_frac,
        distance=len(t)*dist_frac
    )

    # init
    s1_t = s1_amp = s2_t = s2_amp = drift_time = trig_delay = None
    rise1 = rise2 = area_s2 = None

    if peaks.size >= 2:
        rt = compute_rise_times(V_s, t, peaks, window)
        i_s1, i_s2 = classify_s1_s2(peaks, rt)

        s1_t, s1_amp = t[i_s1], V_s[i_s1]
        s2_t, s2_amp = t[i_s2], V_s[i_s2]
        rise1, rise2 = rt.min(), rt.max()

        drift_time = s2_t - s1_t
        tp = np.argmax(trig_v)
        if i_s1 < tp:
            trig_delay = trig_t[tp] - s1_t

        area_s2 = compute_s2_area(V, V_s, t, i_s2, window)

    return {
        'file': os.path.basename(path),
        's1_t_s': s1_t,
        's1_amp_V': s1_amp,
        's2_t_s': s2_t,
        's2_amp_V': s2_amp,
        'rise_time_s1_ns': rise1*1e9 if rise1 is not None else None,
        'rise_time_s2_ns': rise2*1e9 if rise2 is not None else None,
        'drift_time_ns': drift_time*1e9 if drift_time is not None else None,
        'trigger_delay_ns': trig_delay*1e9 if trig_delay is not None else None,
        'area_s2_uV_s': area_s2
    }


def main():
    p = argparse.ArgumentParser(
        description='Process RUN*.wfm files: extract S1/S2 peaks, rise times, drift & trigger delays, S2 area.'
    )
    p.add_argument('data_dir', help='folder containing RUN*.wfm files')
    p.add_argument('trigger_file', help='trigger waveform .wfm')
    p.add_argument('-o', '--output', default='results.csv',
                   help='output CSV filename')
    args = p.parse_args()

    trig = wfm2read(args.trigger_file)
    trig_t, trig_v = trig[1], trig[0]
    trig_v = trig_v / (2*np.max(trig_v))

    files = sorted(glob.glob(os.path.join(args.data_dir, 'RUN*.wfm')))
    records = [process_file(f, trig_t, trig_v) for f in files]

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f'Wrote {len(df)} entries to {args.output}')


if __name__ == '__main__':
    main()