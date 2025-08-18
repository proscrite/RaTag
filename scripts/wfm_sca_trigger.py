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


def process_file(path, trig_t, trig_v,
                 window=50, height_frac=0.2, dist_frac=1/20):
    wf = wfm2read(path)
    t, V = wf[1], -wf[0]

    # smooth
    V_s = moving_average(V, window)

    # find peaks
    peaks, props = find_peaks(
        V_s,
        height=np.max(V_s) * height_frac,
        distance=len(t) * dist_frac
    )

    # init results
    s1_t = s1_amp = s2_t = s2_amp = drift_time = trig_delay = None

    if len(peaks) >= 2:
        # identify S2 as the highest peak, S1 as the first peak before it
        main_idx = np.argmax(props['peak_heights'])
        s2_i = peaks[main_idx]
        s1_i = peaks[0] if peaks[0] < s2_i else None

        if s1_i is not None:
            s1_t, s1_amp = t[s1_i], V[s1_i]
            s2_t, s2_amp = t[s2_i], V[s2_i]
            drift_time = s2_t - s1_t

            # trigger delay
            trig_peak = np.argmax(trig_v)
            if s1_i < trig_peak:
                trig_delay = trig_t[trig_peak] - s1_t

    return {
        'file': os.path.basename(path),
        's1_t': s1_t,
        's1_amp': s1_amp,
        's2_t': s2_t,
        's2_amp': s2_amp,
        'drift_time_ns': (drift_time * 1e9) if drift_time is not None else None,
        'trigger_delay_ns': (trig_delay * 1e9) if trig_delay is not None else None
    }


def main():
    p = argparse.ArgumentParser(
        description='Process .wfm files, extract S1/S2 peaks, rise times, drift and trigger delays.'
    )
    p.add_argument('data_dir', help='folder containing .wfm files')
    p.add_argument('trigger_file', help='single .wfm trigger waveform')
    p.add_argument('-o', '--output', default='results.csv',
                   help='output CSV filename')
    args = p.parse_args()

    # read & normalize trigger
    trig = wfm2read(args.trigger_file)
    trig_t, trig_v = trig[1], trig[0]
    trig_v = trig_v / (2 * np.max(trig_v))

    files = sorted(glob.glob(os.path.join(args.data_dir, 'RUN*.wfm')))
    rows = [process_file(f, trig_t, trig_v) for f in files]

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f'Wrote {len(df)} entries to {args.output}')


if __name__ == '__main__':
    main()