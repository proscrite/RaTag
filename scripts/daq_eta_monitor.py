"""DAQ efficiency monitoring script.

Usage: python daq_efficiency_monitor.py <path_to_daq_dir> <total_files>

This script:
- monitors .wfm files in the given directory using the file creation time (st_ctime)
- computes time differences between consecutive files
- determines a moving-average window size based on the number of files (min 10, max 100)
  - if only 1 file is present it waits until at least 10 files are available
- plots only the moving-average of the CTime differences
- prints an estimated remaining time formatted like "1 hour, 37 min"

Only the creation time (st_ctime) is used for live estimation.
"""
from __future__ import annotations

import argparse
import time
from glob import glob
from pathlib import Path
from typing import Tuple, Optional

# plotting disabled: comment out matplotlib dependency for headless use
# import matplotlib.pyplot as plt
import numpy as np


def get_ctimes(path_daq: Path) -> np.ndarray:
    """Return sorted creation times (relative to first file) for all .wfm files."""
    file_list = sorted(glob(str(path_daq / "*.wfm")))
    if not file_list:
        return np.array([])
    times = np.array(sorted([Path(f).stat().st_ctime for f in file_list]))
    times = times - times[0]
    return times


def diffs_from_ctimes(ctimes: np.ndarray) -> np.ndarray:
    """Return array of time differences between consecutive creation times.

    If fewer than 2 files exist, returns an empty array.
    """
    if ctimes.size < 2:
        return np.array([])
    return ctimes[1:] - ctimes[:-1]


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if data.size == 0 or window_size < 1:
        return np.array([])
    if window_size == 1:
        return data.copy()
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def format_eta(seconds: float) -> str:
    """Format seconds as 'H hour(s), M min' or 'M min' if under 1 hour."""
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        hour_label = "hour" if hours == 1 else "hours"
        return f"{hours} {hour_label}, {minutes} min"
    return f"{minutes} min"


def choose_window_size(n_files: int) -> int:
    """Choose window_size according to rules:
    - If only 1 file: caller should wait until >=10
    - When >=10 files available, use at least 10
    - When >=20 files available, use 20
    - Cap at 100
    Implementation: window = min(max(10, n_files), 100)
    The actual window used for moving average must not exceed len(diffs).
    """
    return min(max(10, n_files), 100)


def monitor_and_plot(path: Path,
                     total_files: int,
                     poll_interval: float = 5.0,
                     method: str = "ewma",
                     alpha: float = 0.25,
                     plot: bool = False,
                     max_no_new: int = 3) -> None:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    # If the provided path contains subdirectories, choose the first-level child
    # which contains the most recently modified .wfm file. This lets you specify
    # a parent folder and automatically monitor the newest run.
    def _choose_latest_child(parent: Path) -> Optional[Path]:
        child_dirs = [p for p in parent.iterdir() if p.is_dir()]
        best = None
        best_time = -1.0
        for d in child_dirs:
            files = sorted(glob(str(d / "*.wfm")))
            if not files:
                continue
            # use file modified time for selection (most recent activity)
            mtimes = [Path(f).stat().st_mtime for f in files]
            mtime_max = max(mtimes)
            if mtime_max > best_time:
                best_time = mtime_max
                best = d
        return best

    # If the path itself has subdirs with .wfm files, switch to the most recent child
    chosen_child = _choose_latest_child(path)
    if chosen_child is not None:
        print(f"Parent path given; auto-selecting most-recent child: {chosen_child}")
        path = chosen_child

    print(f"Monitoring directory: {path}  — target total files: {total_files}")

    # Optionally prepare plotting (import matplotlib only when requested)
    fig = ax = None
    lines = {}
    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        lines['movavg'], = ax.plot([], [], label='CTime moving avg')
        if method == 'ewma':
            lines['ewma'], = ax.plot([], [], label=f'EWMA (alpha={alpha})')
        ax.set(xlabel='Index (moving average)', ylabel='Time between files (s)',
               title=f'DAQ Efficiency Monitoring — {path.name}')
        # secondary axis for ETA (minutes)
        ax_eta = ax.twinx()
        lines['eta'], = ax_eta.plot([], [], color='C2', linestyle='--', label='ETA (min)')
        ax_eta.set_ylabel('Estimated time remaining (min)', color='C2')
        ax.legend(loc='upper left')

    try:
        # Continuous monitoring loop; recompute ETA based on the most recent moving-average value
        # guardrail: abort if no new files are created for N consecutive polls
        last_count = 0
        no_new_iterations = 0
        eta_history = []
        iteration = 0
        while True:
            ctimes = get_ctimes(path)
            n_files = 0 if ctimes.size == 0 else (ctimes.size)

            if n_files == 0:
                print("No .wfm files found yet. Waiting...")
                time.sleep(poll_interval)
                continue

            if n_files == 1:
                print("Only 1 file present — waiting until at least 10 files are available...")
                time.sleep(poll_interval)
                continue

            # Guardrail: detect if no new files are being created. Count only after at least 2 files exist.
            if n_files > last_count:
                last_count = n_files
                no_new_iterations = 0
            else:
                no_new_iterations += 1
                if no_new_iterations >= max_no_new:
                    print(f"No new files created for {no_new_iterations} consecutive polls ({poll_interval}s each); aborting monitoring.")
                    break

            diffs = diffs_from_ctimes(ctimes)
            if diffs.size == 0:
                print("Not enough timestamps yet. Waiting...")
                time.sleep(poll_interval)
                continue

            chosen_window = choose_window_size(n_files)
            actual_window = min(chosen_window, diffs.size)
            movavg = moving_average(diffs, actual_window)

            # Compute estimator according to method
            total_avg = None
            if method == 'last':
                # most recent moving-average
                if movavg.size > 0:
                    total_avg = float(movavg[-1])
                else:
                    total_avg = float(np.mean(diffs))
            elif method == 'median':
                # median of last up-to-10 movavg values
                if movavg.size > 0:
                    k = min(10, movavg.size)
                    total_avg = float(np.median(movavg[-k:]))
                else:
                    total_avg = float(np.median(diffs))
            elif method == 'ewma':
                # compute EWMA across the movavg series for plotting and estimator
                if movavg.size > 0:
                    ewma = np.empty_like(movavg)
                    ewma[0] = movavg[0]
                    for i in range(1, len(movavg)):
                        ewma[i] = alpha * movavg[i] + (1 - alpha) * ewma[i - 1]
                    total_avg = float(ewma[-1])
                else:
                    total_avg = float(np.mean(diffs))
            else:
                # fallback to last
                if movavg.size > 0:
                    total_avg = float(movavg[-1])
                else:
                    total_avg = float(np.mean(diffs))

            current_count = n_files
            remaining_files = max(0, total_files - current_count)

            if remaining_files <= 0:
                print(f"Target reached: {current_count} files (>= {total_files}).")
                break

            estimated = remaining_files * total_avg
            formatted = format_eta(estimated)
            print(f"Current: {current_count} files. Estimated time remaining for {remaining_files} files: {formatted}")

            # record ETA history (in minutes)
            eta_minutes = estimated / 60.0
            eta_history.append(eta_minutes)

            # Update plot when enabled
            if plot and movavg.size > 0:
                x = np.arange(len(movavg))
                lines['movavg'].set_data(x, movavg)
                if method == 'ewma' and 'ewma' in lines:
                    lines['ewma'].set_data(x, ewma)
                # update eta plot
                x_eta = np.arange(len(eta_history))
                lines['eta'].set_data(x_eta, eta_history)
                # rescale both axes
                ax.relim()
                ax.autoscale_view()
                try:
                    ax_eta.relim()
                    ax_eta.autoscale_view()
                except Exception:
                    pass
                # update legends to include ETA
                try:
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax_eta.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, loc='upper left')
                except Exception:
                    pass
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()

            # Sleep until next poll
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Monitoring canceled by user.")
    finally:
        # If plotting is re-enabled, close the figure to free resources:
        # plt.close(fig)
        pass


def main():
    p = argparse.ArgumentParser(description="DAQ efficiency monitoring (uses st_ctime only)")
    p.add_argument("path", type=str, help="Path where DAQ stores .wfm files")
    p.add_argument("total_files", type=int, help="Desired total number of files to acquire")
    p.add_argument("--method", choices=["last", "ewma", "median"], default="ewma",
                   help="Estimator method: 'last' uses the last movavg, 'ewma' uses EWMA, 'median' uses recent median")
    p.add_argument("--alpha", type=float, default=0.25, help="Alpha for EWMA (only used with --method ewma)")
    p.add_argument("--plot", action="store_true", help="Enable plotting (imports matplotlib and shows live plot)")
    p.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polling the directory")
    p.add_argument("--max-no-new", type=int, default=3, help="Consecutive polls with no new files before abort")
    args = p.parse_args()
    monitor_and_plot(Path(args.path), int(args.total_files), poll_interval=float(args.poll_interval),
                     method=args.method, alpha=float(args.alpha), plot=bool(args.plot), max_no_new=int(args.max_no_new))


if __name__ == "__main__":
    main()
