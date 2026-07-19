import numpy as np
import re
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from RaTag.core.datatypes import Run, SetPmt
from RaTag.core.functional import map_over
from RaTag.core.decorators import allow_force, load_cached_metadata, load_cached_npz, write_metadata, write_npz_arrays, limit_frames
from RaTag.core.config import TimingConfig
from RaTag.io import file_ops
from RaTag.waveform.preprocessing import subtract_pedestal
from RaTag.thgem_tpc.timing_workflow import find_s2



@allow_force
@load_cached_metadata(target_attr='n_areas_purity')
@load_cached_npz(signal_type='purity_metrics')
@write_metadata(target_attr='n_areas_purity')
@write_npz_arrays(signal_type='purity_metrics')
@limit_frames
def resolve_set_purity(set_pmt: SetPmt, max_files: int = None,
                       config: TimingConfig = TimingConfig(),
                       force: bool = False) -> tuple[SetPmt, dict]:
    """
    Worker function: Finds S2s, integrates area, calculates exact pulse width, 
    and tags every valid event with the real-world acquisition timestamp.
    """
    
    all_s2_areas, all_s2_widths, all_timestamps, uids_out = [], [], [], []
    total_frames = 0
    valid_frames = 0

    # Guarantee the filenames list matches the iterator bounds exactly
    filenames_to_process = set_pmt.filenames[:max_files] if max_files else set_pmt.filenames

    # ZIP the physical filename with the loaded waveform to guarantee 1:1 mapping
    wfm_iterator = file_ops.iter_waveforms(set_pmt, max_files=max_files, show_progress=True)
    
    for filename, wf in zip(filenames_to_process, wfm_iterator):
        wf_sub = subtract_pedestal(wf, config.n_pedestal)
        total_frames += wf.nframes
        
        # 1. Parse chronological time for this specific file
        file_timestamp = file_ops.parse_wfm_timestamp(filename)
        
        # 2. S2 Search (Pure S2 tracking, ignoring S1)
        t_min_s2 = -2.5 
        s2_out = find_s2(wf, config, t_min_s2)
        s2_starts, s2_ends = s2_out[0], s2_out[1]
        
        has_s2 = ~np.isnan(s2_starts) & ~np.isnan(s2_ends)
        
        # 3. Apply your late-S2 background cut 
        s2_start_max = getattr(config, 's2_start_max', 0.5)
        is_prompt = s2_starts <= s2_start_max
        
        valid_mask = has_s2 & is_prompt
        if not np.any(valid_mask):
            continue
            
        valid_frames += np.sum(valid_mask)
        
        # 4. Math: Width & Integration
        s2_widths = s2_ends - s2_starts
        
        dt = wf_sub.t[1] - wf_sub.t[0]
        t_2d = wf_sub.t[np.newaxis, :]
        starts_2d = s2_starts[:, np.newaxis]
        ends_2d = s2_ends[:, np.newaxis]
        
        int_mask = (t_2d >= starts_2d) & (t_2d <= ends_2d)
        v_full = wf_sub.v if wf_sub.ff else wf_sub.v[np.newaxis, :]
        
        # Mask and integrate
        v_valid = v_full[valid_mask]
        mask_valid = int_mask[valid_mask]
        areas = np.sum(v_valid * mask_valid, axis=1) * dt
        
        # Create a broadcasted array of the timestamp matching the number of valid frames
        timestamps_arr = np.full(np.sum(valid_mask), file_timestamp)
        
        # 5. Accumulate
        all_s2_areas.append(areas)
        all_s2_widths.append(s2_widths[valid_mask])
        all_timestamps.append(timestamps_arr)
        uids_out.append(wf.uids[valid_mask])

    # Package the custom purity payload
    arrays = {
        "uids": np.concatenate(uids_out) if uids_out else np.array([]),
        "s2_areas": np.concatenate(all_s2_areas) if all_s2_areas else np.array([]),
        "s2_widths": np.concatenate(all_s2_widths) if all_s2_widths else np.array([]),
        "timestamps": np.concatenate(all_timestamps) if all_timestamps else np.array([])
    }
    
    stats = {'n_areas_purity': int(valid_frames)}
    print(f"  {set_pmt.source_dir.name}: {valid_frames}/{total_frames} purity events logged.")
    
    return replace(set_pmt, **stats), arrays

def map_purity_extraction(run: Run, max_frames: int = None,
                          config: TimingConfig = TimingConfig(),
                          force: bool = False) -> Run:
    """Entry point: Maps purity extraction across all sets in the Run."""
    print("\n" + "="*60 + f"\nEXTRACTING PURITY METRICS: {run.run_id}\n" + "="*60)
    bound_timing = lambda s: resolve_set_purity(s, max_frames=max_frames, config=config, force=force)
    
    new_sets = map_over(run.sets, bound_timing, catch_errors=True)
    return replace(run, sets=new_sets)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from RaTag.io.file_ops import load_npz_arrays

# Execute the workflow
# runif = map_purity_extraction(runif, force=True)

def plot_gas_purity_evolution(run):
    print(f"Aggregating Purity data for Run {run.run_id}...")
    
    all_areas = []
    all_widths = []
    all_times = []
    
    for set_pmt in run.sets:
        arrays = load_npz_arrays(set_pmt, 'purity_metrics')
        if not arrays:
            continue
            
        all_areas.append(arrays.get('s2_areas', np.array([])))
        all_widths.append(arrays.get('s2_widths', np.array([])))
        all_times.append(arrays.get('timestamps', np.array([])))
        
    if not all_areas:
        print("No purity data found.")
        return
        
    # Flatten the arrays
    areas_flat = np.concatenate(all_areas)
    widths_flat = np.concatenate(all_widths)
    times_flat = np.concatenate(all_times)
    
    # Convert UNIX timestamps to Pandas Datetime
    df = pd.DataFrame({
        'Time': pd.to_datetime(times_flat, unit='s'),
        'S2_Area': areas_flat,
        'S2_Width': widths_flat
    })
    
    # Sort chronologically just in case sets were loaded out of order
    df = df.sort_values('Time')
    
    # ==========================================
    # Plotting
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, layout="constrained")
    
    # Format the X-axis for beautiful datetime strings
    date_format = mdates.DateFormatter('%a, %H:%M') # e.g., "Sun, 12:20"
    
    # 1. Plot S2 Area (Electron Lifetime proxy)
    ax1.scatter(df['Time'], df['S2_Area'], alpha=0.05, s=2, color='tab:blue')
    
    # Overlay a rolling median to show the trend clearly through the noise
    rolling_area = df['S2_Area'].rolling(window=500, center=True).median()
    ax1.plot(df['Time'], rolling_area, color='red', lw=2, label='500-Event Median')
    
    ax1.set_title(f"Gas Purification Evolution - Run {run.run_id}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("S2 Area [mV·μs]", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. Plot S2 Width (Longitudinal Diffusion proxy)
    ax2.scatter(df['Time'], df['S2_Width'], alpha=0.05, s=2, color='tab:green')
    
    rolling_width = df['S2_Width'].rolling(window=500, center=True).median()
    ax2.plot(df['Time'], rolling_width, color='darkgreen', lw=2, label='500-Event Median')
    
    ax2.set_ylabel("S2 Width [µs]", fontsize=12)
    ax2.set_xlabel("Acquisition Time", fontsize=12)
    ax2.xaxis.set_major_formatter(date_format)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.show()
