import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.figure import Figure

from RaTag.waveform.preprocessing import moving_average, subtract_pedestal
from RaTag.core.datatypes import PMTWaveform, SiliconWaveform
from RaTag.core.config import TimingConfig


def plot_s2_diagnostic(wf, s2_results: dict, config, frame_idx: int) -> None:
    """
    Diagnostic plotter that explicitly mirrors the find_s2 engine physics
    and reads the boolean ledger for absolute diagnostic truth.
    """
    # 1. Exact Engine Physics Replication
    wf_sub0 = subtract_pedestal(wf, n_points=config.n_pedestal)
    v_raw_sub = wf_sub0.v[frame_idx] if wf_sub0.ff else wf_sub0.v
    
    wf_smooth = moving_average(wf, window=config.s2_window_ma)
    wf_sub_smooth = subtract_pedestal(wf_smooth, n_points=config.n_pedestal)
    
    v_smooth = wf_sub_smooth.v[frame_idx] if wf_sub_smooth.ff else wf_sub_smooth.v
    t_frame = wf.t
    frame_uid = wf.uids[frame_idx]
    
    # 2. Extract 1:1 Data Lineage
    frame_area = s2_results.get('raw_areas', np.zeros(wf.nframes))[frame_idx]
    true_peak = s2_results.get('v_smooth_peaks', np.zeros(wf.nframes))[frame_idx]
    
    ledger = s2_results.get('cut_ledger', {})
    if not ledger:
        print("Cut ledger missing. Run find_s2 with updated engine.")
        return

    # 3. Determine Exact Failure Cause
    status_msg = "ACCEPTED"
    plot_color = 'green'
    
    if not ledger['pass_clip'][frame_idx]:
        status_msg = "REJECTED (Failed Anti-Clip)"
        plot_color = 'crimson'
    elif not ledger['pass_v_min'][frame_idx]:
        status_msg = f"REJECTED (Failed v_min: Peak is {true_peak:.3f} mV)"
        plot_color = 'darkorange'
    elif not ledger['pass_area'][frame_idx]:
        status_msg = "REJECTED (Failed Area Bounds)"
        plot_color = 'purple'
        
    # 4. Render
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t_frame, v_raw_sub, label='Raw Waveform (Subtracted)', color='blue', alpha=0.5)
    ax.plot(t_frame, v_smooth, label=f'Smoothed (MA: {config.s2_window_ma})', color='black', linewidth=2)
    
    ax.axhline(config.s2_v_min, color='goldenrod', linestyle='-.', label=f'V_min Cut ({config.s2_v_min} mV)')
    
    # Calculate thresholds for visual reference only
    search_mask = t_frame > config.s2_start_min
    if np.any(search_mask):
        peak_val = np.max(v_smooth[search_mask])
        ax.axhline(peak_val * config.s2_fraction_left, color='green', linestyle='--', label=f'Left Thresh ({config.s2_fraction_left*100}%)')
        ax.axhline(peak_val * config.s2_fraction_right, color='red', linestyle='--', label=f'Right Thresh ({config.s2_fraction_right*100}%)')

    if status_msg == "ACCEPTED":
        try:
            idx = np.where(s2_results['uids'] == frame_uid)[0][0]
            ax.axvspan(s2_results['start_times'][idx], s2_results['end_times'][idx], color='green', alpha=0.2, label='Integration Window')
        except IndexError:
            pass

    ax.set_title(f"UID: {frame_uid} | {status_msg}\nComputed Area: {frame_area:.3f} mV·µs", color=plot_color, fontweight='bold')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)


# ============================================================================
# 1. PHYSICS & MATH (Pure Compute Helpers)
# ============================================================================

def compute_pmt_diagnostics(wf_pmt: PMTWaveform, config: TimingConfig, t_start_delay: float = 0.0, integ_interval: float = 1.5) -> dict:
    """Re-inlines the S2 detection math to expose all traces for plotting."""
    dt = wf_pmt.dt
    t = wf_pmt.t
    v_raw = wf_pmt.v
    
    baseline_pmt = np.mean(v_raw[:config.n_pedestal])
    v_sub = v_raw - baseline_pmt
    v_env = maximum_filter1d(v_sub, size=config.s2_window_ma)
    
    search_mask = t > config.s2_start_min
    start_t, end_t, s2_area, peak_v, dynamic_thresh = 0.0, 0.0, 0.0, 0.0, 0.0
    status = "REJECTED"
    
    if np.any(search_mask):
        search_indices = np.where(search_mask)[0]
        peak_idx_local = np.argmax(v_env[search_mask])
        peak_idx_global = search_indices[peak_idx_local]
        peak_v = v_env[peak_idx_global]
        
        dynamic_thresh = peak_v * config.s2_fraction
        below_thresh = v_env <= dynamic_thresh
        
        # 1D Boundary Searches anchoring from the peak
        left_side = below_thresh[:peak_idx_global]
        start_idx = np.where(left_side)[0][-1] if np.any(left_side) else 0
        
        right_side = below_thresh[peak_idx_global:]
        end_idx = np.where(right_side)[0][0] + peak_idx_global if np.any(right_side) else len(v_env) - 1

        peak_t = t[peak_idx_global]
        start_t = t[start_idx]
        end_t = t[end_idx]
        
        # int_mask = (t >= peak_t) & (t <= end_t)
        int_mask = (t >= peak_t + t_start_delay) & (t <= peak_t + integ_interval - t_start_delay)
        s2_area = np.sum(v_sub[int_mask]) * dt
        
        if (s2_area > config.s2_min_area) and (s2_area < config.s2_max_area):
            status = "ACCEPTED"

    return {
        't': t, 'v_raw': v_raw, 'v_sub': v_sub, 'v_env': v_env,
        'search_mask': search_mask, 'dynamic_thresh': dynamic_thresh,
        # 'start_t': peak_t, 'end_t': end_t, 's2_area': s2_area, 
        'start_t': peak_t + t_start_delay, 'end_t': peak_t + integ_interval - t_start_delay, 's2_area': s2_area, 
        'peak_v': peak_v, 'status': status
    }


def compute_alpha_diagnostics(wf_alpha: SiliconWaveform, isotope_ranges_V: dict = None) -> dict:
    """Reconstructs the alpha energy and tags the isotope based on CI95 voltage boundaries."""
    t = wf_alpha.t
    v_raw = wf_alpha.v
    
    baseline = np.median(v_raw[:200])
    v_sub = v_raw - baseline
    
    v_smooth = savgol_filter(v_sub, window_length=501, polyorder=3)
    
    peak_idx = np.argmax(v_smooth)
    peak_v = v_smooth[peak_idx]
    peak_t = t[peak_idx]
    energy = peak_v / 1.058  # Instrumental calibration factor
    
    tagged_isotope = "Out of Bounds"
    if isotope_ranges_V:
        for iso, (v_min, v_max) in isotope_ranges_V.items():
            if v_min <= peak_v <= v_max:
                tagged_isotope = iso
                break

    return {
        't': t, 'v_raw': v_raw, 'v_smooth': v_smooth,
        'peak_v': peak_v, 'peak_t': peak_t, 'energy': energy,
        'tagged_isotope': tagged_isotope
    }

# ============================================================================
# 2. PRESENTATION LAYER (Pure Side-Effects)
# ============================================================================

def plot_pmt_diagnostic(ax: plt.Axes, data: dict, frame_idx: int):
    """Declarative plotter for the PMT S2 channel."""
    # ax.plot(data['t'], data['v_raw'], color='lightgray', linewidth=1, label='Raw Waveform')
    ax.plot(data['t'], data['v_sub'], color='blue', linewidth=0.8, alpha=0.7, label='Raw waveform')
    # ax.plot(data['t'], data['v_env'], color='green', linewidth=1.2, alpha=0.8, label='S2 Envelope (Max Filter)')
    
    if np.any(data['search_mask']):
        # ax.axhline(data['dynamic_thresh'], color='orange', linestyle=':', label=f'Threshold ({data['dynamic_thresh']:.1f} mV)')
        ax.axvspan(data['start_t'], data['end_t'], color='red', alpha=0.1, label='Found S2 Boundaries')
        
    ax.set_ylabel('PMT Signal (mV)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    
    diag_text = (
        f"Frame {frame_idx} Diagnostics | Status: {data['status']}\n"
        f"---------------------------------------------------\n"
        f"S2 Area:   {data['s2_area']:.2f} mV·µs\n"
        f"Start:     {data['start_t']:.2f} µs\n"
        f"End:       {data['end_t']:.2f} µs\n"
        f"Peak:      {data['peak_v']:.2f} mV"
    )
    ax.text(0.02, 0.95, diag_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def plot_alpha_diagnostic(ax: plt.Axes, data: dict):
    """Declarative plotter for the Alpha channel."""
    ax.plot(data['t'], data['v_raw'], color='lightgray', linewidth=1, label='Raw Alpha')
    ax.plot(data['t'], data['v_smooth'], color='red', linewidth=1.5, alpha=0.9, label='SavGol Filtered')
    
    # Isotope Tagging Marker
    tag = data['tagged_isotope']
    marker_color = 'purple' if tag != "Out of Bounds" else 'black'
    label_str = f"Peak: {data['energy']:.2f} MeV [{tag}]"
    
    ax.plot(data['peak_t'], data['peak_v'], marker='o', color=marker_color, label=label_str)
    
    ax.set_ylabel('Alpha Signal (mV)', fontweight='bold')
    ax.set_xlabel('Time (µs)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)

# ============================================================================
# 3. ORCHESTRATOR
# ============================================================================

def plot_full_coincidence_diagnostic(wf_pmt: PMTWaveform, 
                                     wf_alpha: SiliconWaveform, 
                                     isotope_ranges_V: dict = None,
                                     config: TimingConfig = TimingConfig()) -> Figure:
    """
    Depth-of-1 Orchestrator:
    Manages layout state, delegates math to compute helpers, and hands results to plot helpers.
    """
    fig, ax_pmt = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.05)
    
    # 1. Math
    pmt_data = compute_pmt_diagnostics(wf_pmt, config)
    
    
    if wf_alpha is not None:
        alpha_data = compute_alpha_diagnostics(wf_alpha, isotope_ranges_V)
        fig, (ax_pmt, ax_alpha) = plt.subplots(2, 1, sharex=True, figsize=(14, 10))
        plot_alpha_diagnostic(ax_alpha, alpha_data)
        # ax_alpha.set_xlim(pmt_data['t'][0], pmt_data['t'][-1])
    
    plot_pmt_diagnostic(ax_pmt, pmt_data, wf_pmt.frame_idx)
    
    return fig