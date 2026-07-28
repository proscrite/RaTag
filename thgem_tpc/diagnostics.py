import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d
from matplotlib.figure import Figure

from RaTag.core.datatypes import PMTWaveform, SiliconWaveform
from RaTag.core.config import TimingConfig

# ============================================================================
# 1. PHYSICS & MATH (Pure Compute Helpers)
# ============================================================================

def compute_pmt_diagnostics(wf_pmt: PMTWaveform, config: TimingConfig) -> dict:
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
        
        int_mask = (t >= peak_t) & (t <= end_t)
        s2_area = np.sum(v_sub[int_mask]) * dt
        
        if (s2_area > config.s2_min_area) and (s2_area < config.s2_max_area):
            status = "ACCEPTED"

    return {
        't': t, 'v_raw': v_raw, 'v_sub': v_sub, 'v_env': v_env,
        'search_mask': search_mask, 'dynamic_thresh': dynamic_thresh,
        'start_t': peak_t, 'end_t': end_t, 's2_area': s2_area, 
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
    # fig, (ax_pmt, ax_alpha) = plt.subplots(2, 1, sharex=True, figsize=(14, 10))
    fig, ax_pmt = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.05)
    
    # 1. Delegate Math
    pmt_data = compute_pmt_diagnostics(wf_pmt, config)
    # alpha_data = compute_alpha_diagnostics(wf_alpha, isotope_ranges_V)
    
    # 2. Delegate Side-Effects
    plot_pmt_diagnostic(ax_pmt, pmt_data, wf_pmt.frame_idx)
    # plot_alpha_diagnostic(ax_alpha, alpha_data)
    
    # 3. Enforce Shared Context
    # ax_alpha.set_xlim(pmt_data['t'][0], pmt_data['t'][-1])
    
    return fig