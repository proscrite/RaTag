import numpy as np
import matplotlib.pyplot as plt

def mark_peak_position(ax, x0, name, color='red', y_fraction=0.95):
    """
    Helper to mark a peak position with a vertical line and label.
    """
    ax.axvline(x0, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(x0, ax.get_ylim()[1] * y_fraction, name, 
            rotation=90, ha='left', va='top', fontsize=10, 
            fontweight='bold', color=color)


def plot_fit_peak(ax, fit_result, name, color, x_range=None, n_points=300):
    """
    Helper function to plot a fitted peak with vertical line marker.
    """
    # Extract peak position (works for both regular CB and Po212 composite fits)
    x0 = fit_result.params['cb_x0'].value
    
    # Define evaluation grid
    if x_range is None:
        E_fine = np.linspace(x0 - 0.5, x0 + 0.5, n_points)
    else:
        E_fine = np.linspace(x_range[0], x_range[1], n_points)
    
    # Evaluate fit and plot
    fit_curve = fit_result.eval(x=E_fine)
    ax.plot(E_fine, fit_curve, color=color, linewidth=2, 
            label=f'{name} (x₀={x0:.3f} MeV)', alpha=0.9)
    ax.axvline(x0, color=color, linestyle='--', linewidth=1, alpha=0.8)


def _get_twin_axes(ax_top):
    ax_bottom = ax_top.twiny()

    # 1. Force ax_top (Primary SCA) to the top
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position('top')
    
    # 2. Force ax_bottom (Twin Energy) to the bottom
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['bottom'].set_visible(True)
    ax_bottom.spines['bottom'].set_position(('axes', 0))
    ax_bottom.xaxis.tick_bottom()
    ax_bottom.xaxis.set_label_position('bottom')
    return ax_bottom

def _plot_fitted_spectrum(ax, energies_SCA, fit_results, ranges_V, calib_coeffs):
    """ax = Plots the primary histogram, the fit curves, and the range spans."""

    V_MIN, V_MAX = 3.5, 8.0
    ax.hist(energies_SCA, bins=200, range=(V_MIN, V_MAX), histtype='step', color='black', label='Data')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fit_results)))
    for i, (name, fit) in enumerate(fit_results.items()):
        x0_SCA = fit.params['cb_x0'].value
        sigma_SCA = fit.params['cb_sigma'].value
        
        x_grid_SCA = np.linspace(x0_SCA - 4*sigma_SCA, x0_SCA + 4*sigma_SCA, 200)
        y_grid = fit.eval(x=x_grid_SCA)
            
        ax.plot(x_grid_SCA, y_grid, color=colors[i], linewidth=2)
        ax.axvline(x0_SCA, color=colors[i], linestyle='--', alpha=0.7)
        ax.text(x0_SCA, ax.get_ylim()[1] * 0.95, name, rotation=90, 
                ha='right', va='top', color=colors[i], fontweight='bold')

    span_colors = plt.cm.Set3(np.linspace(0, 1, len(ranges_V)))
    for i, (name, (e_min, e_max)) in enumerate(ranges_V.items()):
        ax.axvspan(e_min, e_max, alpha=0.3, color=span_colors[i], label=f'{name} Window')
        
    ax.set(xlabel='Instrumental Scale [V]', ylabel='Counts', xlim=(V_MIN, V_MAX))


def _add_calibrated_twin_axis(ax, V_range, calib_coeffs):
    """Maps the secondary SCA axis onto the top of the plot."""
    a, b, c = calib_coeffs
    order = 2 if c is not None else 1
    
    def SCA_to_E(S_vals):
        return a * S_vals**2 + b * S_vals + c if order == 2 else a * S_vals + b

    ax.set_xlim(SCA_to_E(np.array(V_range)))
    ax.set_xlabel('Energy [MeV] (Calibrated Scale)', color='blue', fontweight='bold')
    ax.tick_params(axis='x', colors='blue')


def _add_calibration_inset(ax, fit_results, peak_definitions, calib_coeffs):
    """Draws the calibration curve in the top right corner."""
    a, b, c = calib_coeffs
    order = 2 if c is not None else 1
    
    def SCA_to_E(S_vals):
        return a * S_vals**2 + b * S_vals + c if order == 2 else a * S_vals + b

    ax_ins = ax.inset_axes([0.7, 0.55, 0.25, 0.35])
    
    SCA_anchors, E_anchors, anchor_names = [], [], []
    for peak in peak_definitions:
        name = peak['name']
        if name in fit_results:
            SCA_anchors.append(fit_results[name].params['cb_x0'].value)
            E_anchors.append(peak['ref_energy'])
            anchor_names.append(name)
            
    SCA_arr, E_arr = np.array(SCA_anchors), np.array(E_anchors)
    
    S_grid = np.linspace(min(SCA_arr)*0.95, max(SCA_arr)*1.05, 50)
    E_grid = SCA_to_E(S_grid)
    
    ax_ins.plot(S_grid, E_grid, 'r-', alpha=0.7)
    ax_ins.plot(SCA_arr, E_arr, 'ko', markersize=4)
    
    for n, x, y in zip(anchor_names, SCA_arr, E_arr):
        ax_ins.annotate(n, (x, y), xytext=(5, -5), textcoords='offset points', fontsize=8)
        
    ax_ins.set(xlabel='SCA', ylabel='MeV', xlim=(min(SCA_arr)*0.95, max(SCA_arr)*1.05), title='Calibration Curve (Ord. {})'.format(order))
    ax_ins.tick_params(labelsize=8)
    ax_ins.grid(True, alpha=0.3)


def plot_calibration_summary(energies_SCA, fit_results, calib_coeffs, ranges_V, peak_definitions):
    """Master Plotter: Orchestrates the construction of the QA dashboard."""

    fig, ax_top = plt.subplots(figsize=(12, 7), layout='constrained')
    ax_bottom = _get_twin_axes(ax_top)
    
    _plot_fitted_spectrum(ax_top, energies_SCA, fit_results, ranges_V, calib_coeffs)
    V_range = ax_top.get_xlim()
    _add_calibrated_twin_axis(ax_bottom, V_range, calib_coeffs)
    _add_calibration_inset(ax_top, fit_results, peak_definitions, calib_coeffs)
    
    # Legend is handled at the master level
    ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
    return fig