"""
Isotope preparation pipeline - Setup for multi-isotope analysis.

This pipeline handles prerequisite steps for multi-isotope runs:
1. Energy map generation (UID → energy mapping files)

Structure mirrors run_preparation.py and recoil_only.py:
- Declarative composition using functools.partial
- No iteration logic (handled by workflow functions)
- Single-purpose, composable stages
"""

from functools import partial

from RaTag.core.datatypes import Run
from RaTag.core.functional import pipe_run
from RaTag.workflows.energy_mapping import create_energy_maps_in_run, plot_energy_spectra_in_run


# ============================================================================
# PIPELINE - FUNCTIONAL COMPOSITION
# ============================================================================

def prepare_isotope_separation(run: Run,
                               files_per_chunk: int = 10,
                               fmt: str = "8b",
                               scale: float = 0.1,
                               pattern: str = "*Ch4.wfm",
                               savgol_window: int = 501,
                               nbins: int = 120,
                               energy_range: tuple = (4, 8)) -> Run:
    """
    Prepare isotope separation by generating energy maps and spectra plots.
    
    This pipeline creates binary mapping files (energy_map_*.bin) that enable
    UID-to-energy lookups for isotope assignment in multi-isotope runs, and
    generates alpha energy spectrum plots for quality control.
    
    Prerequisites:
    - Run must be initialized (sets populated)
    - Raw alpha channel waveform files must exist
    
    Pipeline stages:
    1. create_energy_maps_in_run: Generate UID → energy binary maps
       - Creates energy_maps/SETNAME/ directories
       - Writes chunked binary files with caching
    2. plot_energy_spectra_in_run: Generate alpha energy spectra
       - One plot per set + one aggregated plot
       - Saved to plots/alpha_spectra/
    
    Args:
        run: Initialized Run object
        files_per_chunk: Waveform files per binary chunk (10-100 typical)
        fmt: Binary format - "8b" (accurate) or "6b" (compact)
        scale: For "6b" format: keV per LSB (default: 0.1)
        pattern: Glob pattern for alpha files (default: "*Ch4.wfm")
        savgol_window: Savitzky-Golay window size (default: 501 samples ≈ 100 ns)
                       Must be odd. Larger = more smoothing (21-5001 range)
        nbins: Number of histogram bins for energy spectra
        energy_range: (min, max) energy range [MeV] for histograms
        
    Returns:
        Unchanged Run (operation has file I/O side effects)
        
    Example:
        >>> from RaTag.pipelines import prepare_isotope_separation
        >>> run = initialize_run(my_run)
        >>> run = prepare_isotope_separation(run, files_per_chunk=10, savgol_window=501)
    """
    print("\n" + "="*60)
    print(f"ISOTOPE PREPARATION: {run.run_id}")
    print("="*60)
    
    # Pipeline stages
    steps = [
        partial(create_energy_maps_in_run,
                files_per_chunk=files_per_chunk,
                fmt=fmt,
                scale=scale,
                pattern=pattern,
                savgol_window=savgol_window),
        partial(plot_energy_spectra_in_run,
                nbins=nbins,
                energy_range=energy_range)
    ]
    
    result = pipe_run(run, *steps)
    
    print("\n" + "="*60)
    print("ISOTOPE PREPARATION COMPLETE")
    print("="*60)
    
    return result
