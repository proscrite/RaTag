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
from RaTag.workflows.energy_mapping import create_energy_maps_in_run


# ============================================================================
# PIPELINE - FUNCTIONAL COMPOSITION
# ============================================================================

def prepare_isotope_separation(run: Run,
                               files_per_chunk: int = 10,
                               fmt: str = "8b",
                               scale: float = 0.1,
                               pattern: str = "*Ch4.wfm") -> Run:
    """
    Prepare isotope separation by generating energy maps.
    
    This pipeline creates binary mapping files (energy_map_*.bin) that enable
    UID-to-energy lookups for isotope assignment in multi-isotope runs.
    
    Prerequisites:
    - Run must be initialized (sets populated)
    - Raw alpha channel waveform files must exist
    
    Pipeline stages:
    1. create_energy_maps_in_run: Generate UID → energy binary maps
       - Creates energy_maps/SETNAME/ directories
       - Writes chunked binary files with caching
    
    Args:
        run: Initialized Run object
        files_per_chunk: Waveform files per binary chunk (10-100 typical)
        fmt: Binary format - "8b" (accurate) or "6b" (compact)
        scale: For "6b" format: keV per LSB (default: 0.1)
        pattern: Glob pattern for alpha files (default: "*Ch4.wfm")
        
    Returns:
        Unchanged Run (operation has file I/O side effects)
        
    Example:
        >>> from RaTag.pipelines import prepare_isotope_separation
        >>> run = initialize_run(my_run)
        >>> run = prepare_isotope_separation(run, files_per_chunk=10)
    """
    print("\n" + "="*60)
    print(f"ISOTOPE PREPARATION: {run.run_id}")
    print("="*60)
    
    # Single stage: energy map generation with caching
    steps = [
        partial(create_energy_maps_in_run,
                files_per_chunk=files_per_chunk,
                fmt=fmt,
                scale=scale,
                pattern=pattern)
    ]
    
    result = pipe_run(run, *steps)
    
    print("\n" + "="*60)
    print("ISOTOPE PREPARATION COMPLETE")
    print("="*60)
    
    return result
