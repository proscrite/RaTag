# workflows/run_construction.py

"""
Workflow for constructing and preparing Run objects.

Functions for populating sets, computing fields, and transport properties.
"""

from dataclasses import replace
from typing import Optional, cast

from core.datatypes import Run, SetPmt
from core.physics import with_gas_density
from core.constructors import populate_run as _populate_run, set_from_dir, set_fields, set_transport_properties
from core.dataIO import save_set_metadata, load_set_metadata
from core.functional import map_over


def add_gas_density(run: Run) -> Run:
    """Add gas density to run based on experimental parameters."""
    run = with_gas_density(run)
    print(f"Gas density: {run.gas_density:.3e} cm⁻³")
    return run


def populate_sets(run: Run, max_files: Optional[int] = None) -> Run:
    """
    Populate sets from run directory.
    
    Args:
        run: Run with root_directory set
        max_files: Optional limit on files per set (for testing)
        
    Returns:
        Run with sets populated
    """
    run = _populate_run(run)
    
    if max_files is not None:
        print(f"⚠ TEST MODE: Limiting to {max_files} files per set")
        limited_sets = [set_from_dir(s.source_dir, nfiles=max_files) for s in run.sets]
        run = replace(run, sets=limited_sets)
    
    print(f"Loaded {len(run.sets)} sets")
    return run


def compute_fields_and_transport(run: Run, 
                                  force_recompute: bool = False) -> Run:
    """
    Compute electric fields and transport properties for all sets.
    
    Args:
        run: Run with sets (each set must have t_s1 in metadata)
        force_recompute: Force recomputation even if cached
        
    Returns:
        Run with fields and transport properties set
    """
    # Ensure gas density is present before passing to set_fields (avoid Optional[float])
    if run.gas_density is None:
        run = with_gas_density(run)
        print(f"  ✓ Gas density: {run.gas_density:.3e} cm⁻³")

    def process_set(s: SetPmt) -> SetPmt:
        # Try to load from cache
        if not force_recompute:
            loaded = load_set_metadata(s)
            if loaded and loaded.time_drift is not None:
                return loaded
        
        # Compute fields
        s_with_fields = set_fields(
            s,
            drift_gap_cm=run.drift_gap,
            el_gap_cm=run.el_gap,
            gas_density=cast(float, run.gas_density)
        )
        
        # Compute transport properties
        s_with_transport = set_transport_properties(
            s_with_fields,
            drift_gap_cm=run.drift_gap,
            transport=None
        )
        
        return s_with_transport
    
    updated_sets = map_over(run.sets, process_set, catch_errors=True)
    return replace(run, sets=updated_sets)


def initialize_run(run: Run, max_files: Optional[int] = None) -> Run:
    """
    Initialize run with gas density, sets, fields, and transport properties.
    
    This aggregates the "free" operations that don't touch waveform data:
    1. Add gas density (from experimental parameters)
    2. Populate sets (parse directory structure)
    3. Compute fields and transport (from set voltages and gas properties)
    
    All three operations use cached values when available.
    
    Args:
        run: Run with root_directory and experiment parameters
        max_files: Optional limit on files per set (for testing)
        
    Returns:
        Run with sets populated and fields/transport computed
    """
    print("\n" + "="*60)
    print("RUN INITIALIZATION")
    print("="*60)
    
    # [1] Add gas density
    print("\n[1/3] Gas density...")
    run = with_gas_density(run)
    print(f"  ✓ Gas density: {run.gas_density:.3e} cm⁻³")
    
    # [2] Populate sets
    print("\n[2/3] Populating sets...")
    run = _populate_run(run)
    
    if max_files is not None:
        print(f"  ⚠ TEST MODE: Limiting to {max_files} files per set")
        limited_sets = [set_from_dir(s.source_dir, nfiles=max_files) for s in run.sets]
        run = replace(run, sets=limited_sets)
    
    print(f"  ✓ Loaded {len(run.sets)} sets")
    
    # [3] Compute fields and transport
    print("\n[3/3] Computing fields and transport properties...")
    
    def process_set(s: SetPmt) -> SetPmt:
        # Check if metadata file exists on disk
        metadata_file = s.source_dir.parent / "processed_data" / ".metadata.json"
        
        if metadata_file.exists():
            # Try to load from cache
            loaded = load_set_metadata(s)
            if loaded and loaded.time_drift is not None:
                print(f"  📂 {s.source_dir.name}: Loaded from cache")
                return loaded
        
        # Compute fields
        s_with_fields = set_fields(
            s,
            drift_gap_cm=run.drift_gap,
            el_gap_cm=run.el_gap,
            gas_density=run.gas_density
        )
        
        # Compute transport properties
        s_with_transport = set_transport_properties(
            s_with_fields,
            drift_gap_cm=run.drift_gap,
            transport=None
        )
        
        # Save to cache immediately!
        save_set_metadata(s_with_transport)
        print(f"  ✓ {s.source_dir.name}: Computed and saved to cache")
        
        return s_with_transport
    
    updated_sets = map_over(run.sets, process_set, catch_errors=True)
    run = replace(run, sets=updated_sets)
    
    print("\n✓ Run initialization complete")
    return run