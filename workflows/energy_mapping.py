"""
Energy mapping workflow - Generate UID-to-energy binary maps for isotope separation.

This module provides workflows for creating energy map files that enable
multi-isotope analysis by mapping unique identifiers (UIDs) to alpha energies.

Structure mirrors timing_estimation.py and recoil_integration.py:
- Set-level workflow handles complete operation with side effects
- Run-level orchestration with caching
"""

from pathlib import Path
from typing import Optional
from dataclasses import replace

from RaTag.core.datatypes import SetPmt, Run
from RaTag.core.functional import apply_workflow_to_run
from RaTag.alphas.energy_map_writer import writer_main


# ============================================================================
# SET-LEVEL WORKFLOW (Complete with side effects)
# ============================================================================

def generate_energy_maps_for_set(set_pmt: SetPmt,
                                 files_per_chunk: int = 10,
                                 fmt: str = "8b",
                                 scale: float = 0.1,
                                 pattern: str = "*Ch4.wfm") -> SetPmt:
    """
    Generate energy map files for a single set.
    
    Creates binary chunk files mapping UIDs to alpha energies, stored in
    energy_maps/SETNAME/ directory alongside plots and processed_data.
    
    Args:
        set_pmt: SetPmt object with source_dir pointing to raw waveform files
        files_per_chunk: Number of waveform files per chunk (default: 10)
                        Typical values: 10-100 depending on file size
        fmt: Binary format - "8b" (uint32+float32) or "6b" (uint32+uint16 scaled)
        scale: For "6b" format, keV per LSB (default: 0.1)
        pattern: Glob pattern for alpha channel files (default: "*Ch4.wfm")
        
    Returns:
        Unchanged SetPmt (operation has file I/O side effects)
        
    Side Effects:
        Creates energy_maps/SETNAME/energy_map_f{start:06d}-f{end:06d}.bin files
        
    Example:
        >>> set_pmt = generate_energy_maps_for_set(set_pmt, files_per_chunk=10)
        # Creates: energy_maps/FieldScan_Gate0050_Anode1950/energy_map_*.bin
    """
    # Setup output directory structure
    energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
    energy_maps_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate energy maps
    writer_main(indir=str(set_pmt.source_dir),
                outdir=str(energy_maps_dir),
                files_per_chunk=files_per_chunk,
                fmt=fmt,
                scale=scale,
                pattern=pattern)
    
    print(f"  ✓ Generated energy maps")
    
    return set_pmt


# ============================================================================
# RUN-LEVEL ORCHESTRATION (with caching)
# ============================================================================

def create_energy_maps_in_run(run: Run,
                              files_per_chunk: int = 10,
                              fmt: str = "8b",
                              scale: float = 0.1,
                              pattern: str = "*Ch4.wfm") -> Run:
    """
    Generate energy mapping files for all sets in a run (with caching).
    
    This function creates binary chunk files that map unique identifiers (UIDs)
    to alpha particle energies, enabling isotope separation in multi-isotope runs.
    
    Caching: Checks for existing energy map files before regenerating.
    
    Prerequisites:
    - Run must be initialized (sets populated from directory)
    - Raw waveform files must exist in each set's source_dir
    
    Output Structure:
        energy_maps/
        ├── SETNAME_1/
        │   ├── energy_map_f000001-f000010.bin
        │   └── ...
        └── SETNAME_2/
    
    Args:
        run: Run object with populated sets
        files_per_chunk: Number of waveform files per chunk (10-100 typical)
        fmt: Binary format - "8b" (accurate) or "6b" (compact)
        scale: For "6b" format only: keV per LSB (default: 0.1)
        pattern: Glob pattern for alpha channel files (default: "*Ch4.wfm")
        
    Returns:
        Unchanged Run (operation has file I/O side effects)
        
    Example:
        >>> run = create_energy_maps_in_run(run, files_per_chunk=10)
    """
    # Custom cache check: look for energy_map_*.bin files
    def _check_energy_maps_exist(set_pmt: SetPmt) -> bool:
        """Check if energy maps already exist for this set."""
        energy_maps_dir = set_pmt.source_dir.parent / "energy_maps" / set_pmt.source_dir.name
        if not energy_maps_dir.exists():
            return False
        # Check if any .bin files exist
        bin_files = list(energy_maps_dir.glob("energy_map_*.bin"))
        return len(bin_files) > 0
    
    print("\n" + "="*60)
    print("ENERGY MAPPING GENERATION")
    print("="*60)
    print(f"Format: {fmt}, Files per chunk: {files_per_chunk}")
    if fmt == "6b":
        print(f"Scale: {scale} keV/LSB")
    print(f"Pattern: {pattern}")
    
    updated_sets = []
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check cache
        if _check_energy_maps_exist(set_pmt):
            print(f"  📂 Energy maps already exist (skipping)")
            updated_sets.append(set_pmt)
            continue
        
        try:
            updated_set = generate_energy_maps_for_set(set_pmt,
                                                       files_per_chunk=files_per_chunk,
                                                       fmt=fmt,
                                                       scale=scale,
                                                       pattern=pattern)
            updated_sets.append(updated_set)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            # import traceback
            # traceback.print_exc()
            updated_sets.append(set_pmt)
    
    print("\n✓ Energy mapping complete")
    return replace(run, sets=updated_sets)
