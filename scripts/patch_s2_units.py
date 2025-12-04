#!/usr/bin/env python3
"""
Post-mortem patch: Fix S2 area units in existing NPZ files.

This script corrects S2 area values that were integrated with incorrect dt value.
The bug was: dt = 0.2 µs in config (wrong) vs dt = 0.0002 µs (correct for 5 GS/s).
This resulted in areas being 1000× too large.

Usage:
    python scripts/patch_s2_units.py /path/to/processed_data
    
The script will:
1. Find all s2_areas.npz files recursively
2. Back up each file to .npz.backup (only if backup doesn't exist)
3. Apply correction factor of 1/1000 to areas
4. Save corrected version back to original filename

Example:
    python scripts/patch_s2_units.py /Volumes/KINGSTON/RaTag_data/RUN18/processed_data
"""

import argparse
from pathlib import Path
import numpy as np
import shutil


def fix_s2_areas(data_dir: Path, correction_factor: float = 1/1000, dry_run: bool = False):
    """
    Fix S2 area NPZ files by applying correction factor.
    
    Args:
        data_dir: Directory containing s2_areas.npz files (searches recursively)
        correction_factor: Multiply areas by this (default 1/1000 for dt fix)
        dry_run: If True, only show what would be done without modifying files
    """
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    # Find all s2_areas.npz files
    npz_files = list(data_dir.glob("**/*s2_areas.npz"))
    
    if len(npz_files) == 0:
        print(f"⚠️  No s2_areas.npz files found in {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"S2 AREA UNITS PATCH")
    print(f"{'='*60}")
    print(f"Found {len(npz_files)} S2 area files to patch")
    print(f"Correction factor: {correction_factor}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print(f"{'='*60}\n")
    
    total_corrected = 0
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"[{i}/{len(npz_files)}] Processing: {npz_file.name}")
        
        try:
            # Load data
            data = np.load(npz_file)
            areas = data['s2_areas']
            uids = data['uids']
            
            # Calculate statistics
            old_mean = areas.mean()
            old_median = np.median(areas)
            corrected_areas = areas * correction_factor
            new_mean = corrected_areas.mean()
            new_median = np.median(corrected_areas)
            
            print(f"  Original: mean={old_mean:.2f}, median={old_median:.2f}")
            print(f"  Corrected: mean={new_mean:.4f}, median={new_median:.4f}")
            
            if not dry_run:
                # Backup original (only if backup doesn't exist)
                backup = npz_file.with_suffix('.npz.backup')
                if not backup.exists():
                    shutil.copy(npz_file, backup)
                    print(f"  💾 Backed up to: {backup.name}")
                else:
                    print(f"  ℹ️  Backup already exists: {backup.name}")
                
                # Save corrected version
                np.savez_compressed(npz_file, s2_areas=corrected_areas, uids=uids)
                print(f"  ✅ Saved corrected areas")
                total_corrected += 1
            else:
                print(f"  [DRY RUN] Would save corrected version")
            
        except Exception as e:
            print(f"  ❌ Error processing {npz_file.name}: {e}")
            continue
        
        print()
    
    print(f"{'='*60}")
    if dry_run:
        print(f"DRY RUN COMPLETE - No files were modified")
        print(f"Run without --dry-run to apply changes")
    else:
        print(f"PATCH COMPLETE")
        print(f"Successfully corrected {total_corrected}/{len(npz_files)} files")
        print(f"Original files backed up with .backup extension")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Fix S2 area units in NPZ files (dt misconfiguration patch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('data_dir', type=Path,
                       help='Directory containing s2_areas.npz files (searches recursively)')
    parser.add_argument('--correction-factor', type=float, default=1/1000,
                       help='Correction factor to apply (default: 1/1000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without modifying files')
    
    args = parser.parse_args()
    
    fix_s2_areas(args.data_dir, 
                 correction_factor=args.correction_factor,
                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
