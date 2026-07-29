import numpy as np
from pathlib import Path
import argparse
from RaTag.core.uid_utils import make_uid, parse_file_seq_from_name
from RaTag.io import file_ops
from RaTag.io.bootstrap import bootstrap_from_config

def generate_corrected_uids(filenames: list[str], frames_per_file: int) -> np.ndarray:
    """
    Pure operation to map the physical filenames to their deterministic FastFrame UIDs.
    No I/O or state changes occur here.
    """
    all_uids = []
    for fn in filenames:
        file_seq = parse_file_seq_from_name(fn)
        file_uids = [make_uid(file_seq, frame_idx) for frame_idx in range(frames_per_file)]
        all_uids.extend(file_uids)
        
    return np.array(all_uids, dtype=np.int64)

def patch_npz_uids(config_path: Path, signal_type: str):
    """
    Declarative API: Bootstraps the run, recalculates UIDs, and patches the archives.
    Limits Depth to 2: Evaluates constraints, orchestrates the operation, and handles I/O.
    """
    # 1. Explicitly bootstrap the physics configuration
    run = bootstrap_from_config(config_path)

    for set_alpha in run.alpha_sets:
        # 2. Skip sets that haven't been processed yet
        if not file_ops.check_npz_exists(set_alpha, signal_type):
            print(f"Skipping {set_alpha.source_dir.name}: NPZ missing.")
            continue
        frames_per_file = set_alpha.nframes
        # 3. Purely generate the correct UID array from the ordinally strict filename list
        corrected_uids = generate_corrected_uids(set_alpha.filenames, frames_per_file)
        
        # 4. Load the existing parallel arrays (e.g., alpha_energies)
        existing_arrays = file_ops.load_npz_arrays(set_alpha, signal_type)

        # 5. Strict Validation to prevent array misalignment 
        if len(corrected_uids) != len(existing_arrays['uids']):
            raise ValueError(
                f"Array size mismatch in {set_alpha.source_dir.name}: "
                f"Generated {len(corrected_uids)} UIDs for {len(existing_arrays['uids'])} entries."
            )

        # 6. Patch the dictionary and persist to disk
        existing_arrays['uids'] = corrected_uids
        saved_path = file_ops.save_npz_arrays(set_alpha, signal_type, existing_arrays)
        print(f"  → Patched {len(corrected_uids)} UIDs in: {saved_path.name}")

if __name__ == "__main__":
    # Explicit entry point passing parameters directly

    parser = argparse.ArgumentParser(description='RaTag UID Key Reset Utility')
    parser.add_argument('config', type=Path, help='Path to YAML config file')
    args = parser.parse_args()
    # Execute the key reset
    patch_npz_uids(
        config_path=args.config, 
        signal_type="alpha_energies"
    )