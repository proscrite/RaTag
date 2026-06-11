# RaTag/io/file_ops.py
import yaml
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional, Union
from dataclasses import asdict, replace, fields
from functools import lru_cache
from itertools import islice
from lmfit.model import ModelResult
PathLike = Union[str, Path]

from RaTag.core.dataIO import load_wfm
from RaTag.core.paths import get_output_root
from RaTag.core.datatypes import PMTWaveform, Waveform, SetPmt, Run, S2Areas

# --- Lazy loader ---
def iter_waveforms(set_pmt: SetPmt, max_files: int = None) -> Iterator[PMTWaveform]:
    """Yield PMTWaveform objects lazily, one by one."""
    
    if max_files is not None:
            waveforms = islice(set_pmt.filenames, max_files)
    else:
        waveforms = set_pmt.filenames

    for fn in waveforms:
        yield load_wfm(Path(set_pmt.source_dir) / Path(fn))


# --- Extract single waveform from FastFrame ---
def extract_single_frame(wf: Waveform, frame_idx: int) -> Waveform:
    """Extract a single frame from a FastFrame waveform."""
    
    v_single = wf.v[frame_idx, :]
    return Waveform(t=wf.t, v=v_single,
                    source=wf.source, file_seq=wf.file_seq, frame_idx=frame_idx, 
                    ff=False, nframes=1)


def load_random_waveform(set_pmt: SetPmt) -> Tuple[PMTWaveform, Optional[int]]:
    """
    Safely selects and loads a random waveform from a set for validation.
    Handles FastFrame geometry internally.
    """
    if not set_pmt.filenames:
        raise ValueError(f"No files available in set {set_pmt.source_dir.name}.")
        
    fn = np.random.choice(set_pmt.filenames)
    wf = load_wfm(set_pmt.source_dir / fn)
    
    frame = np.random.randint(0, set_pmt.nframes) if set_pmt.ff else None
    
    return wf, frame

def load_waveform_by_uid(set_pmt: SetPmt, uid: int) -> Tuple[PMTWaveform, Optional[int]]:
    """Resolves a UID to its physical file and frame, and loads the PMTWaveform."""
    from RaTag.core.uid_utils import decode_uid, parse_file_seq_from_name
    
    file_seq, frame_idx = decode_uid(uid)
    
    # Find the corresponding filename in the set
    target_fn = next((fn for fn in set_pmt.filenames if parse_file_seq_from_name(fn) == file_seq), None)
    if not target_fn:
        raise ValueError(f"File sequence {file_seq} not found in set {set_pmt.source_dir.name}")
        
    wf = load_wfm(set_pmt.source_dir / target_fn)
    return wf, frame_idx if set_pmt.ff else None

def iter_frames(set_pmt: SetPmt, max_files: int = None) -> Iterator[Waveform]:
    """
    Iterate over individual frames from a set, handling both FastFrame and single-frame.
    
    This is the canonical way to iterate over frames in the codebase.
    All analysis functions should use this to ensure consistency.
    
    Args:
        set_pmt: SetPmt to iterate over
        max_files: Optional limit on number of files to process
        
    Yields:
        Individual PMTWaveform objects (with ff=False)
    """
    waveforms = iter_waveforms(set_pmt)
    
    if max_files is not None:
        waveforms = islice(waveforms, max_files)
    
    for file_seq, wf in enumerate(waveforms):
        wf.file_seq = file_seq  # annotate with file sequence for UID calculation
        if wf.ff and wf.nframes > 1:
            # FastFrame: yield each frame individually
            for frame_idx in range(wf.nframes):
                yield extract_single_frame(wf, frame_idx)
        else:
            # Single frame: yield as-is
            yield wf

def load_yaml(config_path: Path) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def scan_for_set_directories(run_dir: Path) -> List[Path]:
    """
    Scans a run directory for valid measurement sets.
    Extracts the exact logic previously in `populate_run`.
    """
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and 'FieldScan' in d.name]
    # Return a list of directory_paths
    return sorted(subdirs)


def parse_run_id(run_dir: Path) -> Optional[str]:
    """
    Extracts run ID from the run directory name using regex patterns.
    Example: if the directory is named "RUN8_Th228_2375Vcm", it will extract "RUN8".
    """
    if m := re.search(r"Run(\d+)", run_dir.name, re.IGNORECASE):
        return f"RUN{m.group(1)}"
    if m := re.search(r"RUN_(\d+)", run_dir.name, re.IGNORECASE):
        return f"RUN{m.group(1)}"

def parse_target_isotope(run_dir: Path) -> Optional[str]:
    """
    Extracts target isotope from the run directory name using regex patterns.
    Example: if the directory is named "RUN8_Th228_2375Vcm", it will extract "Th228".
    """
    if m := re.search(r"_([a-zA-Z]{2}\d+)", run_dir.name):
        return m.group(1)

def parse_el_field(run_dir: Path) -> Optional[int]:
    """
    Extracts electron field from the run directory name using regex patterns.
    Example: if the directory is named "RUN8_Th228_2375Vcm", it will extract 2375.
    """
    if m := re.search(r"_(\d+)Vcm", run_dir.name):
        return int(m.group(1))

def parse_subdir_name(name: str) -> dict:
    """
    Extract acquisition parameters from subdir name.
    Handles inconsistent patterns (Anode vs EL).
    """
    out = {}
    if m := re.search(r"(\d+)GSsec", name):
        out["sampling_rate"] = int(m.group(1)) * 1e9
    if m := re.search(r"Anode(\d+)", name):
        out["anode"] = int(m.group(1))
    elif m := re.search(r"EL(\d+)", name):
        out["anode"] = int(m.group(1))   # treat EL as Anode synonym
    if m := re.search(r"Gate(\d+)", name):
        out["gate"] = int(m.group(1))
    return out


def parse_filename(fname: str) -> dict:
    """
    Extract run/date/gate/anode/event_id/channel from filename.
    """
    out = {}
    if m := re.search(r"RUN(\d+)", fname):
        out["run"] = int(m.group(1))
    if m := re.search(r"_(\d{6,8})_", fname):
        out["date"] = m.group(1)  # keep raw string, could be ddmmyyyy or yyyymmdd
    if m := re.search(r"Gate(\d+)", fname):
        out["gate"] = int(m.group(1))
    if m := re.search(r"(?:Anode|EL)(\d+)", fname):
        out["anode"] = int(m.group(1))
    if m := re.search(r"P(\d+)", fname):
        out["position"] = int(m.group(1))
    if m := re.search(r"_(\d+)(?:_ch(\d+))?\.wfm$", fname):
        out["event_id"] = int(m.group(1))
        if m.group(2):
            out["channel"] = int(m.group(2))
    return out

def find_set_files(set_dir: Path, nfiles: Optional[int] = None) -> List[str]:
    """Finds .wfm files in a set directory, optionally limiting to nfiles."""
    wfm_files = sorted(set_dir.glob("*.wfm"))
    if nfiles is not None:
        wfm_files = wfm_files[:nfiles]
    
    filenames = [f.name for f in wfm_files]
    print(f"  Found {len(filenames)} .wfm files in {set_dir.name}")
    return filenames

def detect_multiiso_set(filenames: List[str]) -> bool:
    """
    Detects if the set is a multi-isotope set based on presence of Ch1 files.
    """
    return any(f.endswith("_Ch1.wfm") for f in filenames)

def detect_fastframe_properties(set_dir: Path, filenames: List[str]) -> Tuple[bool, int]:
    """
    Detects if the set uses fast frame acquisition and extracts properties.
    """
    first_wf = load_wfm(set_dir / filenames[0])
    ff = first_wf.ff
    nframes = first_wf.nframes if ff else 1
    
    return ff, nframes

# ============================================================================
# JSON CACHE MANAGEMENT (for computed attributes )
# ===========================================================================


def _get_cache_path(set_pmt: SetPmt) -> Path:
    """Helper to centralize where the JSON lives."""
    metadata_dir = get_output_root(set_pmt.source_dir.parent) / "set_summaries"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f"{set_pmt.source_dir.name}_metadata.json"


def save_cache(set_pmt: SetPmt) -> None:
    """
    STRICT OVERWRITE: Saves the current computed state to JSON.
    Relies on the object being fully hydrated beforehand to prevent data loss.
    """
    cache_path = _get_cache_path(set_pmt)
    
    # Structural fields that belong to the raw scan, not the compute cache
    exclude = {"source_dir", "filenames", "multiiso", "ff", "nframes"}
    
    # Filter out None values and excluded fields; round only numeric values
    patch_data = {}
    for k, v in asdict(set_pmt).items():
        if k in exclude or v is None:
            continue
        # Only round floats/ints; preserve strings and other types as-is
        if isinstance(v, (int, float)):
            patch_data[k] = round(v, 3) if isinstance(v, float) else v
        else:
            patch_data[k] = v
    
    with open(cache_path, 'w') as f:
        json.dump(patch_data, f, indent=2)

def load_cache(set_pmt: SetPmt) -> Optional[SetPmt]:
    """Loads computed fields from JSON cache if available and valid."""
    cache_path = _get_cache_path(set_pmt)
    
    if not cache_path.exists():
        return None
        
    with open(cache_path, 'r') as f:
        data = json.load(f)
        
    valid_keys = {f.name for f in fields(SetPmt)}
    
    # Pure 1:1 mapping.
    update_kwargs = {k: v for k, v in data.items() if k in valid_keys}
    
    if 'source_dir' in update_kwargs and isinstance(update_kwargs['source_dir'], str):
        update_kwargs['source_dir'] = Path(update_kwargs['source_dir'])
    
    return replace(set_pmt, **update_kwargs)


# ============================================================================
#  .npz load/save for dense payloads (e.g. areas, timings)
# ============================================================================
def save_npz_payload(set_pmt: SetPmt, signal_type: str, payload: dict) -> Path:
    """Generic helper to save dense .npz payloads."""
    out_dir = get_output_root(set_pmt.source_dir.parent) / signal_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = out_dir / f"{set_pmt.source_dir.name}_{signal_type}.npz"
    np.savez_compressed(data_file, **payload)
    return data_file

def save_run_npz_payload(run: Run, signal_type: str, payload: dict) -> Path:
    """Generic helper to save dense .npz payloads at the Run level."""
    out_dir = get_output_root(run.root_directory) / signal_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = out_dir / f"{run.run_id}_{signal_type}.npz"
    np.savez_compressed(data_file, **payload)
    return data_file


def load_npz_payload(set_pmt: SetPmt, signal_type: str) -> dict:
    """Generic helper to cleanly load dense .npz payloads (e.g., 'timing', 's2_areas')."""
    data_file = get_output_root(set_pmt.source_dir.parent) / f"{signal_type}" / f"{set_pmt.source_dir.name}_{signal_type}.npz"
    return dict(np.load(data_file)) if data_file.exists() else {}


def load_s2areas_from_path(file_path: Union[Path, str]) -> S2Areas:
    """Constructs an S2Areas transient payload from an explicit file path."""
    payload = dict(np.load(file_path))
    return S2Areas(
        uids=payload.get("uids", np.array([])),
        areas=payload.get("s2_areas", np.array([]))
    )

def load_s2areas(set_pmt: SetPmt) -> S2Areas:
    """Constructs an S2Areas transient payload from a SetPmt's disk state."""
    payload = load_npz_payload(set_pmt, 's2_areas')
    return S2Areas(
        uids=payload.get("uids", np.array([])),
        areas=payload.get("s2_areas", np.array([]))
    )
# ============================================================================
#  Plot saving helper (for consistent naming and RAM management)
# ============================================================================

def save_figure(fig, filename: PathLike, dpi: int = 150) -> None:
    """
    Save matplotlib figure to disk.
    
    Args:
        fig: Matplotlib figure
        filename: Output path
        dpi: Resolution for raster formats
    """
    import matplotlib.pyplot as plt
    
    filename = Path(filename)
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {filename}")

# ============================================================================
#  Load/save fitting results (e.g., from lmfit)
# ============================================================================


def save_fit_result(result: ModelResult, output_path: Union[str, Path]) -> Path:
    """
    Saves an lmfit ModelResult to a human-readable JSON file.
    
    Args:
        result: The output from an lmfit Model.fit()
        output_path: Destination path (e.g., 'processed/run_25/fits/alpha_1.json')
        
    Returns:
        Path object to the saved file.
    """
    output_path = Path(output_path)
    
    # Ensure the target directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # result.dumps() serializes the model, parameters, bounds, and fit statistics
    json_data = result.dumps()
    
    with open(output_path, 'w') as f:
        f.write(json_data)
        
    return output_path

def load_fit_result(input_path: Union[str, Path], **kwargs): # -> Optional[ModelResult]
    """
    Loads an lmfit ModelResult from a JSON file for downstream plotting or analysis.
    Warns and returns None if the file is missing or corrupted.
    """
    from lmfit.model import load_modelresult
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"  ⚠ Warning: Fit result file not found: {input_path.name}")
        return None
        
    try:
        return load_modelresult(str(input_path), **kwargs)
    except Exception as e:
        print(f"  ⚠ Warning: Failed to load fit result from {input_path.name}: {e}")
        return None