# RaTag/io/file_ops.py
import yaml
import json
import re
import numpy as np
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, TypeVar, Union
from dataclasses import asdict, replace, fields
from itertools import islice
from lmfit.model import ModelResult
PathLike = Union[str, Path]

from RaTag.core.decorators import track_iterator_progress
from RaTag.core.dataIO import load_wfm, load_alpha
from RaTag.core.paths import get_output_root
from RaTag.core.datatypes import PMTWaveform, SetAlpha, Waveform, SiliconWaveform, SetPmt, SetAlpha, Run, S2Areas
SetT = TypeVar("SetT", SetPmt, SetAlpha)

# --- Lazy loader ---
@track_iterator_progress
def iter_waveforms(set_pmt: SetPmt, max_files: Optional[int] = None,
                   show_progress: bool = True ) -> Iterator[PMTWaveform]:
    """Yield PMTWaveform objects lazily, one by one."""
    
    if max_files is not None:
        waveforms = islice(set_pmt.filenames, max_files)
    else:
        waveforms = set_pmt.filenames

    for fn in waveforms:
        yield load_wfm(Path(set_pmt.source_dir) / Path(fn))

@track_iterator_progress
def iter_alpha_waveforms(set_alpha: SetAlpha, max_files: Optional[int] = None,
                         show_progress: bool = True) -> Iterator[SiliconWaveform]:
    """Yields parsed SiliconWaveforms for the alpha detector channel."""
    if max_files:
        files_to_process = set_alpha.filenames[:max_files] 
    else:
        files_to_process = set_alpha.filenames
    for fn in files_to_process:
        yield load_alpha(set_alpha.source_dir / fn)

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

def iter_frames(set_pmt: SetPmt, max_frames: Optional[int] = None, start_frame: int = 0) -> Iterator[Waveform]:
    """
    Lazily yields individual frames from a set.
    Handles FastFrame by extracting and yielding each frame sequentially as a standalone object.
    
    Args:
        set_pmt: SetPmt to iterate over
        max_frames: Maximum number of frames to yield
        start_frame: Number of absolute frames to skip before yielding
    """
    frames_yielded = 0
    frames_skipped = 0
    
    for file_seq, fn in enumerate(set_pmt.filenames):
        wf = load_wfm(set_pmt.source_dir / fn)
        wf.file_seq = file_seq 
        
        if wf.ff and wf.nframes > 1:
            for frame_idx in range(wf.nframes):
                if frames_skipped < start_frame:
                    frames_skipped += 1
                    continue
                    
                if max_frames is not None and frames_yielded >= max_frames:
                    return
                    
                yield extract_single_frame(wf, frame_idx)
                frames_yielded += 1
        else:
            if frames_skipped < start_frame:
                frames_skipped += 1
                continue
                
            if max_frames is not None and frames_yielded >= max_frames:
                return
                
            yield wf
            frames_yielded += 1

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

def detect_multiiso_set(filenames: List[str], pattern: str = '_Ch3.wfm') -> bool:
    """
    Detects if the set is a multi-isotope set based on presence of Ch3 files.
    """
    return any(f.endswith(pattern) for f in filenames)

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


def _get_cache_path(set_obj: Union[SetPmt, SetAlpha]) -> Path:
    """Helper to centralize where the JSON lives."""
    metadata_dir = get_output_root(set_obj.source_dir.parent) / "set_summaries"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Route to different files based on the class of the object
    if isinstance(set_obj, SetAlpha):
        suffix = "_alphas_metadata.json"
    else:
        suffix = "_metadata.json"
    return metadata_dir / f"{set_obj.source_dir.name}{suffix}"


def save_cache(set_obj: Union[SetPmt, SetAlpha]) -> None:
    """
    Saves the current computed state to JSON.
    Relies on the object being fully hydrated beforehand to prevent data loss.
    """
    cache_path = _get_cache_path(set_obj)
    
    # Structural fields that belong to the raw scan, not the compute cache
    exclude = {"source_dir", "filenames", "multiiso"}
    
    # Filter out None values and excluded fields; round only numeric values
    patch_data = {}
    for k, v in asdict(set_obj).items():
        if k in exclude or v is None:
            continue
        # Only round floats/ints; preserve strings and other types as-is
        if isinstance(v, (int, float)):
            patch_data[k] = round(v, 3) if isinstance(v, float) else v
        else:
            patch_data[k] = v
    
    with open(cache_path, 'w') as f:
        json.dump(patch_data, f, indent=2)

def load_cache(set_obj: Union[SetPmt, SetAlpha]) -> Optional[Union[SetPmt, SetAlpha]]:
    """Loads computed fields from JSON cache if available and valid."""
    cache_path = _get_cache_path(set_obj)
    
    if not cache_path.exists():
        return None
        
    with open(cache_path, 'r') as f:
        data = json.load(f)
        
    valid_keys = {f.name for f in fields(set_obj)}
    
    # Pure 1:1 mapping.
    update_kwargs = {k: v for k, v in data.items() if k in valid_keys}
    
    if 'source_dir' in update_kwargs and isinstance(update_kwargs['source_dir'], str):
        update_kwargs['source_dir'] = Path(update_kwargs['source_dir'])
    
    return replace(set_obj, **update_kwargs)


# ============================================================================
#  .npz load/save for dense arrays (e.g. areas, timings)
# ============================================================================
def _get_npz_path(set_obj, signal_type: str) -> Path:
    """Centralized path routing for all NPZ files."""
    root = get_output_root(set_obj.source_dir.parent)
    
    # Routing for multi-isotope spawned sets (e.g., 'isotope_areas/Th228/')
    target_isotope = getattr(set_obj, 'target_isotope', None)
    if target_isotope:
        return root / "isotope_areas" / target_isotope / f"{set_obj.name}_{signal_type}.npz"
        
    return root / signal_type / f"{set_obj.name}_{signal_type}.npz"

def check_npz_exists(set_obj: Union[SetPmt, SetAlpha], signal_type: str) -> bool:
    """Returns True if the specified NPZ file exists."""
    return _get_npz_path(set_obj, signal_type).exists()

def save_npz_arrays(set_obj: Union[SetPmt, SetAlpha], signal_type: str, arrays: dict) -> Path:
    """Generic helper to save dense .npz arrays."""
    data_file = _get_npz_path(set_obj, signal_type)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(data_file, **arrays)
    return data_file

def save_run_npz_arrays(run: Run, signal_type: str, arrays: dict) -> Path:
    """Generic helper to save dense .npz arrays at the Run level."""
    out_dir = get_output_root(run.root_directory) / signal_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = out_dir / f"{run.run_id}_{signal_type}.npz"
    np.savez_compressed(data_file, **arrays)
    return data_file


def load_npz_arrays(set_obj: Union[SetPmt, SetAlpha], signal_type: str) -> dict:
    """Generic helper to cleanly load dense .npz arrays (e.g., 'timing', 's2_areas')."""
    data_file = _get_npz_path(set_obj, signal_type)
    print(f"  Loading {signal_type} arrays from: {data_file}")
    return dict(np.load(data_file)) if data_file.exists() else {}


def load_s2areas_from_path(file_path: Union[Path, str]) -> S2Areas:
    """Constructs an S2Areas transient arrays from an explicit file path."""
    arrays = dict(np.load(file_path))
    return S2Areas(
        uids=arrays.get("uids", np.array([])),
        areas=arrays.get("s2_areas", np.array([]))
    )

def load_s2areas(set_pmt: SetPmt) -> S2Areas:
    """Constructs an S2Areas transient arrays from a SetPmt's disk state."""
    arrays = load_npz_arrays(set_pmt, 's2_areas')
    return S2Areas(
        uids=arrays.get("uids", np.array([])),
        areas=arrays.get("s2_areas", np.array([]))
    )


def load_s1areas(set_pmt: SetPmt) -> S2Areas:
    """Constructs an S2Areas transient arrays from a SetPmt's disk state."""
    arrays = load_npz_arrays(set_pmt, 's2_areas')
    return S2Areas(
        uids=arrays.get("uids", np.array([])),
        areas=arrays.get("s1_areas", np.array([]))
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