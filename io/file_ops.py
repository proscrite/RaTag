# RaTag/io/file_ops.py
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict, replace, fields

from RaTag.core.datatypes import SetPmt
from RaTag.core.dataIO import load_wfm
from RaTag.core.paths import get_output_root


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
    print(f"  Found {len(filenames)} .wfm files in {set_dir.name}: {filenames[:5]}{'...' if len(filenames) > 5 else ''}")
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
    
    # Pure 1:1 mapping. No type coercion, no bloat.
    update_kwargs = {k: v for k, v in data.items() if k in valid_keys}
            
    return replace(set_pmt, **update_kwargs)