from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

from RaTag.core.datatypes import SetAlpha, SetPmt

if TYPE_CHECKING:
    from RaTag.core.datatypes import Run



def get_processed_root_from_env() -> Optional[Path]:
    """Return processed root from `RATAG_PROCESSED_ROOT` env var if set."""
    p = os.environ.get("RATAG_PROCESSED_ROOT")
    return Path(p) if p else None


def get_processed_run_dir(run: Run | Path) -> Path:
    """Return the directory where processed artifacts for a run should live.

    Priority:
    1. `RATAG_PROCESSED_ROOT` env var -> <root>/<run_id>
    2. Fallback: use run.root_directory.parents[1] / 'processed' -> <data_root>/processed/<run_id>

    This keeps raw data locations untouched and centralises processed outputs.
    """
    # Allow passing a Path (legacy call-sites) or a Run
    if isinstance(run, Path):
        run_root = run
        run_id = run.name
    else:
        run_root = run.root_directory
        run_id = run.run_id

    env_root = get_processed_root_from_env()
    if env_root:
        return env_root / run_id

    # Fallback: pick parent.parent + /processed
    parents = list(run_root.parents)
    if len(parents) >= 2:
        data_root = parents[1]
    else:
        data_root = run_root.parent

    return data_root / "processed" / run_id


def get_output_root(run: Run | Path) -> Path:
    """Return preferred output root for a run.

    Priority:
    Behaviour:
    - If the run path contains a raw-data marker (e.g. `raw_waveforms` or `raw_data`),
      swap that segment for `processed` and return the resulting path.
    - Otherwise fall back to the previous parent/processed/<same-folder-name> layout.
    """
    # Determine a path object from input
    if isinstance(run, Path):
        p = run
    else:
        p = getattr(run, 'root_directory', None)
        if p is None:
            # Last resort: delegate to get_processed_run_dir behaviour
            return get_processed_run_dir(run)

    s = str(p)
    # Common raw-data directory tokens to replace
    tokens = ["/raw_waveforms/", "/raw_data/", "/raw/"]
    for t in tokens:
        if t in s:
            return Path(s.replace(t, "/processed/"))

    # Also handle cases without both-side slashes
    tokens_simple = ["raw_waveforms", "raw_data", "raw"]
    for t in tokens_simple:
        if t in s:
            return Path(s.replace(t, "processed"))

    # Fallback: keep parent of run and place a processed/<same-folder-name> there
    parents = list(p.parents)
    if len(parents) >= 2:
        data_root = parents[1]
    else:
        data_root = p.parent

    return data_root / "processed" / p.name

def get_cache_path(set_obj: Union[SetPmt, SetAlpha]) -> Path:
    """Helper to centralize where the JSON lives. Routes to isotope subfolder if tagged."""
    metadata_dir = get_output_root(set_obj.source_dir.parent) / "set_summaries"

    target_isotope = getattr(set_obj, 'target_isotope', None)
    if target_isotope:
        metadata_dir = metadata_dir / target_isotope

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Route to different files based on the class of the object
    if isinstance(set_obj, SetAlpha):
        suffix = "_alphas_metadata.json"
    else:
        suffix = "_metadata.json"
    return metadata_dir / f"{set_obj.name}{suffix}"

def get_npz_path(set_obj, signal_type: str) -> Path:
    """Centralized path routing for all NPZ files."""
    root = get_output_root(set_obj.source_dir.parent)
    
    # Routing for multi-isotope spawned sets (e.g., 'isotope_areas/Th228/')
    target_isotope = getattr(set_obj, 'target_isotope', None)
    if target_isotope:
        return root / "isotope_areas" / target_isotope / f"{set_obj.name}_{signal_type}.npz"
        
    return root / signal_type / f"{set_obj.name}_{signal_type}.npz"

def get_fit_path(set_pmt, suffix: str) -> Path:
    out_dir = get_output_root(set_pmt.source_dir.parent) / "fits"
    target_isotope = getattr(set_pmt, 'target_isotope', None)
    if target_isotope:
        out_dir = out_dir / target_isotope
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{set_pmt.name}_{suffix}.json"


def get_plot_dir(obj, subfolder: str) -> Path:
    is_run = hasattr(obj, 'run_id')
    root = get_output_root(obj) if is_run else get_output_root(obj.source_dir.parent)
    out_dir = root / "plots" / subfolder
    
    target_isotope = getattr(obj, 'target_isotope', None)
    if target_isotope:
        out_dir = out_dir / target_isotope
        
    return out_dir