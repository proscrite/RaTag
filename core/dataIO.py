import numpy as np 
import pandas as pd
from pathlib import Path
from typing import Union, Iterator, Optional
import re
import json
import itertools
from dataclasses import replace, asdict, fields

import lmfit
from lmfit.model import ModelResult

from .units import V_to_mV, s_to_us
from .datatypes import SiliconWaveform, Waveform, SetPmt, Run, S2Areas, PMTWaveform
from .wfm2read_fast import wfm2read # type: ignore
PathLike = Union[str, Path]
from RaTag.core.uid_utils import parse_file_seq_from_name
from RaTag.core.paths import get_processed_run_dir, get_output_root

# -------------------------------------
# --- Load waveform from .wfm file  ---
# -------------------------------------

def _load_wfm_V_s(path: PathLike) -> PMTWaveform:
    """Load waveform from a .wfm file storing (t, v)."""
    wfm = wfm2read(str(path))
    t, v = wfm[1], wfm[0]
    v = -v  # Invert signal polarity

    file_seq = parse_file_seq_from_name(str(path))
    if len(v.shape) > 1:  # FastFrame format
        ff = True
        nframes = v.shape[0]
    else:
        ff = False
        nframes = 1
    return PMTWaveform(t, v, source=str(path), ff=ff, nframes=nframes, file_seq=file_seq)

def load_wfm(path: PathLike) -> PMTWaveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wf = _load_wfm_V_s(path)
    t_s, v_V = wf.t, wf.v

    v_mV = V_to_mV(v_V)
    t_us = s_to_us(t_s)

    return PMTWaveform(t=t_us, v=v_mV, source=wf.source, ff=wf.ff, nframes=wf.nframes, file_seq=wf.file_seq)


def load_alpha(path: PathLike) -> SiliconWaveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wf = _load_wfm_V_s(path)
    t_s, v_V = wf.t, -wf.v

    t_us = s_to_us(t_s)

    return SiliconWaveform(t=t_us, v=v_V, source=wf.source, ff=wf.ff, nframes=wf.nframes, file_seq=wf.file_seq)

# --- Lazy loader ---
def iter_waveforms(set_pmt: SetPmt) -> Iterator[PMTWaveform]:
    """Yield PMTWaveform objects lazily, one by one."""
    
    for fn in set_pmt.filenames:
        yield load_wfm(set_pmt.source_dir / fn)


# --- Extract single waveform from FastFrame ---
def extract_single_frame(wf: Waveform, frame: int = 0) -> Waveform:
    """Extract a single frame from a FastFrame waveform."""
    if not wf.ff:
        raise ValueError("Waveform is not FastFrame format")
    if frame < 0 or frame >= wf.nframes:
        raise ValueError(f"Frame index {frame} out of range [0, {wf.nframes})")
    v_single = wf.v[frame, :]
    return Waveform(t=wf.t, v=v_single, source=wf.source, ff=False, nframes=1)


def iter_frames(set_pmt, max_files: int = None) -> Iterator[Waveform]:
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
        waveforms = itertools.islice(waveforms, max_files)
    
    for wf in waveforms:
        if wf.ff and wf.nframes > 1:
            # FastFrame: yield each frame individually
            for frame_idx in range(wf.nframes):
                yield extract_single_frame(wf, frame_idx)
        else:
            # Single frame: yield as-is
            yield wf

# ----------------------------------------
# --- Subdirectory parsers for set constructions  ---
# -------------------------------------

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

# ----------------------------------------
# --- Set metadata IO:     ---------------
# --- store transport properties, s1, s2...
# ----------------------------------------

# core/dataIO.py - Add simple save/load functions

def save_set_metadata(set_pmt: SetPmt) -> None:
    """
    Save complete set metadata to JSON file in processed_data directory.
    
    Merges with existing metadata on disk to preserve data from multiple workflows.
    Only saves non-None values to avoid polluting cache with incomplete data.
    
    File location: {run_dir}/processed_data/{set_name}_metadata.json
    """
    metadata_dir = get_output_root(set_pmt.source_dir.parent) / "set_summaries"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = metadata_dir / f"{set_pmt.source_dir.name}_metadata.json"
    
    # Load existing metadata if it exists
    existing_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
    
    # Start with existing metadata, then update with new values
    metadata = {
        **existing_metadata,  # Keep existing data
        "set_name": set_pmt.source_dir.name,
        "source_dir": str(set_pmt.source_dir),
    }

    exclude_fields = {"source_dir", "filenames", "multiiso", "ff", "nframes"}
    set_dict = asdict(set_pmt)

    # Filter out None values and formatting
    for key, value in set_dict.items():
        if key in exclude_fields:
            continue
        if value is not None:
            if isinstance(value, float):
                metadata[key] = round(value, 3)
            elif isinstance(value, dict):
                # XRayMetadata converts to a dict containing subfields via asdict
                metadata[key] = {k: (round(v, 3) if isinstance(v, float) else v) 
                                 for k, v in value.items() if v is not None}
            elif isinstance(value, tuple) or isinstance(value, list):
                # E.g., area_s2_ci95 tuple
                metadata[key] = [(round(v, 3) if isinstance(v, float) else v) for v in value]
            else:
                metadata[key] = value
                
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_set_metadata(set_pmt: SetPmt) -> Optional[SetPmt]:
    """
    Load set metadata from JSON if it exists.
    
    File location: {run_dir}/processed_data/{set_name}_metadata.json
    
    Returns:
        Updated SetPmt with loaded metadata, or None if file doesn't exist
    """
    from RaTag.core.datatypes import XRayMetadata
    
    # Look in central processed run directory
    metadata_dir = get_output_root(set_pmt.source_dir.parent) / "set_summaries"
    metadata_file = metadata_dir / f"{set_pmt.source_dir.name}_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    valid_keys = {f.name for f in fields(SetPmt)}
    exclude_fields = {"source_dir", "filenames", "multiiso", "ff", "nframes"}
    
    update_kwargs = {}
    for k, v in data.items():
        if k in valid_keys and k not in exclude_fields:
            if k == "xray_metadata" and v is not None:
                update_kwargs[k] = XRayMetadata(**v)
            elif v is not None:
                if isinstance(v, list):
                    # Coerce iterables back to tuples if needed, e.g. for ci95
                    v = tuple(v)
                update_kwargs[k] = v
    
    return replace(set_pmt, **update_kwargs)

# ----------------------------------------
# --- Run metadata storing           -----
# ----------------------------------------

def save_run_metadata(run: Run) -> None:
    """
    Save run-level metadata to JSON file.
    
    File location: {run_dir}/metadata/run_info.json
    """
    
    metadata_dir = run.root_directory / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = metadata_dir / "run_info.json"
    
    # Collect run-level info
    metadata = {
        "run_id": run.run_id,
        "target_isotope": run.target_isotope,
        "pressure": run.pressure,
        "temperature": run.temperature,
        "sampling_rate": run.sampling_rate,
        "drift_gap": run.drift_gap,
        "el_gap": run.el_gap,
        "el_field": run.el_field,
        "gas_density": run.gas_density,
        "W_value": run.W_value,
        "E_gamma_xray": run.E_gamma_xray,
        
        # Set summaries
        "n_sets": len(run.sets),
        "sets": [
            {
                "name": s.source_dir.name,
                "v_gate": s.gate,
                "v_anode": s.anode,
                "drift_field": s.drift_field,
                "time_drift": s.time_drift,
                "n_waveforms": len(s.filenames) if s.filenames else 0,
            }
            for s in run.sets
        ]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)



# ----------------------------------------
# --- S2Areas storage and retrieval  -----
# ----------------------------------------

def store_s2area(s2: S2Areas, 
                 set_pmt: Optional[SetPmt] = None,
                 output_dir: Optional[Path] = None,
                 suffix: str = "s2_areas") -> None:
    """
    Store S2Areas object to disk in processed_data/all/ directory.
    
    Saves NPZ file with areas and UIDs. Can be used for both recoil S2 areas
    and X-ray areas by changing the suffix.
    
    Args:
        s2: S2Areas object with integration and fit results
        set_pmt: Optional SetPmt to save metadata from
        output_dir: Optional custom output directory (for testing)
        suffix: Filename suffix (default: "s2_areas", use "xray_areas" for X-rays)
    """
    # Determine base directory
    if output_dir is None:
        base_dir = get_output_root(s2.source_dir.parent)
    else:
        base_dir = output_dir

    # Save to s2_areas/ subdirectory
    all_dir = base_dir / "s2_areas"
    all_dir.mkdir(parents=True, exist_ok=True)
    
    set_name = s2.source_dir.name
    
    # Save areas with appropriate key name
    # For "s2_areas" suffix -> key is "s2_areas"
    # For "xray_areas" suffix -> key is "xray_areas"
    path_areas = all_dir / f"{set_name}_{suffix}.npz"
    area_key = suffix if suffix.endswith('_areas') else f"{suffix}_areas"
    
    np.savez_compressed(path_areas, 
                       uids=s2.uids.astype(np.uint32), 
                       **{area_key: s2.areas})
    
    print(f"    💾 Saved {suffix.replace('_', ' ')} to s2_areas/{path_areas.name}")

    if set_pmt is not None:
        save_set_metadata(set_pmt)
"""
    # Build complete results dictionary
    results_dict = {
        "method": s2.method,
        "params": s2.params,
        "mean": float(s2.mean) if s2.mean is not None else None,
        "sigma": float(s2.sigma) if s2.sigma is not None else None,
        "ci95": float(s2.ci95) if s2.ci95 is not None else None,
        "fit_success": s2.fit_success,
    }
    
    # Add set metadata if provided
    if set_pmt is not None:
        results_dict["set_metadata"] = {
            "t_s1": getattr(set_pmt, "t_s1", None),
            "t_s1_std": getattr(set_pmt, "t_s1_std", None),
            "t_s2_start": getattr(set_pmt, "t_s2_start", None),
            "t_s2_start_std": getattr(set_pmt, "t_s2_start_std", None),
            "t_s2_end": getattr(set_pmt, "t_s2_end", None),
            "t_s2_end_std": getattr(set_pmt, "t_s2_end_std", None),
            "s2_duration": getattr(set_pmt, "s2_duration", None),
            "s2_duration_std": getattr(set_pmt, "s2_duration_std", None),
            "drift_field": float(set_pmt.drift_field) if set_pmt.drift_field is not None else None,
            "EL_field": float(set_pmt.EL_field) if set_pmt.EL_field is not None else None,
            "time_drift": float(set_pmt.time_drift) if set_pmt.time_drift is not None else None,
            "speed_drift": float(set_pmt.speed_drift) if set_pmt.speed_drift is not None else None,
            "red_drift_field": float(set_pmt.red_drift_field) if set_pmt.red_drift_field is not None else None,
        }
    
    # Save complete results as JSON
    path_results = output_dir / f"{set_name}_s2_results.json"
    with open(path_results, "w") as f:
        json.dump(results_dict, f, indent=2)
"""

def load_s2area(set_pmt: SetPmt, input_dir: Optional[Path] = None) -> S2Areas:
    """
    Load S2Areas object from processed_data/all/ directory.
    
    Args:
        set_pmt: SetPmt object with source_dir
        input_dir: Optional custom input directory (for testing)
        
    Returns:
        S2Areas with all saved attributes populated
    """
    # Determine base directory
    if input_dir is None:
        base_dir = get_output_root(set_pmt.source_dir.parent)
    else:
        base_dir = input_dir
    
    set_name = set_pmt.source_dir.name
    
    # Load raw areas from all/ subdirectory
    path_areas = base_dir / f"{set_name}_s2_areas.npz"
    arr = np.load(path_areas, allow_pickle=True)
    
    # Try to load complete results from the set_summaries metadata directory
    metadata_file = get_output_root(set_pmt.source_dir.parent) / "set_summaries" / f"{set_name}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            results = json.load(f)
        
        return S2Areas(
            source_dir=set_pmt.source_dir,
            areas=arr['s2_areas'],
            uids=arr['uids'],
            method=results.get("method", "loaded_from_npz"),
            params=results.get("params", {"set_metadata": asdict(set_pmt)}),
            mean=results.get("mean"),
            sigma=results.get("sigma"),
            ci95=results.get("ci95"),
            fit_success=results.get("fit_success", False),
            fit_result=None
        )
    else:
        return S2Areas(
            source_dir=set_pmt.source_dir,
            areas=arr['s2_areas'],
            uids=arr['uids'],
            method="loaded_from_npz",
            params={"set_metadata": asdict(set_pmt)}
        )

def store_isotope_df(df: pd.DataFrame, filepath: PathLike) -> None:
    """
    Store the isotope-assigned dataframe in compact binary parquet format.
    """
    df.to_parquet(filepath, index=False)
    print(f"Saved dataframe → {filepath}")


# ----------------------------------------
# --- Figure saving utility  ------------
# ----------------------------------------

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
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {filename}")


# ----------------------------------------
# --- Fit result saving and loading  -----
# ----------------------------------------

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

def load_fit_result(input_path: Union[str, Path]) -> ModelResult:
    """
    Loads an lmfit ModelResult from a JSON file for downstream plotting or analysis.
    """
    return lmfit.model.load_modelresult(str(input_path))