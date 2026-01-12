import numpy as np 
import pandas as pd
from pathlib import Path
from typing import Union, Iterator, Optional
import re
import json
import itertools
from dataclasses import replace

from .units import V_to_mV, s_to_us
from .datatypes import SiliconWaveform, Waveform, SetPmt, Run, S2Areas, PMTWaveform, XRayResults, FrameProxy
from .wfm2read_fast import wfm2read # type: ignore
PathLike = Union[str, Path]

# -------------------------------------
# --- Load waveform from .wfm file  ---
# -------------------------------------

def _load_wfm_V_s(path: PathLike) -> PMTWaveform:
    """Load waveform from a .wfm file storing (t, v)."""
    wfm = wfm2read(str(path))
    t, v = wfm[1], wfm[0]
    v = -v  # Invert signal polarity
    if len(v.shape) > 1:  # FastFrame format
        ff = True
        nframes = v.shape[0]
    else:
        ff = False
        nframes = 1
    return PMTWaveform(t, v, source=str(path), ff=ff, nframes=nframes)

def load_wfm(path: PathLike) -> PMTWaveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wf = _load_wfm_V_s(path)
    t_s, v_V = wf.t, wf.v

    v_mV = V_to_mV(v_V)
    t_us = s_to_us(t_s)

    return PMTWaveform(t=t_us, v=v_mV, source=wf.source, ff=wf.ff, nframes=wf.nframes)


def load_alpha(path: PathLike) -> SiliconWaveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wf = _load_wfm_V_s(path)
    t_s, v_V = wf.t, -wf.v

    t_us = s_to_us(t_s)

    return SiliconWaveform(t=t_us, v=v_V, source=wf.source, ff=wf.ff, nframes=wf.nframes)

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

def iter_frameproxies(set_pmt: SetPmt, chunk_dir: str = None, fmt='8b', scale=0.1, max_files: int=None, show_progress: bool = False) -> Iterator[FrameProxy]:
    """
    Iterate over files in the SetPmt and yield FrameProxy objects.
    - set_pmt.filenames must be list of file paths
    - parse file_seq via your existing logic (use parse_file_seq_from_name or sequential)

    Optional progress printing:
    - When `show_progress=True` prints integer percent completion (frames) and
      the current file number / total files. The percent is printed only when
      the integer value increases (1% steps).
    """
    from RaTag.alphas.energy_map_writer import parse_file_seq_from_name

    if chunk_dir is None:     # Handle custom chunk_dirs, if None, default is the set_pmt raw_data loc
        chunk_dir = set_pmt.source_dir

    files = set_pmt.filenames if max_files is None else set_pmt.filenames[:max_files]

    total_files = len(files)
    total_frames = total_files * (set_pmt.nframes or 0)

    frames_processed = 0
    last_percent = -1
    current_file_num = 0

    for p in files:
        file_seq = parse_file_seq_from_name(p)  # or use manifest mapping if you used it earlier
        current_file_num += 1
        # optionally verify path exists
        for frame_idx in range(set_pmt.nframes):
            # Update progress counters
            frames_processed += 1
            if show_progress and total_frames > 0:
                percent = int((frames_processed * 100) / total_frames)
                if percent > last_percent:
                    last_percent = percent
                    print(f"  {percent}% complete — file {current_file_num}/{total_files}")

            yield FrameProxy(file_path=set_pmt.source_dir / p,
                             file_seq=file_seq,
                             frame_idx=frame_idx,
                             chunk_dir=chunk_dir,
                             fmt=fmt, scale=scale)
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
    metadata_dir = set_pmt.source_dir.parent / "processed_data"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = metadata_dir / f"{set_pmt.source_dir.name}_metadata.json"
    
    # Load existing metadata if it exists
    existing_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
    
    attrs = ['drift_field', 'EL_field', 'red_drift_field', 'red_EL_field', 'speed_drift', 'time_drift']

    # Start with existing metadata, then update with new values
    metadata = {
        **existing_metadata,  # Keep existing data
        "set_name": set_pmt.source_dir.name,
        "source_dir": str(set_pmt.source_dir),
    }
    
    # Update field/transport properties (only if not None)
    for key in attrs:
        value = getattr(set_pmt, key)
        if value is not None:
            metadata[key] = round(value, 3) if isinstance(value, float) else value
    
    # Update metadata fields (only if not None)
    for key, value in set_pmt.metadata.items():
        if value is not None:
            metadata[key] = round(value, 3) if isinstance(value, float) else value
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_set_metadata(set_pmt: SetPmt) -> Optional[SetPmt]:
    """
    Load set metadata from JSON if it exists.
    
    File location: {run_dir}/processed_data/{set_name}_metadata.json
    
    Returns:
        Updated SetPmt with loaded metadata, or None if file doesn't exist
    """
    # Look in processed_data at run level
    metadata_dir = set_pmt.source_dir.parent / "processed_data"
    metadata_file = metadata_dir / f"{set_pmt.source_dir.name}_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    # Load ALL metadata keys (not just specific ones)
    metadata = {k: v for k, v in data.items() 
                if k not in ["set_name", "source_dir", "drift_field", "EL_field", 
                            "red_drift_field", "red_EL_field", "speed_drift", 
                            "time_drift", "diffusion_coefficient"]}
    
    # Restore SetPmt with all properties
    return replace(
        set_pmt,
        metadata={**set_pmt.metadata, **metadata},  # Merge with existing
        drift_field=data.get("drift_field"),
        EL_field=data.get("EL_field"),
        red_drift_field=data.get("red_drift_field"),
        red_EL_field=data.get("red_EL_field"),
        speed_drift=data.get("speed_drift"),
        time_drift=data.get("time_drift"),
        diffusion_coefficient=data.get("diffusion_coefficient")
    )

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
                "v_gate": s.metadata['gate'],
                "v_anode": s.metadata['anode'],
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
        base_dir = s2.source_dir.parent / "processed_data"
    else:
        base_dir = output_dir
    
    # Always save to all/ subdirectory
    all_dir = base_dir / "all"
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
    
    print(f"    💾 Saved {suffix.replace('_', ' ')} to all/{path_areas.name}")

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
            "t_s1": set_pmt.metadata.get("t_s1"),
            "t_s1_std": set_pmt.metadata.get("t_s1_std"),
            "t_s2_start": set_pmt.metadata.get("t_s2_start"),
            "t_s2_start_std": set_pmt.metadata.get("t_s2_start_std"),
            "t_s2_end": set_pmt.metadata.get("t_s2_end"),
            "t_s2_end_std": set_pmt.metadata.get("t_s2_end_std"),
            "s2_duration": set_pmt.metadata.get("s2_duration"),
            "s2_duration_std": set_pmt.metadata.get("s2_duration_std"),
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
        base_dir = set_pmt.source_dir.parent / "processed_data"
    else:
        base_dir = input_dir
    
    # Always load from all/ subdirectory
    all_dir = base_dir / "all"
    
    set_name = set_pmt.source_dir.name
    
    # Load raw areas from all/ subdirectory
    path_areas = all_dir / f"{set_name}_s2_areas.npz"
    arr = np.load(path_areas, allow_pickle=True)
    
    # Try to load complete results
    # Note: metadata.json is at root level, not in all/ subdirectory
    path_results = base_dir / f"{set_name}_metadata.json"
    if path_results.exists():
        with open(path_results, "r") as f:
            results = json.load(f)
        
        return S2Areas(
            source_dir=set_pmt.source_dir,
            areas=arr['s2_areas'],
            uids=arr['uids'],
            method=results.get("method", "loaded_from_npz"),
            params=results.get("params", {"set_metadata": set_pmt.metadata}),
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
            params={"set_metadata": set_pmt.metadata}
        )

# ------------------------------------------
# --- XRayResults storage and retrieval  ---
# ------------------------------------------
# NOTE: These functions are obsolete and kept for backward compatibility only.
# New X-ray workflow uses store_s2area() with suffix="xray_areas" instead.

def store_xray_results(xr, path: Optional[PathLike] = None) -> None:
    """
    DEPRECATED: Store XRayResults in .npy file inside the set's directory.
    
    This function is obsolete. Use store_s2area() with suffix="xray_areas" instead.
    """
    raise NotImplementedError(
        "store_xray_results() is deprecated. "
        "Use store_s2area(s2_areas, set_pmt, suffix='xray_areas') instead."
    )


def store_xrayset(xrays, outdir: Optional[Path] = None) -> None:
    """
    DEPRECATED: Store results of X-ray classification.
    
    This function is obsolete. Use store_s2area() with suffix="xray_areas" instead.
    """
    raise NotImplementedError(
        "store_xrayset() is deprecated. "
        "Use store_s2area(s2_areas, set_pmt, suffix='xray_areas') instead."
    )

    # Aggregate rejection reasons
    rejection_reasons = {}
    for ev in xrays.events:
        if not ev.accepted:
            reason = ev.reason or "unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    # Filter out non-serializable items from params (e.g., 'integrator' function)
    params_serializable = {
        k: v for k, v in xrays.params.items() 
        if k != 'integrator' and not callable(v)
    }
    
    # Summary statistics only (no individual events)
    meta = {
        "set_id": str(xrays.set_id),  # Convert Path to string for JSON serialization
        "params": params_serializable,
        "n_events": len(xrays.events),
        "n_accepted": sum(ev.accepted for ev in xrays.events),
        "n_rejected": sum(not ev.accepted for ev in xrays.events),
        "rejection_reasons": rejection_reasons,
        "accepted_area_stats": {
            "mean": float(np.mean(accepted_areas)) if accepted_areas else None,
            "std": float(np.std(accepted_areas)) if accepted_areas else None,
            "min": float(np.min(accepted_areas)) if accepted_areas else None,
            "max": float(np.max(accepted_areas)) if accepted_areas else None,
        }
    }

    with open(outdir / "xray_results.json", "w") as f:
        json.dump(meta, f, indent=2)



def load_xray_results(run: Run) -> dict[str, XRayResults]:
    """
    Load X-ray classification results for all sets in a run.
    
    Args:
        run: Run object with sets to load X-ray results from
        
    Returns:
        Dictionary mapping set names to XRayResults objects
        
    Raises:
        ValueError: If no X-ray results could be loaded from any set
    """
    raise NotImplementedError(
        "store_xrayset() is deprecated. "
        "Use store_s2area(s2_areas, set_pmt, suffix='xray_areas') instead."
    )
    xray_results = {}
    
    for set_pmt in run.sets:
        xray_file = set_pmt.source_dir / 'xray_areas.npy'
        
        if not xray_file.exists():
            print(f"Warning: X-ray results file not found for {set_pmt.source_dir.name}")
            continue
            
        try:
            # Load the numpy array of XRayEvent objects
            events = np.load(xray_file, allow_pickle=True)
            
            # Convert array to list if needed
            if isinstance(events, np.ndarray):
                events = events.tolist()
            
            # Create XRayResults object
            xray_result = XRayResults(
                set_id=set_pmt.source_dir,
                events=events,
                params={}  # Add params if you have them stored elsewhere
            )
            
            xray_results[set_pmt.source_dir.name] = xray_result
            
        except Exception as e:
            print(f"Warning: Failed to load X-ray results from {xray_file}: {e}")
            continue
    
    if not xray_results:
        raise ValueError("No X-ray results could be loaded from any set")
    
    return xray_results


def aggregate_xray_areas(run: Run, output_dir: Optional[Path] = None) -> Run:
    """
    Aggregate X-ray areas from all sets in a run and save combined result.
    
    X-rays are not affected by drift field (except statistics), so we combine
    all accepted X-ray events across all sets to improve statistics for fitting.
    
    Args:
        run: Run object with X-ray classification completed
        output_dir: Optional output directory (defaults to run.root_directory/processed_data)
        
    Returns:
        Unchanged run object (pure side effect: saves combined areas to disk)
        
    Raises:
        FileNotFoundError: If no X-ray results found for any set
    """
    all_areas = []
    all_uids = []
    
    for set_pmt in run.sets:
        xray_file = set_pmt.source_dir.parent / "processed_data" / "all" / f"{set_pmt.source_dir.name}_xray_areas.npz"
        
        if not xray_file.exists():
            print(f"  ⚠ No X-ray results for {set_pmt.source_dir.name} - skipping")
            continue
        
        data = np.load(xray_file)
        areas = data['xray_areas']
        uids = data['uids']
        all_areas.append(areas)
        all_uids.append(uids)
        print(f"  ✓ Loaded {len(areas)} X-ray events from {set_pmt.source_dir.name}")
    
    if not all_areas:
        raise FileNotFoundError("No X-ray results found for any set in run")
    
    combined_areas = np.concatenate(all_areas)
    combined_uids = np.concatenate(all_uids)
    print(f"\n  📊 Combined: {len(combined_areas)} total X-ray events")
    
    # Save combined areas and UIDs
    if output_dir is None:
        output_dir = run.root_directory / "processed_data"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{run.run_id}_xray_areas_combined.npz"
    np.savez_compressed(output_file,
                       xray_areas=combined_areas,
                       uids=combined_uids.astype(np.uint32))
    print(f"  💾 Saved combined areas and UIDs to {output_file.name}")
    
    return run  # ← Return run unchanged


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

