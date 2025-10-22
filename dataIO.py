import numpy as np
from pathlib import Path
from typing import Union, Iterator, Optional
import re
import json

from .datatypes import Waveform, SetPmt, Run, S2Areas, PMTWaveform, XRayResults

from RaTag.scripts.wfm2read_fast import wfm2read # type: ignore
PathLike = Union[str, Path]

def load_wfm(path: PathLike) -> Waveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wfm = wfm2read(str(path))
    t, v = wfm[1], -wfm[0]  # Invert signal polarity
    if len(v.shape) > 1:  # FastFrame format
        ff = True
        nframes = v.shape[0]
    else:
        ff = False
        nframes = 1
    return Waveform(t, v, source=str(path), ff=ff, nframes=nframes)

# --- Lazy loader ---
def iter_waveforms(set_pmt: SetPmt) -> Iterator[PMTWaveform]:
    """Yield PMTWaveform objects lazily, one by one."""
    
    for fn in set_pmt.filenames:
        yield load_wfm(set_pmt.source_dir / fn)


# --- Extract single waveform from FastFrame ---
def extract_single_frame(wf: Waveform, frame: int = 0) -> PMTWaveform:
    """Extract a single frame from a FastFrame waveform."""
    if not wf.ff:
        raise ValueError("Waveform is not FastFrame format")
    if frame < 0 or frame >= wf.nframes:
        raise ValueError(f"Frame index {frame} out of range [0, {wf.nframes})")
    v_single = wf.v[frame, :]
    return PMTWaveform(t=wf.t, v=v_single, source=wf.source, ff=False, nframes=1)

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

def store_s2area(s2: S2Areas, set_pmt: Optional[SetPmt] = None) -> None:
    """
    Store S2Areas object to disk, including all fit results and set metadata.
    
    Saves two files:
    - s2_areas.npy: Raw area array for quick access
    - s2_results.json: Complete metadata including fit results and analysis parameters
    
    Args:
        s2: S2Areas object with integration and fit results
        set_pmt: Optional SetPmt to extract complete metadata from
    """
    # Save raw areas as numpy array (for backward compatibility)
    path_areas = s2.source_dir / "s2_areas.npy"
    np.save(path_areas, s2.areas)
    
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
            "t_s2_start_mean": set_pmt.metadata.get("t_s2_start_mean"),
            "t_s2_start_std": set_pmt.metadata.get("t_s2_start_std"),
            "t_s2_end_mean": set_pmt.metadata.get("t_s2_end_mean"),
            "t_s2_end_std": set_pmt.metadata.get("t_s2_end_std"),
            "s2_duration_mean": set_pmt.metadata.get("s2_duration_mean"),
            "s2_duration_std": set_pmt.metadata.get("s2_duration_std"),
            "drift_field": float(set_pmt.drift_field) if set_pmt.drift_field is not None else None,
            "EL_field": float(set_pmt.EL_field) if set_pmt.EL_field is not None else None,
            "time_drift": float(set_pmt.time_drift) if set_pmt.time_drift is not None else None,
            "speed_drift": float(set_pmt.speed_drift) if set_pmt.speed_drift is not None else None,
            "red_drift_field": float(set_pmt.red_drift_field) if set_pmt.red_drift_field is not None else None,
        }
    
    # Save complete results as JSON
    path_results = s2.source_dir / "s2_results.json"
    with open(path_results, "w") as f:
        json.dump(results_dict, f, indent=2)

def load_s2area(set_pmt: SetPmt) -> S2Areas:
    """
    Load S2Areas object from disk, including fit results if available.
    
    Args:
        set_pmt: SetPmt object with source_dir pointing to data location
        
    Returns:
        S2Areas with all saved attributes populated
    """
    # Load raw areas
    path_areas = set_pmt.source_dir / "s2_areas.npy"
    areas = np.load(path_areas)
    
    # Try to load complete results
    path_results = set_pmt.source_dir / "s2_results.json"
    if path_results.exists():
        with open(path_results, "r") as f:
            results = json.load(f)
        
        return S2Areas(
            source_dir=set_pmt.source_dir,
            areas=areas,
            method=results.get("method", "loaded_from_npy"),
            params=results.get("params", {"set_metadata": set_pmt.metadata}),
            mean=results.get("mean"),
            sigma=results.get("sigma"),
            ci95=results.get("ci95"),
            fit_success=results.get("fit_success", False),
            fit_result=None  # fit_result object not saved in JSON
        )
    else:
        # Fallback for backward compatibility (old format without metadata)
        return S2Areas(
            source_dir=set_pmt.source_dir,
            areas=areas,
            method="loaded_from_npy",
            params={"set_metadata": set_pmt.metadata}
        )

def store_xray_results(xr: XRayResults, path: Optional[PathLike] = None) -> None:
    """Store XRayResults in .npy file inside the set's directory."""
    if path is None:
        path = xr.set_id / "xray_results.npy"
    np.save(path, xr.events)


def store_xrayset(xrays: XRayResults, outdir: Optional[Path] = None) -> None:
    """
    Store results of X-ray classification.

    Saves:
      - accepted areas as a .npy array (fast reload for histograms)
      - summary statistics as a .json (compact metadata)
    """
    if outdir is None:
        outdir = Path(xrays.set_id)  # assume set_id is a directory name
    outdir = Path(outdir)

    # Extract numeric data (only accepted areas)
    accepted_areas = [ev.area for ev in xrays.events if ev.accepted and ev.area is not None]
    np.save(outdir / "xray_areas.npy", np.array(accepted_areas))

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
    xray_results = {}
    
    for set_pmt in run.sets:
        xray_file = set_pmt.source_dir / 'xray_results.npy'
        
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


def store_xray_areas_combined(areas: np.ndarray, run: Run, output_dir: Optional[Path] = None) -> None:
    """
    Store combined X-ray areas from all sets in a run.
    
    Args:
        areas: Combined X-ray areas array
        run: Run object
        output_dir: Directory to save to (defaults to run.root_directory)
    """
    if output_dir is None:
        output_dir = run.root_directory
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save areas
    np.save(output_dir / f"{run.run_id}_xray_areas_combined.npy", areas)
    
    # Save metadata
    metadata = {
        "run_id": run.run_id,
        "n_areas": len(areas),
        "sets": [s.source_dir.name for s in run.sets],
        "mean": float(np.mean(areas)),
        "std": float(np.std(areas)),
    }
    
    with open(output_dir / f"{run.run_id}_xray_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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

