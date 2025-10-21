import numpy as np
from pathlib import Path
from typing import Union, Iterator
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

def store_s2area(s2: S2Areas) -> None:
    """
    Store S2Areas object to disk, including all fit results.
    
    Saves two files:
    - s2_areas.npy: Raw area array for quick access
    - s2_results.json: Complete metadata including fit results
    """
    # Save raw areas as numpy array (for backward compatibility)
    path_areas = s2.source_dir / "s2_areas.npy"
    np.save(path_areas, s2.areas)
    
    # Save complete results as JSON
    path_results = s2.source_dir / "s2_results.json"
    results_dict = {
        "method": s2.method,
        "params": s2.params,
        "mean": float(s2.mean) if s2.mean is not None else None,
        "sigma": float(s2.sigma) if s2.sigma is not None else None,
        "ci95": float(s2.ci95) if s2.ci95 is not None else None,
        "fit_success": s2.fit_success,
        # Note: fit_result (lmfit ModelResult) is not JSON-serializable
        # If needed for reload, consider using pickle for fit_result separately
    }
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

def store_xray_results(xr: XRayResults, path: PathLike = None) -> None:
    """Store XRayResults in .npy file inside the set's directory."""
    if path is None:
        path = xr.set_id / "xray_results.npy"
    np.save(path, xr.events)


def store_xrayset(xrays: XRayResults, outdir: Path = None) -> None:
    """
    Store results of X-ray classification.

    Saves:
      - accepted areas as a .npy array (fast reload for histograms)
      - full classification log as a .json (audit trail)
    """
    if outdir is None:
        outdir = Path(xrays.set_id)  # assume set_id is a directory name
    outdir = Path(outdir)

    # Extract numeric data (only accepted areas)
    accepted_areas = [ev.area for ev in xrays.events if ev.accepted and ev.area is not None]
    np.save(outdir / "xray_areas.npy", np.array(accepted_areas))

    # Full event-level log
    log = []
    for ev in xrays.events:
        log.append({
            "wf_id": ev.wf_id,
            "accepted": ev.accepted,
            "reason": ev.reason,
            "area": ev.area if ev.area is not None else None,
        })

    meta = {
        "set_id": xrays.set_id,
        "params": xrays.params,
        "n_events": len(xrays.events),
        "n_accepted": sum(ev.accepted for ev in xrays.events),
        "n_rejected": sum(not ev.accepted for ev in xrays.events),
        "events": log,
    }

    with open(outdir / "xray_results.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_xray_results(run: Run) -> np.ndarray:
    """
    Load all X-ray results from a run's sets.

    Args:
        run: Run object with sets populated

    Returns:
        Array of accepted X-ray S2 areas (mV·µs)
    """
    xray_areas = []
    
    for set_pmt in run.sets:
        xray_file = set_pmt.source_dir / 'xray_results.npy'
        
        if not xray_file.exists():
            print(f"Warning: No X-ray results found for {set_pmt.source_dir.name}")
            continue
            
        try:
            xres = np.load(xray_file, allow_pickle=True).item()
            # Extract accepted events from all waveforms
            areas = np.array([
                event.area 
                for wfm_events in xres.events 
                for event in wfm_events 
                if event.accepted
            ])
            xray_areas.append(areas)
        except Exception as e:
            print(f"Error loading X-ray results from {set_pmt.source_dir.name}: {e}")
    
    if not xray_areas:
        raise ValueError("No X-ray results could be loaded from any set")
    
    # Flatten all areas into single array
    return np.concatenate(xray_areas)

