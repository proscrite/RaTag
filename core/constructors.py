from dataclasses import replace
from pathlib import Path
from typing import  Optional

from .dataIO import load_wfm, parse_subdir_name
from .datatypes import SetPmt, Run
from .units import V, to_Td, cm_to_mm
from .physics import compute_reduced_field, redfield_to_speed

# ------------------------
# --- Run constructor  ---
# ------------------------
def populate_run(run: Run, nfiles: Optional[int] = None) -> Run:
    """
    Populate a Run with all measurement sets from subdirectories.
    
    Each subdirectory in run.root_directory becomes a SetPmt.
    FastFrame properties are automatically detected per set.
    
    Args:
        run: Run object with root_directory
        nfiles: Optional limit on files per set
        
    Returns:
        Run with sets populated
    """
    sets = []
    subdirs = [d for d in run.root_directory.iterdir() if (d.is_dir() and 'FieldScan' in d.name)]
    
    for subdir in sorted(subdirs):
        try:
            set_pmt = set_from_dir(subdir, nfiles=nfiles)
            sets.append(set_pmt)
            
            # Log FastFrame info
            ff_info = f"FastFrame ({set_pmt.nframes} frames/file)" if set_pmt.ff else "single-frame"
            print(f"  Loaded: {subdir.name} - {len(set_pmt)} files ({set_pmt.n_waveforms} waveforms) [{ff_info}]")
        except Exception as e:
            print(f"  Warning: Failed to load {subdir.name}: {e}")
    
    return replace(run, sets=sets)

# ------------------------
# --- Set constructors ---
# ------------------------

def set_from_dir(source_dir: Path, nfiles: Optional[int] = None) -> SetPmt:
    """
    Create SetPmt from directory by lazy-loading filenames.
    
    Automatically detects FastFrame properties from the first file.
    
    Args:
        source_dir: Path to directory containing .wfm files
        nfiles: Optional limit on number of files to load
        
    Returns:
        SetPmt with filenames and FastFrame properties detected
    """
    source_dir = Path(source_dir)
    
    # Get all .wfm files
    all_files = sorted(source_dir.glob("*.wfm"))
    
    if not all_files:
        raise FileNotFoundError(f"No .wfm files found in {source_dir}")
    
    # Limit files if requested
    files_to_use = all_files[:nfiles] if nfiles is not None else all_files
    filenames = [f.name for f in files_to_use]
    
    # Detect FastFrame properties from first file
    first_wf = load_wfm(files_to_use[0])
    ff = first_wf.ff
    nframes = first_wf.nframes if ff else 1
    
    # Parse metadata from directory name
    metadata = parse_subdir_name(source_dir.name)
    
    return SetPmt(
        source_dir=source_dir,
        filenames=filenames,
        metadata=metadata,
        ff=ff,
        nframes=nframes
    )


def set_fields(set_pmt: SetPmt, drift_gap_cm: float, el_gap_cm: float,
               gas_density: float = None) -> SetPmt:
    """
    Return a new SetPmt with drift/EL fields and reduced fields.
    """
    try:
        v_gate = V(set_pmt.metadata.get("gate"))
        v_anode = V(set_pmt.metadata.get("anode"))
    except Exception as e:
        raise ValueError(f"Voltage metadata missing or invalid in {set_pmt.source_dir.name}: {e}")

    drift_field = v_gate / drift_gap_cm if v_gate is not None else None
    EL_field = (v_anode - v_gate) / el_gap_cm if v_anode is not None and v_gate is not None else None

    red_drift_Td = None
    red_EL_Td = None
    if gas_density and drift_field and EL_field:
        red_drift_Vcm2 = compute_reduced_field(drift_field, gas_density) # V·cm²
        red_drift_Td = to_Td(red_drift_Vcm2)  # Convert to Td
        
        red_EL_Vcm2 = compute_reduced_field(EL_field, gas_density) # V·cm²
        red_EL_Td = to_Td(red_EL_Vcm2)  # Convert to Td

    return replace(set_pmt,
                   drift_field=drift_field,
                   EL_field=EL_field,
                   red_drift_field=red_drift_Td,
                   red_EL_field=red_EL_Td)

def set_transport_properties(set_pmt: SetPmt,
                             drift_gap_cm: float,
                             transport = None) -> SetPmt:
    """
    Given a SetPmt and geometry + transport model,
    return a new SetPmt with drift speed, drift time,
    and diffusion coefficient filled in.
    Args:
        set_pmt: input SetPmt with red_drift_field set
        drift_gap_cm: drift length in cm
        transport: module with transport functions (e.g. RaTag.transport)
    Returns: a new SetPmt with transport properties set.
    """
    if set_pmt.red_drift_field is None:
        raise ValueError("red_drift_field must be set before calling set_transport_properties")

    # calculate drift speed from reduced field
    speed_mmus = redfield_to_speed(set_pmt.red_drift_field)  # returns mm/us

    # drift time = L / v
    if not drift_gap_cm or drift_gap_cm <= 0:
        raise ValueError("drift_gap_cm must be positive")
    
    drift_gap_mm = cm_to_mm(drift_gap_cm)

    time_drift_us = drift_gap_mm / speed_mmus if speed_mmus else None

    # diffusion coefficient (model-dependent)
    # diffusion = transport.redfield_to_diffusion(set_pmt.red_drift_field)
    diffusion = None  # Placeholder if no diffusion model provided

    return replace(set_pmt,
                   speed_drift=speed_mmus,
                   time_drift=time_drift_us,
                   diffusion_coefficient=diffusion)