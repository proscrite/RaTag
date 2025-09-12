import numpy as np
from pathlib import Path
from typing import Union
import re

from .datatypes import Waveform, SetPmt, S2Areas

from RaTag.scripts.wfm2read_fast import wfm2read # type: ignore
PathLike = Union[str, Path]

def load_wfm(path: PathLike) -> Waveform:
    """Load waveform from a .wfm file storing (t, -v)."""
    wfm = wfm2read(str(path))
    t, v = wfm[1], -wfm[0]  # Invert signal polarity
    return Waveform(t, v, source=str(path))

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
    """Store areas in .npy file inside the set's directory."""
    path = s2.source_dir / "s2_areas.npy"
    np.save(path, s2.areas)

def load_s2area(set_pmt: SetPmt) -> S2Areas:
    """Load areas from .npy file inside the set's directory."""
    path = set_pmt.source_dir / "s2_areas.npy"
    areas = np.load(path)
    return S2Areas(
        set_id=set_pmt.source_dir.name,
        areas=areas,
        method="loaded_from_npy",
        params={"set_metadata": set_pmt.metadata}
    )