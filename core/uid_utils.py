import numpy as np
import re
import random
from pathlib import Path
from RaTag.core.datatypes import SetPmt

def parse_file_seq_from_name(fname: str) -> int:
    base = Path(fname).name

    if m := re.search(r'_Batch(\d+)', base, re.IGNORECASE):
        return int(m.group(1))
    
    if m := re.search(r'_(\d+)Wfm', base, re.IGNORECASE):
        return int(m.group(1))
        
    #  Legacy fallback: Anchored to the end of the filename
    # Matches "_001.wfm" or "_001_CH3.wfm", physically avoiding datestamps
    if m := re.search(r'_(\d+)(?:_CH\d+)?\.wfm$', base, re.IGNORECASE):
        return int(m.group(1))
    
    # Last-resort - raise to avoid silent mis-parses
    raise ValueError(f"Cannot parse file_seq from filename {base}")

def make_uid(file_seq: int, frame_idx: int) -> int:
    # frame_idx expected 0..48
    return int(file_seq) * 64 + int(frame_idx)

def decode_uid(uid: int):
    file_seq = uid // 64
    frame_idx = uid % 64
    return file_seq, frame_idx

def generate_all_uids_for_set(set_pmt) -> np.ndarray:
    """Generates all possible UIDs (every frame of every file) for a SetPmt."""
    file_seqs = [parse_file_seq_from_name(fn) for fn in set_pmt.filenames]
    
    # FastFrame files have nframes, standard files have 1 frame
    n_frames = set_pmt.nframes if set_pmt.ff else 1
    
    uids = [make_uid(fs, fi) for fs in file_seqs for fi in range(n_frames)]
    return np.array(uids, dtype=np.uint32)

def sample_validation_waveforms(set_pmt: SetPmt, accepted_uids: np.ndarray, n_samples: int = 4) -> tuple[list, list]:
    """Safely samples accepted and rejected waveforms for a single set."""    
    from RaTag.io.file_ops import load_waveform_by_uid 

    all_uids = generate_all_uids_for_set(set_pmt)
    set_acc_uids = np.intersect1d(all_uids, accepted_uids)
    set_rej_uids = np.setdiff1d(all_uids, set_acc_uids)
    
    acc_sample = random.sample(list(set_acc_uids), min(n_samples, len(set_acc_uids)))
    rej_sample = random.sample(list(set_rej_uids), min(n_samples, len(set_rej_uids)))
    
    # 2. Native I/O
    acc_wfs = [load_waveform_by_uid(set_pmt, u) for u in acc_sample]
    rej_wfs = [load_waveform_by_uid(set_pmt, u) for u in rej_sample]
    
    return acc_wfs, rej_wfs

