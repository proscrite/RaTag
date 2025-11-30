# RaTag/core/uid_utils.py
import re
from pathlib import Path

def parse_file_seq_from_name(fname: str) -> int:
    base = Path(fname).name
    m = re.search(r'P2_(\d+)Wfm', base)
    if m:
        return int(m.group(1))
    # fallback but conservative
    m2 = re.search(r'_(\d+)Wfm', base)
    if m2:
        return int(m2.group(1))
    # Last-resort - raise to avoid silent mis-parses
    raise ValueError(f"Cannot parse file_seq from filename {base}")

def make_uid(file_seq: int, frame_idx: int) -> int:
    # frame_idx expected 0..48
    return int(file_seq) * 64 + int(frame_idx)

def decode_uid(uid: int):
    file_seq = uid // 64
    frame_idx = uid % 64
    return file_seq, frame_idx
