#!/usr/bin/env python3
"""
energy_map_reader.py

Utilities to read the binary energy mapping chunk files produced by energy_map_writer.py.

Supports both formats:
 - 8b: dtype = [('id','<u4'), ('E','<f4')]
 - 6b: dtype = [('id','<u4'), ('E','<u2')]  (must provide scale when decoding)

Functions:
 - load_chunk(path, fmt='8b', scale=0.1) -> returns (ids_array, energies_array_float)
 - find_chunk_for_fileseq(chunk_dir, file_seq) -> returns chunk_path matching filename ranges
 - get_energy_for_frame(chunk_path_or_arrays, file_seq, frame_idx, fmt='8b', scale=0.1)

Also provides a simple CLI for lookup.

Filename convention expected:
    energy_map_f{start:06d}-f{end:06d}.bin
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
import re
from typing import Tuple, List

# global cache per-process
_ENERGY_INDEX_CACHE = {}


def load_chunk(path, fmt='8b', scale=0.1):
    """
    Load chunk file and return (ids, energies_float).
    """
    if fmt == '8b':
        dtype = np.dtype([('id','<u4'), ('E','<f4')])
        arr = np.fromfile(path, dtype=dtype)
        ids = arr['id'].astype(np.uint32)
        Es = arr['E'].astype(np.float32)
        return ids, Es
    elif fmt == '6b':
        dtype = np.dtype([('id','<u4'), ('E','<u2')])
        arr = np.fromfile(path, dtype=dtype)
        ids = arr['id'].astype(np.uint32)
        Es = arr['E'].astype(np.float32) * float(scale)
        return ids, Es
    else:
        raise ValueError("fmt must be '8b' or '6b'")

def parse_chunk_range_from_fname(fname):
    """
    Parse start_file_seq and end_file_seq from filename pattern energy_map_f{start}-f{end}.bin
    Returns (start, end) as ints, or None if not matched.
    """
    b = os.path.basename(fname)
    m = re.match(r'energy_map_f(\d{6})-f(\d{6})\.bin$', b)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def find_chunk_for_fileseq(chunk_dir, file_seq):
    """
    Search chunk_dir for a chunk that covers file_seq. Returns path or None.
    """
    files = sorted(glob(os.path.join(chunk_dir, "energy_map_f*-f*.bin")))
    for f in files:
        rng = parse_chunk_range_from_fname(f)
        if rng is None:
            continue
        start, end = rng
        if start <= file_seq <= end:
            return f
    return None

def get_energy_for_frame(chunk_dir_or_path, file_seq, frame_idx, fmt='8b', scale=0.1, preload_cache=None):
    """
    Get energy for a single frame specified by file_seq and frame_idx.
    - chunk_dir_or_path: either directory with chunk files or a specific chunk file path.
    - preload_cache: optional dict to cache loaded chunks {path: (ids,Es)} for repeated queries.
    Returns energy (float) or None if not found.
    """
    uid = int(file_seq) * 64 + int(frame_idx)
    # determine chunk file
    if os.path.isdir(chunk_dir_or_path):
        chunk_path = find_chunk_for_fileseq(chunk_dir_or_path, file_seq)
        if chunk_path is None:
            return None
    else:
        chunk_path = chunk_dir_or_path

    if preload_cache is not None and chunk_path in preload_cache:
        ids, Es = preload_cache[chunk_path]
    else:
        ids, Es = load_chunk(chunk_path, fmt=fmt, scale=scale)
        if preload_cache is not None:
            preload_cache[chunk_path] = (ids, Es)

    # assuming ids sorted, use searchsorted
    idx = np.searchsorted(ids, uid)
    if idx < len(ids) and ids[idx] == uid:
        return float(Es[idx])
    return None


def _load_chunk_file(path: str, fmt: str='8b', scale: float=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Load single chunk into (ids, energies) arrays."""
    if fmt == '8b':
        dtype = np.dtype([('id','<u4'), ('E','<f4')])
        arr = np.fromfile(path, dtype=dtype)
        return arr['id'].astype(np.uint32), arr['E'].astype(np.float32)
    elif fmt == '6b':
        dtype = np.dtype([('id','<u4'), ('E','<u2')])
        arr = np.fromfile(path, dtype=dtype)
        return arr['id'].astype(np.uint32), arr['E'].astype(np.float32) * float(scale)
    else:
        raise ValueError("fmt must be '8b' or '6b'")

def load_energy_index(chunk_dir: str, fmt: str='8b', scale: float=0.1, force_reload: bool=False):
    """
    Load all chunk files under chunk_dir into a single sorted (ids, energies) pair and cache it.
    Small: for 200k frames this uses ~200k * (4+4) = 1.6MB memory.
    """
    key = f"{os.path.abspath(chunk_dir)}|{fmt}|{scale}"
    if key in _ENERGY_INDEX_CACHE and not force_reload:
        return _ENERGY_INDEX_CACHE[key]
    files = sorted(glob(os.path.join(chunk_dir, "energy_map_f*-f*.bin")))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    id_list = []
    e_list = []
    for f in files:
        ids, Es = _load_chunk_file(f, fmt=fmt, scale=scale)
        id_list.append(ids)
        e_list.append(Es)
    ids = np.concatenate(id_list)
    Es = np.concatenate(e_list)
    order = np.argsort(ids)
    ids = ids[order]
    Es = Es[order]
    _ENERGY_INDEX_CACHE[key] = (ids, Es)
    return ids, Es

def get_energy_for(file_seq: int, frame_idx: int, chunk_dir: str, fmt='8b', scale=0.1):
    """Return energy (float) or None if not found."""
    ids, Es = load_energy_index(chunk_dir, fmt=fmt, scale=scale)
    uid = int(file_seq) * 64 + int(frame_idx)
    i = np.searchsorted(ids, uid)
    if i < len(ids) and ids[i] == uid:
        return float(Es[i])
    return None

def get_energies_for_uids(uids: np.ndarray, chunk_dir: str, fmt='8b', scale=0.1):
    """Vectorized lookup (returns array same shape as uids, with np.nan for missing)."""
    ids, Es = load_energy_index(chunk_dir, fmt=fmt, scale=scale)
    i = np.searchsorted(ids, uids)
    out = np.full(uids.shape, np.nan, dtype=np.float32)
    mask = (i < len(ids)) & (ids[i] == uids)
    out[mask] = Es[i[mask]]
    return out


def bulk_load_all_chunks(chunk_dir, fmt='8b', scale=0.1):
    """
    Load all chunk files in chunk_dir and return concatenated (ids, Es) sorted by id.
    """
    files = sorted(glob(os.path.join(chunk_dir, "energy_map_f*-f*.bin")))
    id_list = []
    e_list = []
    for f in files:
        ids, Es = load_chunk(f, fmt=fmt, scale=scale)
        id_list.append(ids)
        e_list.append(Es)
    if not id_list:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)
    ids = np.concatenate(id_list)
    Es = np.concatenate(e_list)
    # ensure sorted by id
    order = np.argsort(ids)
    return ids[order], Es[order]

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, help="Directory containing chunk files or specific chunk file path")
    p.add_argument("--file-seq", type=int, help="file_seq (integer)")
    p.add_argument("--frame-idx", type=int, help="frame_idx (0..48)")
    p.add_argument("--format", choices=("8b","6b"), default="8b", help="mapping format")
    p.add_argument("--scale", type=float, default=0.1, help="scale for 6b format (keV per LSB)")
    p.add_argument("--dump-all", action="store_true", help="load and print summary of all chunk files")
    args = p.parse_args()

    if args.dump_all:
        ids, Es = bulk_load_all_chunks(args.chunks, fmt=args.format, scale=args.scale)
        print(f"Loaded total entries: {len(ids)}")
        if len(ids) > 0:
            print(f"min id {ids.min()}, max id {ids.max()}, min E {Es.min()}, max E {Es.max()}")
        sys.exit(0)

    if args.file_seq is None or args.frame_idx is None:
        print("Provide --file-seq and --frame-idx to lookup a single frame, or use --dump-all.")
        sys.exit(1)

    val = get_energy_for_frame(args.chunks, args.file_seq, args.frame_idx, fmt=args.format, scale=args.scale)
    if val is None:
        print("NOT FOUND")
        sys.exit(2)
    print(f"Energy (file_seq={args.file_seq} frame_idx={args.frame_idx}) = {val}")
