#!/usr/bin/env python3
"""
energy_map_writer.py

Scan a directory of FastFrames files, compute per-frame alpha energies from
channel-4 waveforms, and write chunked mapping files.

Chunk file format options:
 - "8b": each record = (uint32 unique_id, float32 energy)  => 8 bytes/entry
 - "6b": each record = (uint32 unique_id, uint16 energy_scaled) => 6 bytes/entry
         requires a `scale` parameter (keV per LSB), e.g. scale=0.1 (0.1 keV units)

Unique id encoding:
    unique_id = file_seq * 64 + frame_idx
where frame_idx in [0..48] (49 frames/file). Use file_seq parsed deterministically
from sorted filenames or from an explicit manifest.

Usage examples:
    # write using 100 files per chunk, 8-byte format
    python energy_map_writer.py --indir ./fastframes --outdir ./maps --files-per-chunk 100 --format 8b

    # write using 10 files per chunk, 6-byte format with scale 0.05 keV
    python energy_map_writer.py --indir ./fastframes --outdir ./maps --files-per-chunk 10 --format 6b --scale 0.05

Notes:
 - You MUST adapt `read_fastframe_ch4(path)` to your FastFrames file reader.
 - If you have a compute_energy_from_wfm(wfm) function in your repo, this script will
   attempt to import it if you place it in a module named `alpha_energy` with that name.
   Otherwise a built-in fast estimator is used.
"""

import os
import sys
import re
import argparse
import numpy as np
from glob import glob
import importlib.util
from RaTag.core.dataIO import load_alpha
from RaTag.alphas.wfm2spectra import alpha_peak

# ---------------------------
# User hook: read ch4 waveforms
# ---------------------------
def read_fastframe_ch4(path):
    """
    Read a FastFrames file and return an array of channel-4 waveforms shape (n_frames, n_samples).
    This function MUST be adapted to match your actual FastFrames file format.

    Fallback behavior implemented here:
      - If file is a numpy .npz or .npy that contains 'ch4' or 'ch_4', it loads that.
      - If none matches, raise RuntimeError asking the user to implement the hook.

    Replace/modify this function to call your project's FastFrames reader.
    """
    # try npz / npy convenience
    try:
        if path.endswith('.npz'):
            a = np.load(path)
            for key in ('ch4', 'ch_4', 'ch_04', 'ch_4_wfm'):
                if key in a:
                    return np.asarray(a[key])
        if path.endswith('.npy'):
            arr = np.load(path)
            # assume arr is (n_frames, n_samples) or dict-like
            if isinstance(arr, np.ndarray):
                return arr
    except Exception:
        pass

    # As a last fallback, try to see if there's a text file with per-frame simplified data
    raise RuntimeError(
        "read_fastframe_ch4() cannot read file format. "
        "Please adapt this function to use your FastFrames reader."
    )

# ---------------------------
# Try to import user compute function if available
# ---------------------------
def try_import_user_energy_func():
    """
    Try to import compute_energy_from_wfm(wfm) from a module named `alpha_energy.py`
    placed in the same folder or on PYTHONPATH. If not present, returns None.
    """
    try:
        spec = importlib.util.find_spec("alpha_peak")
        if spec is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "compute_energy_from_wfm"):
            return mod.compute_energy_from_wfm
    except Exception:
        return None
    return None

# ---------------------------
# Fast fallback energy estimator (very simple)
# ---------------------------
def fast_energy_estimator(wfm, pre_samples=50, L=10, R=60, sample_dt=1.0):
    """
    Very fast estimator: baseline median, argmax, integrate window [peak-L, peak+R].
    Returns energy-like float (units are raw ADC * sample_dt). You can scale to keV later.
    """
    w = np.asarray(wfm, dtype=np.float32)
    if w.size == 0:
        return 0.0
    baseline = np.median(w[:pre_samples]) if w.size > pre_samples else np.median(w)
    w = w - baseline
    peak_idx = int(np.argmax(w))
    start = max(0, peak_idx - L)
    end = min(w.size, peak_idx + R)
    # integrate (simple sum); multiply by dt if you want physical units
    charge = np.sum(w[start:end]) * sample_dt
    return float(charge)

# ---------------------------
# Binary writers
# ---------------------------
def write_chunk_8b(entries, out_path):
    """
    entries: iterable of (unique_id:int, energy:float)
    Writes binary file containing structured array dtype [('id','<u4'), ('E','<f4')]
    """
    arr = np.array(entries, dtype=[('id','<u4'), ('E','<f4')])
    # ensure sorted by id (should be already)
    arr.sort(order='id')
    arr.tofile(out_path)

def write_chunk_6b(entries, out_path, scale=0.1):
    """
    entries: iterable of (unique_id:int, energy:float)
    energy is quantized by scale (keV per LSB) to uint16.
    """
    out_list = []
    for uid, E in entries:
        q = int(round(E / scale))
        if q < 0 or q > 0xFFFF:
            raise ValueError(f"energy {E} out of range after scaling (scale={scale}) for uid={uid}")
        out_list.append((np.uint32(uid), np.uint16(q)))
    arr = np.array(out_list, dtype=[('id','<u4'), ('E','<u2')])
    arr.sort(order='id')
    arr.tofile(out_path)

# ---------------------------
# Helper utilities
# ---------------------------
def unique_id(file_seq, frame_idx):
    return int(file_seq) * 64 + int(frame_idx)

def parse_file_seq_from_name(fname):
    """
    Robust parser for your DAQ filenames of the form:
      RUN18_20251104_Gate100_Anode2000_P2_29Wfm_Ch4.wfm
      ..._P2_5Wfm_Ch4.wfm
    Returns the integer after 'P2_' and before 'Wfm' (e.g. 29, 5, ...).
    Raises ValueError if it cannot parse.
    """
    base = os.path.basename(fname)
    # Primary pattern: P2_<digits>Wfm  (covers the provided examples)
    m = re.search(r'P2_(\d+)Wfm', base)
    if m:
        return int(m.group(1))
    # Secondary pattern: fallback to any <digits>Wfm (more permissive)
    m2 = re.search(r'_(\d+)Wfm', base)
    if m2:
        return int(m2.group(1))
    # Last-resort: try last group of digits but ensure we don't pick the channel (Ch4)
    # This is purposely conservative - prefer explicit patterns above.
    m3 = re.search(r'(\d+)(?!.*\d)', base)
    if m3:
        # If this is '4' and the filename contains 'Ch4' we must avoid mis-pick.
        last = int(m3.group(1))
        if 'Ch4' in base:
            # if the filename ends with ..._Ch4.wfm and the last digits are 4,
            # do not accept that; consider this a parsing failure.
            if last == 4:
                raise ValueError(f"Refusing ambiguous parse for filename (trailing '4' likely channel): {base}")
        return last
    raise ValueError(f"Cannot parse file_seq from filename: {base}")

# ---------------------------
# Main processing
# ---------------------------
def build_parsed_list(files, manifest_path=None):
    """
    Build deterministic list of (file_seq, path).
    - If manifest_path provided, it must be a text file with one filename per line.
      Manifest entries are matched against basenames in `files` (so it can be short).
      Each manifest line is assigned sequential file_seq starting at 0 in the manifest order.
    - Otherwise we parse file_seq from filenames using parse_file_seq_from_name().
      If parsed file_seq values contain duplicates or gaps, we will detect and raise
      an informative error (safer than silently overwriting).
    Returns list of (file_seq, full_path) sorted by file_seq.
    """
    files_sorted = sorted(files)
    if manifest_path:
        manifest = []
        with open(manifest_path, 'r') as fh:
            for lineno, line in enumerate(fh, start=1):
                fn = line.strip()
                if not fn:
                    continue
                # try absolute/relative match first
                if os.path.isabs(fn):
                    if fn not in files_sorted:
                        raise ValueError(f"Manifest entry {fn} not found among input files")
                    manifest.append(fn)
                else:
                    # match basename
                    matches = [p for p in files_sorted if os.path.basename(p) == fn]
                    if not matches:
                        raise ValueError(f"Manifest basename {fn} not found among input files")
                    if len(matches) > 1:
                        print(f"Warning: manifest basename {fn} matched multiple files; using first match")
                    manifest.append(matches[0])
        files_sorted = manifest

    # Try parsing file_seq from filenames
    parsed = []
    dup_check = {}
    for p in files_sorted:
        try:
            fs = parse_file_seq_from_name(p)
        except ValueError as e:
            # If parsing fails for any file, abort and ask user to provide a manifest or use sequential fallback.
            raise RuntimeError(f"Failed to parse file_seq from filename '{p}': {e}\n"
                               f"Provide a manifest file or rename files to include 'P2_<N>Wfm' pattern.") from e
        if fs in dup_check:
            # Duplicate parsed file_seq detected: this is likely a filename pattern collision (bad).
            # Instead of silently proceeding, raise a helpful error.
            raise RuntimeError(f"Duplicate file_seq {fs} parsed for files:\n"
                               f"  {dup_check[fs]}\n  {p}\n"
                               "Either provide a manifest file to specify ordering, or rename files to avoid duplicate indices.")
        dup_check[fs] = p
        parsed.append((fs, p))

    # Sort by parsed file_seq before returning
    parsed.sort(key=lambda x: x[0])
    return parsed

def build_file_list(indir, pattern="*"):
    # naive: list files and sort
    files = sorted(glob(os.path.join(indir, pattern)))
    if not files:
        raise RuntimeError(f"No files found in {indir} with pattern {pattern}")
    return files


def writer_main(indir, outdir, files_per_chunk=100, fmt="8b", scale=0.1, pattern="*"):
    # create outdir
    os.makedirs(outdir, exist_ok=True)

    files = build_file_list(indir, pattern)
    parsed = build_parsed_list(files, manifest_path=None) 
    # build list of (file_seq, path) deterministically
    parsed = []
    for p in files:
        try:
            fs = parse_file_seq_from_name(p)
        except ValueError:
            # fallback: use index in sorted list
            fs = len(parsed)
        parsed.append((fs, p))
    # sort by file_seq
    parsed.sort(key=lambda x: x[0])

    # Optionally try to load user function
    # user_fn = try_import_user_energy_func()
    user_fn = alpha_peak
    if user_fn:
        print("Using user-provided compute_energy_from_wfm()")
    else:
        print("Using fast internal estimator (fast_energy_estimator). To use your own, provide alpha_energy.compute_energy_from_wfm(wfm) on PYTHONPATH.")

    # process in chunks of files_per_chunk
    i = 0
    total_files = len(parsed)
    while i < total_files:
        chunk_files = parsed[i : i + files_per_chunk]
        start_fs = chunk_files[0][0]
        end_fs = chunk_files[-1][0]
        out_fname = f"energy_map_f{start_fs:06d}-f{end_fs:06d}.bin"
        out_path = os.path.join(outdir, out_fname)
        entries = []
        # iterate files in chunk
        for file_seq, path in chunk_files:
            # read ch4 frames
            try:
                # ch4_arr = read_fastframe_ch4(path)  # expected shape (n_frames, n_samples)
                wf = load_alpha(path)
                # print('DEBUG: ', wf.v)
                ch4_arr = wf.v
            except Exception as e:
                print(f"ERROR reading {path}: {e}", file=sys.stderr)
                raise

            # sanity: if ch4_arr is 1D assume 1 frame
            if ch4_arr.ndim == 1:
                ch4_arr = ch4_arr[np.newaxis, :]

            n_frames = ch4_arr.shape[0]
            for frame_idx in range(n_frames):
                wfm = ch4_arr[frame_idx]
                if user_fn:
                    try:
                        energy = float(user_fn(wfm))
                    except Exception as e:
                        print(f"User fn failed on {path} frame {frame_idx}: {e}", file=sys.stderr)
                        energy = float(fast_energy_estimator(wfm))
                else:
                    energy = float(fast_energy_estimator(wfm))
                uid = unique_id(file_seq, frame_idx)
                entries.append((uid, energy))
        # write chunk file
        if fmt == "8b":
            write_chunk_8b(entries, out_path)
        elif fmt == "6b":
            write_chunk_6b(entries, out_path, scale=scale)
        else:
            raise ValueError("fmt must be '8b' or '6b'")

        print(f"Wrote {out_path}  ({len(entries)} entries)")
        i += files_per_chunk

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True, help="Directory with FastFrames files")
    p.add_argument("--outdir", required=True, help="Directory to write mapping chunk files")
    p.add_argument("--files-per-chunk", type=int, default=100, help="How many FastFrame files per mapping chunk (e.g. 10 or 100)")
    p.add_argument("--format", choices=("8b","6b"), default="8b", help="Mapping format: '8b' or '6b'")
    p.add_argument("--scale", type=float, default=0.1, help="Scale for 6b mode (keV per LSB).")
    p.add_argument("--pattern", default="*", help="glob pattern to find FastFrame files in indir")
    args = p.parse_args()
    writer_main(args.indir, args.outdir, files_per_chunk=args.files_per_chunk, fmt=args.format, scale=args.scale, pattern=args.pattern)
