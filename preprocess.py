#!/usr/bin/env python3
"""
Preprocess MindBigData "1-2m-brain-signal-data" (Kaggle) into ONE common dataset.

Input (raw):
  data/raw/1-2m-brain-signal-data/**/EP*.txt
  data/raw/1-2m-brain-signal-data/**/MW.txt
  data/raw/1-2m-brain-signal-data/**/MU.txt
  data/raw/1-2m-brain-signal-data/**/IN.txt

Output (preprocessed):
  data/preprocessed_data/
    meta.json
    index.parquet
    chunks/
      chunk_00000.npz
      chunk_00001.npz
      ...

What it does:
- Parses the MindBigData format: id, event, device, channel, code, size, data (tab-separated fields) :contentReference[oaicite:1]{index=1}
- Groups per (device, event) to form a multichannel "epoch"
- Standardizes channel names and enforces a single global channel order (union of all device channels)
- Downsamples (only) higher-rate recordings to the lowest device sampling rate (128 Hz) :contentReference[oaicite:2]{index=2}
- Ensures every epoch has shape: (n_channels_common, n_times_target), filling missing channels with 0
- Saves chunked NPZ (easy for sklearn) + metadata + an index parquet (easy to filter/query)
- Provides helper functions to reload as numpy OR as MNE EpochsArray.

Dependencies:
  pip install numpy pandas scipy pyarrow mne

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


# -----------------------------
# Dataset-specific config
# -----------------------------

DURATION_SEC = 2.0

# Nominal sampling rates described by MindBigData (approx/“in theory”). :contentReference[oaicite:3]{index=3}
NOMINAL_FS_BY_DEVICE = {
    "MW": 512,
    "EP": 128,
    "MU": 220,
    "IN": 128,
}

# Channel lists described by MindBigData (10/20 positions). :contentReference[oaicite:4]{index=4}
CHANNELS_BY_DEVICE_RAW = {
    "MW": ["FP1"],
    "EP": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "MU": ["TP9", "FP1", "FP2", "TP10"],
    "IN": ["AF3", "AF4", "T7", "T8", "PZ"],
}

# Map raw channel names -> MNE-friendly canonical names
# (Consistent naming; especially PZ -> Pz, FP1/2 -> Fp1/Fp2)
CHANNEL_CANONICAL_MAP = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "PZ": "Pz",
    # Keep the rest as-is (AF3, F7, etc.) — these match standard 10/20 labels.
}

DEVICE_TO_INT = {"MW": 0, "EP": 1, "MU": 2, "IN": 3}
INT_TO_DEVICE = {v: k for k, v in DEVICE_TO_INT.items()}


def canonical_ch(name: str) -> str:
    n = name.strip().replace('"', "").replace("'", "")
    n = n.upper()  # normalize
    return CHANNEL_CANONICAL_MAP.get(n, n)


def build_common_channels() -> List[str]:
    # Union of all device channels, canonicalized, then a stable order.
    all_ch = []
    for dev, chs in CHANNELS_BY_DEVICE_RAW.items():
        for ch in chs:
            all_ch.append(canonical_ch(ch))
    # Unique preserving order
    seen = set()
    common = []
    for ch in all_ch:
        if ch not in seen:
            seen.add(ch)
            common.append(ch)
    return common


COMMON_CHANNELS = build_common_channels()
COMMON_CH_TO_IDX = {ch: i for i, ch in enumerate(COMMON_CHANNELS)}


# -----------------------------
# Paths / IO
# -----------------------------

@dataclass
class Paths:
    data_dir: Path
    raw_dir: Path
    out_dir: Path
    chunks_dir: Path


def make_paths(data_dir: Path) -> Paths:
    data_dir = data_dir.resolve()
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "preprocessed_data"
    chunks_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return Paths(data_dir=data_dir, raw_dir=raw_dir, out_dir=out_dir, chunks_dir=chunks_dir)


# -----------------------------
# Parsing
# -----------------------------

@dataclass
class Record:
    rid: int
    event: int
    device: str
    channel: str
    code: int
    size: int
    data: np.ndarray  # float32


def parse_line(line: str) -> Optional[Record]:
    line = line.strip()
    if not line:
        return None

    # Spec says fields separated by TAB. :contentReference[oaicite:5]{index=5}
    parts = line.split("\t")
    if len(parts) < 7:
        # fallback: split on whitespace
        parts = line.split()
        if len(parts) < 7:
            return None

    try:
        rid = int(parts[0])
        event = int(parts[1])
        device = parts[2].strip()
        channel_raw = parts[3].strip()
        code = int(parts[4])
        size = int(parts[5])

        # Data is comma-separated values
        data_str = parts[6]
        # Some lines can be huge; fromstring is fast.
        data = np.fromstring(data_str, sep=",", dtype=np.float32)

        # If parsing failed to read anything, skip
        if data.size == 0:
            return None

        channel = canonical_ch(channel_raw)

        return Record(
            rid=rid,
            event=event,
            device=device,
            channel=channel,
            code=code,
            size=size,
            data=data,
        )
    except Exception:
        return None


# -----------------------------
# Resampling / shaping
# -----------------------------

def downsample_or_pad_to_target(x: np.ndarray, target_n: int) -> np.ndarray:
    """
    Requirement: downsample higher sampling rate to lowest fs.
    We interpret this as:
      - if x is longer than target -> downsample to target
      - if x is shorter than target -> pad (no upsampling)
    """
    n = int(x.size)
    if n == target_n:
        return x.astype(np.float32, copy=False)

    if n > target_n:
        # Resample to exactly target length using polyphase method.
        # Using up=target_n, down=n simplifies by gcd internally and yields ~target_n output.
        y = resample_poly(x, up=target_n, down=n).astype(np.float32, copy=False)
        # resample_poly may return slightly off length; fix deterministically:
        if y.size > target_n:
            y = y[:target_n]
        elif y.size < target_n:
            y = np.pad(y, (0, target_n - y.size), mode="edge")
        return y

    # n < target_n: pad (no upsample)
    return np.pad(x.astype(np.float32, copy=False), (0, target_n - n), mode="edge")


def make_epoch(
    channels_to_series: Dict[str, np.ndarray],
    target_n: int,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n_channels_common, target_n)
      present_mask: (n_channels_common,)
    """
    X = np.full((len(COMMON_CHANNELS), target_n), fill_value, dtype=np.float32)
    present = np.zeros((len(COMMON_CHANNELS),), dtype=np.bool_)

    for ch, series in channels_to_series.items():
        idx = COMMON_CH_TO_IDX.get(ch)
        if idx is None:
            continue
        X[idx] = downsample_or_pad_to_target(series, target_n)
        present[idx] = True

    return X, present


# -----------------------------
# Main preprocessing loop
# -----------------------------

def find_device_files(raw_root: Path) -> Dict[str, List[Path]]:
    """
    Locate expected device files under data/raw/**.
    """
    files = {
        "EP": sorted(raw_root.rglob("MindBigData-EP*/*.txt")),
        "MW": sorted(raw_root.rglob("MindBigData-MW*/*.txt")),
        "MU": sorted(raw_root.rglob("MindBigData-MU*/*.txt")),
        "IN": sorted(raw_root.rglob("MindBigData-IN*/*.txt")),
    }

    # Keep only the main known names if present
    # EP can be EP1.01.txt; MW is MW.txt; MU is MU.txt; IN is IN.txt
    # If multiples, keep them all (EP folder sometimes has just one).
    return files


def expected_channels_for_device(dev: str) -> List[str]:
    raw_list = CHANNELS_BY_DEVICE_RAW.get(dev, [])
    return [canonical_ch(c) for c in raw_list]


def preprocess_device_file(
    path: Path,
    dev: str,
    target_n: int,
    chunk_size: int,
    chunk_writer,
    max_events: Optional[int] = None,
) -> int:
    """
    Reads one device file and emits complete events (epochs) into chunk_writer.
    Returns number of emitted epochs.
    """
    exp_channels = set(expected_channels_for_device(dev))
    n_emitted = 0

    current_event: Optional[int] = None
    current_code: Optional[int] = None
    current_rid: Optional[int] = None
    buf: Dict[str, np.ndarray] = {}

    def flush_event(event_id: int):
        nonlocal n_emitted, buf, current_code, current_rid
        if not buf:
            return

        X, present = make_epoch(buf, target_n=target_n, fill_value=0.0)

        chunk_writer.add(
            X=X,
            y=int(current_code) if current_code is not None else -999,
            device=DEVICE_TO_INT.get(dev, -1),
            event=int(event_id),
            rid=int(current_rid) if current_rid is not None else -1,
            present=present,
        )
        n_emitted += 1

        # reset buffer
        buf = {}
        current_code = None
        current_rid = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec is None:
                continue

            if rec.device != dev:
                # Skip if something odd is inside the file
                continue

            # New event?
            if current_event is None:
                current_event = rec.event

            if rec.event != current_event:
                flush_event(current_event)
                current_event = rec.event

                if max_events is not None and n_emitted >= max_events:
                    break

            # Accumulate
            current_code = rec.code if current_code is None else current_code
            current_rid = rec.rid if current_rid is None else current_rid
            if rec.channel in exp_channels:
                buf[rec.channel] = rec.data

            # If we already have all channels for this device, flush early
            if exp_channels and len(buf) >= len(exp_channels):
                flush_event(current_event)

                if max_events is not None and n_emitted >= max_events:
                    break

    # flush last partial event
    if current_event is not None and buf:
        flush_event(current_event)

    return n_emitted


class ChunkWriter:
    def __init__(self, chunks_dir: Path, chunk_size: int):
        self.chunks_dir = chunks_dir
        self.chunk_size = chunk_size

        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self._device: List[int] = []
        self._event: List[int] = []
        self._rid: List[int] = []
        self._present: List[np.ndarray] = []

        self._chunk_idx = 0
        self.index_rows: List[Dict] = []
        self._global_epoch = 0

    def add(self, X: np.ndarray, y: int, device: int, event: int, rid: int, present: np.ndarray):
        self._X.append(X)
        self._y.append(y)
        self._device.append(device)
        self._event.append(event)
        self._rid.append(rid)
        self._present.append(present)

        if len(self._X) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self._X:
            return

        X = np.stack(self._X, axis=0).astype(np.float32, copy=False)  # (n, ch, t)
        y = np.asarray(self._y, dtype=np.int16)
        device = np.asarray(self._device, dtype=np.int8)
        event = np.asarray(self._event, dtype=np.int64)
        rid = np.asarray(self._rid, dtype=np.int64)
        present = np.stack(self._present, axis=0).astype(np.bool_, copy=False)  # (n, ch)

        out_path = self.chunks_dir / f"chunk_{self._chunk_idx:05d}.npz"
        np.savez(
            out_path,
            X=X,
            y=y,
            device=device,
            event=event,
            rid=rid,
            present=present,
        )

        # Build index entries
        n = X.shape[0]
        for i in range(n):
            self.index_rows.append(
                {
                    "epoch_id": self._global_epoch + i,
                    "chunk": out_path.name,
                    "row": i,
                    "y": int(y[i]),
                    "device": int(device[i]),
                    "device_name": INT_TO_DEVICE.get(int(device[i]), "UNK"),
                    "event": int(event[i]),
                    "rid": int(rid[i]),
                }
            )

        self._global_epoch += n
        self._chunk_idx += 1

        # reset buffers
        self._X.clear()
        self._y.clear()
        self._device.clear()
        self._event.clear()
        self._rid.clear()
        self._present.clear()


def write_meta(out_dir: Path, target_fs: int, target_n: int):
    meta = {
        "dataset": "MindBigData (Kaggle: 1-2m-brain-signal-data)",
        "duration_sec": DURATION_SEC,
        "target_fs_hz": target_fs,
        "target_n_times": target_n,
        "common_channels": COMMON_CHANNELS,
        "device_to_int": DEVICE_TO_INT,
        "nominal_fs_by_device": NOMINAL_FS_BY_DEVICE,
        "note": (
            "Each epoch is one multi-channel event assembled by grouping (device,event). "
            "Channels missing for a device are filled with 0; use `present` mask to know which exist."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# -----------------------------
# Convenience loaders (optional)
# -----------------------------

def load_chunk_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def chunk_to_mne_epochs(chunk: Dict[str, np.ndarray], sfreq: float, channel_names: List[str]):
    """
    Convert one loaded chunk dict to an MNE EpochsArray.
    (Requires `pip install mne`)
    """
    import mne

    X = chunk["X"]  # (n_epochs, n_channels, n_times)
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, verbose=False)
    return epochs


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"), help='Base data folder (default: "data")')
    ap.add_argument("--chunk_size", type=int, default=2048, help="Epochs per output chunk NPZ")
    ap.add_argument("--max_events_per_file", type=int, default=0, help="Debug: limit epochs per device file (0 = no limit)")
    args = ap.parse_args()

    paths = make_paths(args.data_dir)

    # Lowest sampling rate among devices (nominal): 128 Hz. :contentReference[oaicite:6]{index=6}
    target_fs = min(NOMINAL_FS_BY_DEVICE.values())
    target_n = int(round(DURATION_SEC * target_fs))  # 2 sec * 128 Hz = 256

    write_meta(paths.out_dir, target_fs=target_fs, target_n=target_n)

    device_files = find_device_files(paths.raw_dir)
    chunk_writer = ChunkWriter(paths.chunks_dir, chunk_size=args.chunk_size)

    total_epochs = 0
    max_events = args.max_events_per_file if args.max_events_per_file > 0 else None

    for dev, files in device_files.items():
        if not files:
            print(f"[warn] No files found for device {dev} under {paths.raw_dir}")
            continue

        for fp in files:
            print(f"[info] Processing {dev}: {fp}")
            n = preprocess_device_file(
                path=fp,
                dev=dev,
                target_n=target_n,
                chunk_size=args.chunk_size,
                chunk_writer=chunk_writer,
                max_events=max_events,
            )
            print(f"[ok] Emitted epochs from {fp.name}: {n:,}")
            total_epochs += n

    # Flush any remainder
    chunk_writer.flush()

    # Write index parquet
    idx = pd.DataFrame(chunk_writer.index_rows)
    idx_path = paths.out_dir / "index.parquet"
    idx.to_parquet(idx_path, index=False)
    print(f"[ok] Wrote index: {idx_path} ({len(idx):,} epochs)")
    print(f"[ok] Wrote meta:  {paths.out_dir / 'meta.json'}")
    print(f"[ok] Chunks in:  {paths.chunks_dir}")
    print(f"[done] Total epochs: {total_epochs:,} | shape per epoch: ({len(COMMON_CHANNELS)}, {target_n})")


if __name__ == "__main__":
    main()
