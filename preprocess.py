# Last update 5th Jan 2026 Sepehr
"""
Preprocess MindBigData (Kaggle: 1-2m-brain-signal-data) into ONE common dataset.

Input (raw):
  data/raw/1-2m-brain-signal-data/**/EP*.txt
  data/raw/1-2m-brain-signal-data/**/MW.txt
  data/raw/1-2m-brain-signal-data/**/MU.txt
  data/raw/1-2m-brain-signal-data/**/IN.txt

Output (preprocessed) - NOTE: under data/preprocessed/ (your existing folder):
  data/preprocessed/<out_subdir>/
    meta.json
    index.parquet
    chunks/
      chunk_00000.npz
      chunk_00001.npz
      ...

What it does:
- Parses MindBigData lines (tab-separated fields)
- Groups per (device,event) to create one multichannel epoch
- Downsamples higher-rate signals to lowest nominal fs (128 Hz)
- Standardizes channel labels and uses a common channel union
- Saves chunked NPZ + parquet index + meta.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


# -----------------------------
# Dataset-specific config
# -----------------------------

DURATION_SEC = 2.0

# Nominal sampling rates (as described by dataset docs)
NOMINAL_FS_BY_DEVICE = {"MW": 512, "EP": 128, "MU": 220, "IN": 128}

CHANNELS_BY_DEVICE_RAW = {
    "MW": ["FP1"],
    "EP": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "MU": ["TP9", "FP1", "FP2", "TP10"],
    "IN": ["AF3", "AF4", "T7", "T8", "PZ"],
}

CHANNEL_CANONICAL_MAP = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "PZ": "Pz",
}

DEVICE_TO_INT = {"MW": 0, "EP": 1, "MU": 2, "IN": 3}
INT_TO_DEVICE = {v: k for k, v in DEVICE_TO_INT.items()}


def canonical_ch(name: str) -> str:
    n = name.strip().replace('"', "").replace("'", "")
    n = n.upper()
    return CHANNEL_CANONICAL_MAP.get(n, n)


def build_common_channels() -> List[str]:
    all_ch = []
    for dev, chs in CHANNELS_BY_DEVICE_RAW.items():
        for ch in chs:
            all_ch.append(canonical_ch(ch))
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
    preprocessed_dir: Path
    out_dir: Path
    chunks_dir: Path


def make_paths(data_dir: Path, out_subdir: str) -> Paths:
    """
    IMPORTANT: writes into existing data/preprocessed/
    """
    data_dir = data_dir.resolve()
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    out_dir = preprocessed_dir / out_subdir
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        data_dir=data_dir,
        raw_dir=raw_dir,
        preprocessed_dir=preprocessed_dir,
        out_dir=out_dir,
        chunks_dir=chunks_dir,
    )


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

    parts = line.split("\t")
    if len(parts) < 7:
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
        data_str = parts[6]
        data = np.fromstring(data_str, sep=",", dtype=np.float32)
        if data.size == 0:
            return None
        channel = canonical_ch(channel_raw)
        return Record(rid, event, device, channel, code, size, data)
    except Exception:
        return None


# -----------------------------
# Resampling / shaping
# -----------------------------

def downsample_or_pad_to_target(x: np.ndarray, target_n: int) -> np.ndarray:
    n = int(x.size)
    if n == target_n:
        return x.astype(np.float32, copy=False)

    if n > target_n:
        y = resample_poly(x, up=target_n, down=n).astype(np.float32, copy=False)
        if y.size > target_n:
            y = y[:target_n]
        elif y.size < target_n:
            y = np.pad(y, (0, target_n - y.size), mode="edge")
        return y

    # n < target_n: pad (no upsampling)
    return np.pad(x.astype(np.float32, copy=False), (0, target_n - n), mode="edge")


def make_epoch(
    channels_to_series: Dict[str, np.ndarray],
    target_n: int,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
# File discovery
# -----------------------------

def find_device_files(raw_root: Path) -> Dict[str, List[Path]]:
    return {
        "EP": sorted(raw_root.rglob("MindBigData-EP*/*.txt")),
        "MW": sorted(raw_root.rglob("MindBigData-MW*/*.txt")),
        "MU": sorted(raw_root.rglob("MindBigData-MU*/*.txt")),
        "IN": sorted(raw_root.rglob("MindBigData-IN*/*.txt")),
    }


def expected_channels_for_device(dev: str) -> List[str]:
    return [canonical_ch(c) for c in CHANNELS_BY_DEVICE_RAW.get(dev, [])]


# -----------------------------
# Chunked writer
# -----------------------------

class ChunkWriter:
    def __init__(self, chunks_dir: Path, chunk_size: int):
        self.chunks_dir = chunks_dir
        self.chunk_size = chunk_size
        self._chunk_idx = 0
        self._global_epoch = 0

        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self._device: List[int] = []
        self._event: List[int] = []
        self._rid: List[int] = []
        self._present: List[np.ndarray] = []

        self.index_rows: List[Dict] = []

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

        X = np.stack(self._X, axis=0).astype(np.float32, copy=False)
        y = np.asarray(self._y, dtype=np.int16)
        device = np.asarray(self._device, dtype=np.int8)
        event = np.asarray(self._event, dtype=np.int64)
        rid = np.asarray(self._rid, dtype=np.int64)
        present = np.stack(self._present, axis=0).astype(np.bool_, copy=False)

        out_path = self.chunks_dir / f"chunk_{self._chunk_idx:05d}.npz"
        np.savez(out_path, X=X, y=y, device=device, event=event, rid=rid, present=present)

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

        self._X.clear()
        self._y.clear()
        self._device.clear()
        self._event.clear()
        self._rid.clear()
        self._present.clear()


# -----------------------------
# Processing logic
# -----------------------------

def preprocess_device_file(
    path: Path,
    dev: str,
    target_n: int,
    writer: ChunkWriter,
    max_events: Optional[int] = None,
) -> int:
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
        writer.add(
            X=X,
            y=int(current_code) if current_code is not None else -999,
            device=DEVICE_TO_INT.get(dev, -1),
            event=int(event_id),
            rid=int(current_rid) if current_rid is not None else -1,
            present=present,
        )
        n_emitted += 1
        buf = {}
        current_code = None
        current_rid = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec is None or rec.device != dev:
                continue

            if current_event is None:
                current_event = rec.event

            if rec.event != current_event:
                flush_event(current_event)
                current_event = rec.event
                if max_events is not None and n_emitted >= max_events:
                    break

            current_code = rec.code if current_code is None else current_code
            current_rid = rec.rid if current_rid is None else current_rid
            if rec.channel in exp_channels:
                buf[rec.channel] = rec.data

            if exp_channels and len(buf) >= len(exp_channels):
                flush_event(current_event)
                if max_events is not None and n_emitted >= max_events:
                    break

    if current_event is not None and buf:
        flush_event(current_event)

    return n_emitted


def write_meta(out_dir: Path, target_fs: int, target_n: int):
    meta = {
        "duration_sec": DURATION_SEC,
        "target_fs_hz": target_fs,
        "target_n_times": target_n,
        "common_channels": COMMON_CHANNELS,
        "device_to_int": DEVICE_TO_INT,
        "nominal_fs_by_device": NOMINAL_FS_BY_DEVICE,
        "note": (
            "Epoch = one event grouped by (device,event). "
            "Missing channels are filled with 0; `present` mask indicates which channels exist."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# -----------------------------
# MNE helper
# -----------------------------

def chunk_to_mne_epochs(npz_path: Path, meta_path: Path):
    import mne
    meta = json.loads(meta_path.read_text())
    sfreq = float(meta["target_fs_hz"])
    ch_names = meta["common_channels"]
    with np.load(npz_path, allow_pickle=False) as z:
        X = z["X"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.EpochsArray(X, info, verbose=False)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out_subdir", type=str, default="mindbigdata_common",
                    help='Folder inside data/preprocessed/ (default: mindbigdata_common)')
    ap.add_argument("--chunk_size", type=int, default=2048, help="Epochs per NPZ chunk")
    ap.add_argument("--max_events_per_file", type=int, default=0, help="Debug limit (0 = no limit)")
    args = ap.parse_args()

    paths = make_paths(args.data_dir, out_subdir=args.out_subdir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)

    # lowest nominal fs => 128 Hz
    target_fs = min(NOMINAL_FS_BY_DEVICE.values())
    target_n = int(round(DURATION_SEC * target_fs))  # 256

    write_meta(paths.out_dir, target_fs=target_fs, target_n=target_n)

    device_files = find_device_files(paths.raw_dir)
    writer = ChunkWriter(paths.chunks_dir, chunk_size=args.chunk_size)

    total = 0
    max_events = args.max_events_per_file if args.max_events_per_file > 0 else None

    for dev, files in device_files.items():
        if not files:
            print(f"[warn] No files for device {dev} under {paths.raw_dir}")
            continue

        for fp in files:
            print(f"[info] Processing {dev}: {fp}")
            n = preprocess_device_file(fp, dev=dev, target_n=target_n, writer=writer, max_events=max_events)
            total += n
            print(f"[ok] {fp.name}: emitted {n:,} epochs")

    writer.flush()

    idx = pd.DataFrame(writer.index_rows)
    idx_path = paths.out_dir / "index.parquet"
    idx.to_parquet(idx_path, index=False)

    print(f"[ok] Output root: {paths.out_dir}")
    print(f"[ok] meta.json:    {paths.out_dir / 'meta.json'}")
    print(f"[ok] index:       {idx_path} ({len(idx):,} epochs)")
    print(f"[ok] chunks:      {paths.chunks_dir}")
    print(f"[done] epoch shape: ({len(COMMON_CHANNELS)}, {target_n}) | total epochs: {total:,}")


if __name__ == "__main__":
    main()
