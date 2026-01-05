# Last update 5th Jan 2026 Sepehr
"""
Download + prepare Kaggle dataset into local ./data folder:

- Raw files (zip + extracted):        data/raw/
- Manifest (hashes, sizes):           data/meta/manifest.json
- Preprocessed outputs (e.g. parquet):data/preprocessed/

Dataset: vijayveersingh/1-2m-brain-signal-data

Requirements:
  pip install kaggle pandas pyarrow

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

DATASET_SLUG = "vijayveersingh/1-2m-brain-signal-data"


@dataclass
class Paths:
    base: Path
    raw: Path
    preprocessed: Path
    meta: Path


def make_paths(base_dir: Path) -> Paths:
    base = base_dir.resolve()
    raw = base / "raw"
    preprocessed = base / "preprocessed"
    meta = base / "meta"
    for p in (raw, preprocessed, meta):
        p.mkdir(parents=True, exist_ok=True)
    return Paths(base=base, raw=raw, preprocessed=preprocessed, meta=meta)


def have_kaggle_creds() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def kaggle_download_dataset(dataset_slug: str, dest_dir: Path, force: bool = False) -> None:
    if not have_kaggle_creds():
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "1) Kaggle -> Account -> API -> Create New API Token\n"
            "2) Put kaggle.json at ~/.kaggle/kaggle.json\n"
            "3) chmod 600 ~/.kaggle/kaggle.json\n"
            "OR set env vars KAGGLE_USERNAME and KAGGLE_KEY."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir)]
    if force:
        cmd.append("--force")
    run_cmd(cmd)


def unzip_all(raw_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    for z in raw_dir.glob("*.zip"):
        target = raw_dir / z.stem
        target.mkdir(parents=True, exist_ok=True)
        print(f"[info] Unzipping {z.name} -> {target}")
        shutil.unpack_archive(str(z), str(target))
        extracted.append(target)
    return extracted


def discover_data_files(raw_dir: Path) -> List[Path]:
    exts = {".csv", ".tsv", ".parquet"}
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(paths: Paths, data_files: List[Path]) -> Path:
    manifest = {"dataset": DATASET_SLUG, "files": []}
    for f in data_files:
        rel = f.relative_to(paths.base)
        manifest["files"].append(
            {"path": str(rel), "bytes": f.stat().st_size, "sha256": sha256_file(f)}
        )
    out = paths.meta / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] Wrote manifest: {out}")
    return out


def deterministic_split(row_id: int, train_pct: int = 80, val_pct: int = 10) -> str:
    # Stable bucket 0..99
    h = hashlib.blake2b(str(row_id).encode("utf-8"), digest_size=2).digest()
    bucket = int.from_bytes(h, "little") % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def csv_to_parquet_chunked(
    csv_path: Path,
    out_dir: Path,
    chunksize: int,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    add_split: bool = True,
) -> Tuple[int, Path]:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow") from e

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)

    if sep is None:
        sep = "\t" if csv_path.suffix.lower() == ".tsv" else ","

    print(f"[info] Converting to parquet in chunks: {csv_path} (chunksize={chunksize})")
    total = 0
    row_id = 0

    reader = pd.read_csv(
        csv_path,
        sep=sep,
        chunksize=chunksize,
        low_memory=False,
        encoding=encoding,
    )

    for i, chunk in enumerate(reader):
        chunk.insert(0, "row_id", range(row_id, row_id + len(chunk)))
        if add_split:
            chunk["split"] = [deterministic_split(r) for r in chunk["row_id"].astype(int).tolist()]

        # Mild dtype optimization (safe-ish, helps size/speed)
        for col in chunk.select_dtypes(include=["int64"]).columns:
            if col != "row_id":
                chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
        for col in chunk.select_dtypes(include=["float64"]).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast="float")

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        out_file = out_dir / f"part-{i:06d}.parquet"
        pq.write_table(table, out_file, compression="zstd")

        total += len(chunk)
        row_id += len(chunk)

        if (i + 1) % 10 == 0:
            print(f"[progress] wrote {total:,} rows -> {out_dir}")

    print(f"[ok] Parquet conversion complete: {total:,} rows -> {out_dir}")
    return total, out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help='Local folder name to store everything (default: "data")',
    )
    ap.add_argument("--force_download", action="store_true", help="Force Kaggle re-download")
    ap.add_argument("--to_parquet", action="store_true", help="Convert CSV/TSV -> chunked Parquet")
    ap.add_argument("--chunksize", type=int, default=50_000, help="Rows per chunk for parquet")
    ap.add_argument("--no_split", action="store_true", help="Do not add split column")
    ap.add_argument("--encoding", type=str, default=None, help="Optional CSV encoding override")
    args = ap.parse_args()

    paths = make_paths(args.data_dir)

    # 1) Download to data/raw (keep raw always)
    print(f"[info] Downloading {DATASET_SLUG} -> {paths.raw}")
    kaggle_download_dataset(DATASET_SLUG, paths.raw, force=args.force_download)

    # 2) Extract zips (still inside data/raw)
    unzip_all(paths.raw)

    # 3) Find data files
    files = discover_data_files(paths.raw)
    if not files:
        raise RuntimeError(f"No CSV/TSV/Parquet found under {paths.raw}")

    print("[info] Found data files (largest first):")
    for f in files[:10]:
        print(f"  - {f.relative_to(paths.base)} ({f.stat().st_size / (1024**2):.2f} MB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    # 4) Manifest
    write_manifest(paths, files)

    # 5) Preprocess (optional)
    main_file = files[0]
    if args.to_parquet and main_file.suffix.lower() in (".csv", ".tsv"):
        out = paths.preprocessed / "parquet" / main_file.stem
        csv_to_parquet_chunked(
            csv_path=main_file,
            out_dir=out,
            chunksize=args.chunksize,
            sep=None,
            encoding=args.encoding,
            add_split=not args.no_split,
        )
        print("\n[done] Example load:")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_parquet(r'{out / 'part-000000.parquet'}')")
    else:
        print("\n[done] Raw data is ready in data/raw/.")
        print("Tip: run with --to_parquet for faster downstream processing if your main file is CSV/TSV.")

    print(f"\n[ok] All outputs are under: {paths.base}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[error] Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
