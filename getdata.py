# Last update 5th Jan 2026 Sepehr
"""
Download + prepare Kaggle dataset into local ./data folder:

- Raw files (zip + extracted):         data/raw/
- Manifest (hashes, sizes):            data/meta/manifest.json
- Preprocessed outputs (chunked):      data/preprocessed/parquet/<file_stem>/part-000000.parquet

Dataset: vijayveersingh/1-2m-brain-signal-data

Requirements:
  pip install kaggle pandas pyarrow

"""

from __future__ import annotations

import argparse
import csv
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
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


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


def unzip_all(raw_dir: Path) -> None:
    for z in raw_dir.glob("*.zip"):
        target = raw_dir / z.stem
        target.mkdir(parents=True, exist_ok=True)
        print(f"[info] Unzipping {z.name} -> {target}")
        shutil.unpack_archive(str(z), str(target))


def discover_files(raw_dir: Path) -> List[Path]:
    """
    Discover likely data files. This dataset commonly contains .txt (e.g., IN.txt).
    """
    exts = {
        ".csv", ".tsv", ".parquet",
        ".txt", ".dat", ".data",
        ".npy", ".npz", ".mat", ".h5", ".hdf5"
    }
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def summarize_raw_tree(raw_dir: Path, max_show: int = 30) -> None:
    """
    Helpful debug: show extension counts and largest files.
    """
    all_files = [p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    if not all_files:
        print("[warn] No files found under raw directory.")
        return

    ext_counts = {}
    for p in all_files:
        ext = p.suffix.lower() or "<noext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print("[info] Raw contents extension counts (top):")
    for ext, cnt in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {ext:>8} : {cnt}")

    all_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    print(f"[info] Largest files under {raw_dir}:")
    for p in all_files[:max_show]:
        print(f"  - {p.relative_to(raw_dir)}  ({p.stat().st_size / (1024**2):.2f} MB)")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(paths: Paths, files: List[Path]) -> Path:
    manifest = {"dataset": DATASET_SLUG, "files": []}
    for f in files:
        rel = f.relative_to(paths.base)
        manifest["files"].append(
            {"path": str(rel), "bytes": f.stat().st_size, "sha256": sha256_file(f)}
        )
    out = paths.meta / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] Wrote manifest: {out}")
    return out


def infer_delimiter(sample_line: str) -> Optional[str]:
    """
    Try to infer delimiter. Returns:
      - ',' or '\\t' or ';' etc if detected
      - None if it looks whitespace-separated (use sep=r'\\s+')
    """
    s = sample_line.strip()

    # If it's clearly comma/tab separated:
    comma = s.count(",")
    tab = s.count("\t")
    semi = s.count(";")
    pipe = s.count("|")

    # Heuristic: choose the strongest signal
    counts = {",": comma, "\t": tab, ";": semi, "|": pipe}
    best_delim, best_count = max(counts.items(), key=lambda kv: kv[1])

    if best_count >= 2:
        return best_delim

    # Try csv.Sniffer on common delimiters
    try:
        dialect = csv.Sniffer().sniff(s, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        # If no clear delimiter, assume whitespace
        return None


def deterministic_split(row_id: int, train_pct: int = 80, val_pct: int = 10) -> str:
    h = hashlib.blake2b(str(row_id).encode("utf-8"), digest_size=2).digest()
    bucket = int.from_bytes(h, "little") % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def text_to_parquet_chunked(
    path: Path,
    out_dir: Path,
    chunksize: int,
    encoding: Optional[str] = None,
    add_split: bool = True,
) -> Tuple[int, Path]:
    """
    Convert CSV/TSV/TXT-like tabular file to chunked parquet.
    Handles delimiter inference (comma/tab/semicolon/pipe/whitespace).
    """
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow") from e

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read first non-empty line to infer delimiter
    with path.open("r", encoding=encoding or "utf-8", errors="ignore") as f:
        first_line = ""
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            if line.strip():
                first_line = line
                break

    delim = infer_delimiter(first_line) if first_line else None
    if delim is None:
        sep = r"\s+"
        print(f"[info] Inferred delimiter: whitespace (sep='\\s+') for {path.name}")
    else:
        sep = delim
        printable = "\\t" if delim == "\t" else delim
        print(f"[info] Inferred delimiter: '{printable}' for {path.name}")

    total = 0
    row_id = 0

    reader = pd.read_csv(
        path,
        sep=sep,
        chunksize=chunksize,
        engine="python",      # supports regex sep like \s+
        header=None,          # safe default when schema is unknown
        low_memory=False,
        encoding=encoding,
    )

    for i, chunk in enumerate(reader):
        chunk.insert(0, "row_id", range(row_id, row_id + len(chunk)))
        if add_split:
            chunk["split"] = [deterministic_split(r) for r in chunk["row_id"].astype(int).tolist()]

        # Light downcast
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
    ap.add_argument("--data_dir", type=Path, default=Path("data"), help='Base folder (default: "data")')
    ap.add_argument("--force_download", action="store_true", help="Force Kaggle re-download")
    ap.add_argument("--to_parquet", action="store_true", help="Convert largest detected text-like file to parquet")
    ap.add_argument("--chunksize", type=int, default=50_000, help="Rows per chunk")
    ap.add_argument("--no_split", action="store_true", help="Do not add split column")
    ap.add_argument("--encoding", type=str, default=None, help="Optional encoding override")
    args = ap.parse_args()

    paths = make_paths(args.data_dir)

    print(f"[info] Downloading {DATASET_SLUG} -> {paths.raw}")
    kaggle_download_dataset(DATASET_SLUG, paths.raw, force=args.force_download)

    unzip_all(paths.raw)
    summarize_raw_tree(paths.raw)

    files = discover_files(paths.raw)
    if not files:
        raise RuntimeError(
            f"No known data files found under {paths.raw}.\n"
            "Check the extension summary above and add the missing extension(s) to discover_files()."
        )

    print("[info] Candidate data files (largest first):")
    for f in files[:10]:
        print(f"  - {f.relative_to(paths.base)} ({f.stat().st_size / (1024**2):.2f} MB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    write_manifest(paths, files)

    # Preprocess: convert the largest "text-like" file if requested
    if args.to_parquet:
        text_like = [p for p in files if p.suffix.lower() in (".csv", ".tsv", ".txt", ".dat", ".data")]
        if not text_like:
            print("[warn] --to_parquet was set, but no text-like file (.csv/.tsv/.txt/.dat) found.")
            print("[done] Raw data is available in data/raw/.")
            return

        main_file = text_like[0]
        out = paths.preprocessed / "parquet" / main_file.stem
        text_to_parquet_chunked(
            path=main_file,
            out_dir=out,
            chunksize=args.chunksize,
            encoding=args.encoding,
            add_split=not args.no_split,
        )

        print("\n[done] Example load:")
        print("  import pandas as pd")
        print(f"  df = pd.read_parquet(r'{out / 'part-000000.parquet'}')")
        print("  print(df.head())")
    else:
        print("\n[done] Raw data is ready in data/raw/.")
        print("Tip: rerun with --to_parquet to convert the big .txt into fast parquet parts.")

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
