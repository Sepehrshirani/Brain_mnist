# Brain_mnist

This project builds a unified, ML-ready EEG/brain-signal dataset (and later, models) from the Kaggle dataset:

- **Kaggle dataset:** “1-2m-brain-signal-data” by vijayveersingh
- It contains MindBigData text files (e.g., `EP1.01.txt`, `MW.txt`, `MU.txt`, `IN.txt`) distributed via Kaggle.

This README describes what has been implemented so far and what each stage produces. It will be expanded as the project grows.

---

## Data origin

All raw recordings come from the Kaggle dataset above. The raw files are text-based and include multiple “device” collections:

- **EP**: Emotiv EPOC (multi-channel)
- **IN**: InteraXon (multi-channel)
- **MU**: Muse (4 channels)
- **MW**: MindWave (single channel)

The project’s preprocessing step standardizes these into a single common dataset representation.

---

## Directory layout (after running the scripts)

After running the pipeline, your local project will typically contain:

```
Brain_mnist/
  getdata.py
  preprocess.py
  labeldist.py
  data/
    raw/
    meta/
    preprocessed/
```

### `data/raw/` (source-of-truth)
This folder holds the dataset exactly as downloaded (and extracted). It includes:
- The original Kaggle zip file
- The extracted dataset folder
- The large MindBigData `.txt` files (EP / IN / MU / MW)

Example layout:
- `data/raw/1-2m-brain-signal-data.zip`
- `data/raw/1-2m-brain-signal-data/`
  - `MindBigData-EP-v1.0/EP1.01.txt`
  - `MindBigData-MW-v1.0/MW.txt`
  - `MindBigData-MU-v1.0/MU.txt`
  - `MindBigData-IN-v1.06/IN.txt`

### `data/meta/` (download metadata / reproducibility)
This folder holds metadata derived from the raw download, currently:
- `manifest.json`: file paths, sizes, and SHA-256 hashes for the raw files.  
  This helps verify files are complete and unchanged.

### `data/preprocessed/` (derived datasets)
This folder holds derived outputs produced by preprocessing. Current output:
- `data/preprocessed/mindbigdata_common/`
  - `meta.json`
  - `index.parquet`
  - `chunks/` containing `chunk_00000.npz`, `chunk_00001.npz`, …

---

## Stages implemented so far

### Stage 1 — Download & organize raw data (`getdata.py`)
**What happens**
- Downloads the Kaggle dataset into `data/raw/`
- Extracts/unzips the dataset into `data/raw/…`
- Scans discovered raw files and writes a reproducibility manifest into `data/meta/manifest.json`

**What you should see afterward**
- Raw zip + extracted files under `data/raw/`
- `data/meta/manifest.json`

---

### Stage 2 — Build a common preprocessed dataset (`preprocess.py`)
**Goal**
Create one dataset format that is consistent across devices and easy to load later using scientific tooling.

**What happens**
- Reads the raw MindBigData `.txt` recordings from `data/raw/…`
- Defines one “epoch” (trial) by grouping raw records by `(device, event)`:
  - Each epoch corresponds to a single trial/event for a device
  - The epoch is assembled from that device’s available channels for that event
- Enforces channel-label consistency:
  - Channel names are canonicalized (for example, typical label normalization like `FP1 → Fp1`, `PZ → Pz`)
  - A fixed global channel list is used (currently **19 channels**, the union across devices)
  - If a device does not contain a given channel, that channel is filled with zeros for that epoch and marked as missing in a per-epoch mask
- Downsamples to a single sampling rate:
  - Devices with higher sampling rate are resampled down to the **lowest nominal sampling rate (128 Hz)**
  - Epoch length is standardized to **2 seconds**, resulting in **256 samples** per channel at 128 Hz
- Writes results in “chunks”:
  - Instead of one huge file, epochs are saved into chunked `.npz` files for memory-safe loading

**What you should see afterward**
- `data/preprocessed/mindbigdata_common/meta.json`  
  Dataset metadata (target sampling rate, epoch length, channel list, device mapping, etc.)
- `data/preprocessed/mindbigdata_common/index.parquet`  
  A global index with one row per epoch. It records where the epoch lives (which chunk + row) and includes label and device metadata.
- `data/preprocessed/mindbigdata_common/chunks/`  
  Files like `chunk_00000.npz` through `chunk_00091.npz` (the count depends on total epochs and chunk size)

**Chunk logic**
- Each chunk contains up to a fixed number of epochs (default was 2048 in this project)
- Total chunks ≈ ceiling(total_epochs / chunk_size)
- Chunks are numbered sequentially from 0 upward

**Labels / annotations**
- Each epoch has one label stored as `y`:
  - `y` is the “code” value from the raw dataset records
  - In practice, the digit classes are `0–9`
  - A special value `-1` appears in some devices and represents “non-digit / unknown / baseline” trials
- Label counts can be inspected using the next stage (`labeldist.py`)

---

### Stage 3 — Label distribution sanity check (`labeldist.py`)
**What happens**
- Reads `data/preprocessed/mindbigdata_common/index.parquet`
- Prints label counts overall and split by device

**What it tells you**
- Whether labels are balanced (or not)
- How many “-1” trials exist and in which devices
- Useful before training to decide whether to:
  - train only on `0–9`, or
  - include `-1` as an extra class, or
  - filter devices

---

## Current status / next steps (to be expanded)
As the project develops, this README will be extended with:
- dataset filtering strategies (e.g., digits-only vs include baseline, device-specific subsets)
- normalization and signal processing options (e.g., scaling, filtering)
- train/validation/test split strategy
- model training, evaluation, and results
