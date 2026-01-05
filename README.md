# Brain_mnist

This repo builds a unified, ML-ready EEG/brain-signal dataset (and, later, models) from the Kaggle dataset:

- **Kaggle:** https://www.kaggle.com/datasets/vijayveersingh/1-2m-brain-signal-data  
- Contents are MindBigData text files (e.g., `EP1.01.txt`, `MW.txt`, `MU.txt`, `IN.txt`) distributed via Kaggle.

> This README is a living document and will be expanded as the project evolves (training code, evaluation, etc.).

---

## Local data layout

After running the scripts, you should have:

```
data/
  raw/                          # original Kaggle download + extracted files (source of truth)
  meta/                         # hashes/metadata for reproducibility
  preprocessed/                 # derived datasets saved here
```

### `data/raw/`
- `1-2m-brain-signal-data.zip` (the Kaggle download)
- `1-2m-brain-signal-data/` (extracted)
  - `MindBigData-EP-v1.0/EP1.01.txt`
  - `MindBigData-MW-v1.0/MW.txt`
  - `MindBigData-MU-v1.0/MU.txt`
  - `MindBigData-IN-v1.06/IN.txt`

### `data/meta/`
- `manifest.json`  
  Records file paths, sizes, and SHA-256 hashes for reproducibility (helps detect corruption/changes).

### `data/preprocessed/`
- `mindbigdata_common/` (created by `preprocess.py`)
  - `meta.json` (dataset metadata: channels, sampling rate, epoch length, etc.)
  - `index.parquet` (global index: maps each epoch to its chunk + row and stores labels/device info)
  - `chunks/` (chunked NPZ files: `chunk_00000.npz`, …)

---

## What the scripts do (so far)

### 1) `getdata.py` — download + organize raw data
**Purpose**
- Downloads the Kaggle dataset into `data/raw/` via the Kaggle API.
- Unzips it into `data/raw/1-2m-brain-signal-data/`.
- Scans the raw folder and writes `data/meta/manifest.json`.

**Run**
```bash
python getdata.py
```

**Notes**
- Requires Kaggle credentials (e.g., `~/.kaggle/kaggle.json` with correct permissions).

---

### 2) `preprocess.py` — build one common preprocessed dataset
**Purpose**
- Reads the raw MindBigData `.txt` files in `data/raw/...`.
- Produces a single unified dataset saved under:
  - `data/preprocessed/mindbigdata_common/`

**Main logic**
- **Epoch (trial) definition:** group raw lines by `(device, event)` to build one epoch containing all channels for that device.
- **Channel consistency:** map raw channel labels to canonical labels (e.g., `FP1 -> Fp1`, `PZ -> Pz`) and place channels into a fixed, global channel list (currently 19 channels = union across devices).
  - If a device does not have a channel, that channel is **filled with zeros** and marked as missing in `present`.
- **Downsampling:** recordings with higher sampling rate are resampled so every epoch matches the **lowest nominal sampling rate (128 Hz)**.
- **Fixed epoch length:** each epoch is stored as `(n_channels=19, n_times=256)` which corresponds to **2 seconds @ 128 Hz**.

**Output files**
- `meta.json`  
  Contains: `target_fs_hz`, `target_n_times`, `common_channels`, device mapping, etc.
- `index.parquet`  
  One row per epoch:
  - `epoch_id` (global id)
  - `chunk` (which NPZ file)
  - `row` (row index inside that chunk)
  - `y` (label)
  - `device_name` (EP/MW/MU/IN)
  - `event`, `rid` (ids from raw)
- `chunks/chunk_XXXXX.npz`  
  Each chunk stores up to `chunk_size` epochs (default 2048), so the dataset is split into multiple chunk files for memory-safe loading.

**Run**
```bash
python preprocess.py
```

---

### 3) `labeldist.py` — label distribution sanity check
**Purpose**
- Loads `data/preprocessed/mindbigdata_common/index.parquet`.
- Prints label counts overall and per device.

**Interpreting labels (`y`)**
- `y` is the **code field** from the raw MindBigData line format.
- In practice, you will see:
  - `0..9` = digit classes
  - `-1` = “non-digit / unknown / baseline” trials (present mostly in MU and MW; some in EP; none in IN in our current build)

**Run**
```bash
python labeldist.py
```

---

## Quick: how to load the data

### Load by index (recommended)
Use `index.parquet` to locate and load only what you need:

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("data/preprocessed/mindbigdata_common")
idx = pd.read_parquet(root / "index.parquet")

# Example: keep only digit trials
digits = idx[idx["y"].between(0, 9)]

# Load a small sample
sample = digits.sample(1000, random_state=0)

Xs, ys = [], []
for chunk_name, g in sample.groupby("chunk"):
    z = np.load(root / "chunks" / chunk_name)
    rows = g["row"].to_numpy()
    Xs.append(z["X"][rows])  # (n, 19, 256)
    ys.append(z["y"][rows])  # (n,)

X = np.concatenate(Xs, axis=0)
y = np.concatenate(ys, axis=0)
print(X.shape, y.shape)
```

### Use with MNE
```python
import json
import numpy as np
import mne
from pathlib import Path

root = Path("data/preprocessed/mindbigdata_common")
meta = json.loads((root / "meta.json").read_text())

z = np.load(root / "chunks" / "chunk_00000.npz")
X = z["X"]  # (n_epochs, 19, 256)

info = mne.create_info(
    ch_names=meta["common_channels"],
    sfreq=float(meta["target_fs_hz"]),
    ch_types="eeg",
)
epochs = mne.EpochsArray(X, info, verbose=False)
```

---

## Next steps (to be added)
Planned additions (README will be expanded as these are implemented):
- proper train/val/test split strategy (possibly device-wise and/or subject-wise if available)
- normalization/bandpass filtering options
- model training scripts (sklearn / deep learning)
- evaluation and reporting
