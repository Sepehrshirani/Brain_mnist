from pathlib import Path
import pandas as pd

root = Path("data/preprocessed/mindbigdata_common")
idx = pd.read_parquet(root / "index.parquet")

print(idx["y"].value_counts().sort_index())
print(idx.groupby("device_name")["y"].value_counts().sort_index())
