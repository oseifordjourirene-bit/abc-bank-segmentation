"""
Make a smaller sample of the Santander data for the ABC Bank Segmentation pipeline.

Keeps the last six monthly snapshots (Dec 2015 - May 2016) for a random 50,000 customers,
so the same customers can be tracked month to month.

Usage (in the folder that holds train_ver2.csv or train_ver2.csv.zip):
    python make_sample.py
Output:
    santander_sample.csv  (about 300,000 rows, roughly 60 MB)
"""
import os
import pandas as pd

SOURCE = "train_ver2.csv" if os.path.exists("train_ver2.csv") else "train_ver2.csv.zip"
MONTHS = ["2015-12-28", "2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28", "2016-05-28"]
N_CUSTOMERS = 50_000
SEED = 42

# 1. Pick customers from the latest month, then keep their history
ids = pd.read_csv(SOURCE, usecols=["fecha_dato", "ncodpers"])
latest = ids.loc[ids["fecha_dato"] == MONTHS[-1], "ncodpers"].drop_duplicates()
chosen = set(latest.sample(n=min(N_CUSTOMERS, len(latest)), random_state=SEED))
print(f"Chose {len(chosen):,} customers from {MONTHS[-1]}")

# 2. Read the file in chunks so it fits in laptop memory
parts = []
for chunk in pd.read_csv(SOURCE, dtype=str, chunksize=500_000):
    keep = chunk["fecha_dato"].isin(MONTHS) & chunk["ncodpers"].astype(int).isin(chosen)
    parts.append(chunk[keep])

sample = pd.concat(parts, ignore_index=True)
sample.to_csv("santander_sample.csv", index=False)
print(f"Saved santander_sample.csv: {len(sample):,} rows, {sample.shape[1]} columns")
print(sample["fecha_dato"].value_counts().sort_index())
