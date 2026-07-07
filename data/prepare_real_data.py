"""
Combines the real Kaggle "Student Placement Dataset" (sonalshinde123,
https://www.kaggle.com/datasets/sonalshinde123/student-placement-dataset)
train/test CSVs in data/raw/ into a single modeling dataset, dropping the
row-identifier column. train.csv and test.csv have disjoint Student_IDs and
identical schema, so they're safe to concatenate and re-split ourselves.

Run: python data/prepare_real_data.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")

df = pd.concat([train, test], ignore_index=True).drop(columns=["Student_ID"])
df.to_csv(ROOT / "data" / "placement_data.csv", index=False)

print(f"Wrote data/placement_data.csv with {len(df)} rows")
print(df["Placement_Status"].value_counts(normalize=True))
