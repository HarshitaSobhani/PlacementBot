"""
Generates a synthetic Campus Recruitment dataset matching the schema of the
Kaggle "Campus Recruitment" dataset (Ben Roshan), since Kaggle credentials
are not available in this environment. Placement probability is correlated
with academic percentages and work experience so the model has real signal.

Run: python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

gender = rng.choice(["M", "F"], size=N, p=[0.56, 0.44])
ssc_b = rng.choice(["Central", "Others"], size=N)
hsc_b = rng.choice(["Central", "Others"], size=N)
hsc_s = rng.choice(["Commerce", "Science", "Arts"], size=N, p=[0.45, 0.40, 0.15])
degree_t = rng.choice(["Sci&Tech", "Comm&Mgmt", "Others"], size=N, p=[0.40, 0.50, 0.10])
workex = rng.choice(["Yes", "No"], size=N, p=[0.34, 0.66])
specialisation = rng.choice(["Mkt&HR", "Mkt&Fin"], size=N, p=[0.45, 0.55])

ssc_p = np.clip(rng.normal(67, 10, N), 40, 100)
hsc_p = np.clip(rng.normal(66, 11, N), 37, 100)
degree_p = np.clip(rng.normal(66, 7, N), 50, 98)
etest_p = np.clip(rng.normal(72, 13, N), 50, 100)
mba_p = np.clip(rng.normal(62, 5.5, N), 50, 77)

academic_avg = (ssc_p + hsc_p + degree_p) / 3
workex_bonus = np.where(workex == "Yes", 8, 0)

score = (
    0.35 * (academic_avg - 65)
    + 0.15 * (etest_p - 70)
    + 0.12 * (mba_p - 62)
    + workex_bonus
    + rng.normal(0, 4, N)
)
prob_placed = 1 / (1 + np.exp(-(score - 2) / 3))
status = np.where(rng.uniform(size=N) < prob_placed, "Placed", "Not Placed")

salary = np.where(
    status == "Placed",
    np.round(
        np.clip(
            rng.normal(280000, 60000, N) + (academic_avg - 65) * 2000, 180000, 940000
        ),
        -3,
    ),
    np.nan,
)

df = pd.DataFrame(
    {
        "sl_no": np.arange(1, N + 1),
        "gender": gender,
        "ssc_p": ssc_p.round(2),
        "ssc_b": ssc_b,
        "hsc_p": hsc_p.round(2),
        "hsc_b": hsc_b,
        "hsc_s": hsc_s,
        "degree_p": degree_p.round(2),
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p.round(2),
        "specialisation": specialisation,
        "mba_p": mba_p.round(2),
        "status": status,
        "salary": salary,
    }
)

df.to_csv("data/placement_data.csv", index=False)
print(f"Wrote data/placement_data.csv with {len(df)} rows")
print(df["status"].value_counts(normalize=True))
