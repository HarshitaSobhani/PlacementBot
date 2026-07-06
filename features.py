"""Shared feature engineering.

Single source of truth for derived features used by both model/train.py
(training) and api/main.py (serving) -- if these ever drift apart, the model
would be trained on one feature definition and served with another (train/serve
skew), which fails silently rather than erroring.
"""
import pandas as pd


def compute_skill_avg(df: pd.DataFrame) -> pd.Series:
    return df[["Coding_Skills", "Communication_Skills", "Soft_Skills_Rating"]].mean(axis=1)
