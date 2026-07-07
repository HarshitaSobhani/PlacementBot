"""
CI regression guard: fails if the selected model's F1 on the held-out test
split drops below a floor. Catches a broken training run (e.g. a bad merge
that drops feature columns, corrupts the train/test split, or shuffles
labels) that would otherwise only surface as a quiet accuracy regression,
never an error -- run after model/train.py.

The floor (0.65) is well below this project's actual F1 (~0.73-1.0
depending on model/data) but well above what a genuinely broken pipeline
would produce on this imbalanced (~64/36) dataset.

Run: python scripts/check_training_floor.py
"""

import json
import sys
from pathlib import Path

F1_FLOOR = 0.65

ROOT = Path(__file__).resolve().parent.parent
metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
comparison = json.loads((ROOT / "results" / "model_comparison.json").read_text())

best_name = metadata["best_model"]
best_row = next(row for row in comparison if row["Model"] == best_name)
f1 = best_row["F1"]

print(f"Best model: {best_name}, F1={f1:.4f} (floor={F1_FLOOR})")
if f1 < F1_FLOOR:
    print(f"FAIL: F1 {f1:.4f} is below the regression floor of {F1_FLOOR}")
    sys.exit(1)

print("OK: training pipeline meets the F1 regression floor")
