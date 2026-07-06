"""
SHAP explainability: global summary plot + one individual waterfall plot.
Run after model/train.py. Run: python model/explain.py
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
os.makedirs(ROOT / "results" / "shap", exist_ok=True)

model = joblib.load(ROOT / "model" / "best_model.pkl")
with open(ROOT / "model" / "metadata.json") as f:
    meta = json.load(f)

X_test = np.load(ROOT / "model" / "X_test_transformed.npy")
# A few hundred rows is plenty to visualize global SHAP behavior and keeps
# TreeExplainer fast on a 10k-row test split.
rng = np.random.default_rng(42)
if len(X_test) > 500:
    X_test = X_test[rng.choice(len(X_test), 500, replace=False)]
feature_names = meta["feature_names_transformed"]

explainer = shap.TreeExplainer(model) if meta["best_model"] != "Logistic Regression" else shap.LinearExplainer(model, X_test)
shap_values = explainer.shap_values(X_test)
# Binary classifiers may return a list of 2 arrays, or a single array with a
# trailing per-class axis (n_samples, n_features, n_classes) -- keep class 1.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[..., 1]

X_test_df = pd.DataFrame(X_test, columns=feature_names)

plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.tight_layout()
plt.savefig(ROOT / "results" / "shap" / "summary_plot.png", dpi=120, bbox_inches="tight")
plt.close()

# Individual explanation for the first test row.
idx = 0
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[1] if len(np.shape(expected_value)) else expected_value

explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=expected_value,
    data=X_test_df.iloc[idx].values,
    feature_names=feature_names,
)
plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig(ROOT / "results" / "shap" / "waterfall_example_0.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved results/shap/summary_plot.png and waterfall_example_0.png")
