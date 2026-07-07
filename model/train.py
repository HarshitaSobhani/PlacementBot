"""
Feature engineering + model comparison (Logistic Regression, Random Forest,
XGBoost) for PlacementLens, trained on the real Kaggle "Student Placement
Dataset". Selects the best model by F1 and saves it, along with the fitted
preprocessor, to model/. Also writes results/model_comparison.md.

Run: python model/train.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = [
    "Age",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Communication_Skills",
    "Aptitude_Test_Score",
    "Soft_Skills_Rating",
    "Certifications",
    "Backlogs",
    "skill_avg",
]
CATEGORICAL = ["Gender", "Degree", "Branch"]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features import compute_skill_avg  # noqa: E402 (needs ROOT on sys.path first)

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "model", exist_ok=True)

df = pd.read_csv(ROOT / "data" / "placement_data.csv")

# Derived feature: overall skill strength across the three 0-10 rated skill
# dimensions (coding, communication, soft skills). Shared with api/main.py
# via features.py so training and serving can't compute this differently.
df["skill_avg"] = compute_skill_avg(df)

# Missing-value strategy (documented in README): numeric -> median impute,
# categorical -> most-frequent impute. This real dataset has zero missing
# values, but the pipeline stays defensive in case that changes upstream.
X = df[NUMERIC + CATEGORICAL]
y = (df["Placement_Status"] == "Placed").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            NUMERIC,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
                ]
            ),
            CATEGORICAL,
        ),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

Xt_train = preprocessor.fit_transform(X_train, y_train)
Xt_test = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out().tolist()

rows = []
fitted = {}
for name, clf in models.items():
    clf.fit(Xt_train, y_train)
    fitted[name] = clf
    preds = clf.predict(Xt_test)
    proba = clf.predict_proba(Xt_test)[:, 1]
    rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, proba),
        }
    )

results_df = (
    pd.DataFrame(rows)
    .sort_values("F1", ascending=False, kind="stable")
    .reset_index(drop=True)
)
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]

print(results_df.to_string(index=False))

with open(ROOT / "results" / "model_comparison.md", "w") as f:
    f.write("# Model Comparison\n\n")
    f.write(results_df.round(4).to_markdown(index=False))
    f.write(f"\n\n**Selected model (highest F1): {best_name}**\n")

# Machine-readable copy of the same table, for CI's training-regression check
# (parsing the markdown table there would be fragile).
results_df.to_json(
    ROOT / "results" / "model_comparison.json", orient="records", indent=2
)

joblib.dump(best_model, ROOT / "model" / "best_model.pkl")
joblib.dump(preprocessor, ROOT / "model" / "preprocessor.pkl")
with open(ROOT / "model" / "metadata.json", "w") as f:
    json.dump(
        {
            "best_model": best_name,
            "numeric_features": NUMERIC,
            "categorical_features": CATEGORICAL,
            "feature_names_transformed": feature_names,
        },
        f,
        indent=2,
    )

# Keep a held-out test split for SHAP examples downstream, plus a background
# sample from the *training* distribution (needed as the SHAP masker for the
# linear model case -- using the explained instance itself as its own
# background collapses every contribution to ~0).
Xt_test_dense = Xt_test if not hasattr(Xt_test, "toarray") else Xt_test.toarray()
Xt_train_dense = Xt_train if not hasattr(Xt_train, "toarray") else Xt_train.toarray()
np.save(ROOT / "model" / "X_test_transformed.npy", Xt_test_dense)
np.save(ROOT / "model" / "background_transformed.npy", Xt_train_dense[:100])
X_test.reset_index(drop=True).to_csv(ROOT / "model" / "X_test_raw.csv", index=False)
y_test.reset_index(drop=True).to_csv(ROOT / "model" / "y_test.csv", index=False)

print(f"\nBest model: {best_name} -> saved to model/best_model.pkl")
