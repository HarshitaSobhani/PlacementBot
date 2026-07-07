import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.explain_utils import describe_feature, split_transformed_name
from api.schema import PredictionResponse, PlacementInput, TopFeature
from features import compute_skill_avg

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="PlacementLens API",
    description="Predicts student placement likelihood with SHAP-backed explanations.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

model = joblib.load(ROOT / "model" / "best_model.pkl")
preprocessor = joblib.load(ROOT / "model" / "preprocessor.pkl")
metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
FEATURE_NAMES = metadata["feature_names_transformed"]
NUMERIC = metadata["numeric_features"]
CATEGORICAL = metadata["categorical_features"]

_is_tree_model = metadata["best_model"] != "Logistic Regression"
if _is_tree_model:
    _explainer = shap.TreeExplainer(model)
else:
    _background = np.load(ROOT / "model" / "background_transformed.npy")
    _explainer = shap.LinearExplainer(model, _background)


def _to_frame(payload: PlacementInput) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])
    df["skill_avg"] = compute_skill_avg(df)
    return df[NUMERIC + CATEGORICAL]


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["best_model"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PlacementInput):
    raw_row = payload.model_dump()
    X = _to_frame(payload)
    Xt = preprocessor.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    proba = float(model.predict_proba(Xt)[0, 1])
    label: Literal["Placed", "Not Placed"] = "Placed" if proba >= 0.5 else "Not Placed"

    shap_values = _explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    contributions = shap_values[0]
    expected_value_raw = _explainer.expected_value
    if isinstance(expected_value_raw, (list, np.ndarray)):
        ev_array = np.atleast_1d(np.asarray(expected_value_raw, dtype=float))
        expected_value = (
            float(ev_array[1]) if ev_array.shape[0] > 1 else float(ev_array[0])
        )
    else:
        expected_value = float(expected_value_raw)

    # Rank all features by |contribution|, but keep only the first one seen
    # per clean column name -- a multi-category column like Branch or Degree
    # produces several one-hot dummies, and without this guard two dummies
    # for the same underlying column could both land in the "top 3".
    ranked = np.argsort(-np.abs(contributions))
    top_features = []
    seen_columns = set()
    for i in ranked:
        clean_name, _category = split_transformed_name(
            FEATURE_NAMES[i], NUMERIC, CATEGORICAL
        )
        if clean_name in seen_columns:
            continue
        seen_columns.add(clean_name)

        contrib = float(contributions[i])
        direction: Literal["increased", "decreased"] = (
            "increased" if contrib > 0 else "decreased"
        )
        base_prob = 1 / (1 + np.exp(-expected_value))
        bumped_prob = 1 / (1 + np.exp(-(expected_value + contrib)))
        impact_pct = round((bumped_prob - base_prob) * 100, 2)
        top_features.append(
            TopFeature(
                feature=clean_name,
                direction=direction,
                approx_probability_impact_pct=impact_pct,
                explanation=describe_feature(
                    FEATURE_NAMES[i], direction, raw_row, NUMERIC, CATEGORICAL
                ),
            )
        )
        if len(top_features) == 3:
            break

    return PredictionResponse(
        prediction=label,
        probability_placed=round(proba, 4),
        top_features=top_features,
    )


# Serve the static frontend from the same app/port so a single container
# (Render or HF Spaces) hosts both the API and the demo UI.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
