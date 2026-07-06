import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.explain_utils import describe_feature, split_transformed_name
from api.schema import PredictionResponse, PlacementInput, TopFeature

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="PlacementLens API", description="Predicts student placement likelihood with SHAP-backed explanations.")
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
    row = payload.model_dump()
    row["skill_avg"] = (row["Coding_Skills"] + row["Communication_Skills"] + row["Soft_Skills_Rating"]) / 3
    return pd.DataFrame([row])[NUMERIC + CATEGORICAL]


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
    label = "Placed" if proba >= 0.5 else "Not Placed"

    shap_values = _explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    contributions = shap_values[0]
    expected_value = _explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if np.ndim(expected_value) else float(expected_value)

    order = np.argsort(-np.abs(contributions))[:3]
    top_features = []
    for i in order:
        contrib = float(contributions[i])
        direction = "increased" if contrib > 0 else "decreased"
        base_prob = 1 / (1 + np.exp(-expected_value))
        bumped_prob = 1 / (1 + np.exp(-(expected_value + contrib)))
        impact_pct = round((bumped_prob - base_prob) * 100, 2)
        clean_name, _category = split_transformed_name(FEATURE_NAMES[i], NUMERIC, CATEGORICAL)
        top_features.append(
            TopFeature(
                feature=clean_name,
                direction=direction,
                approx_probability_impact_pct=impact_pct,
                explanation=describe_feature(FEATURE_NAMES[i], direction, raw_row, NUMERIC, CATEGORICAL),
            )
        )

    return PredictionResponse(
        prediction=label,
        probability_placed=round(proba, 4),
        top_features=top_features,
    )


# Serve the static frontend from the same app/port so a single container
# (Render or HF Spaces) hosts both the API and the demo UI.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
