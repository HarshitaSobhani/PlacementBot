# PlacementLens

**30-second pitch:** An end-to-end ML product that predicts whether a student will
be placed by a campus recruiter, with per-prediction SHAP explanations ("why did the
model say this?"), served behind a FastAPI endpoint and a zero-build HTML/JS
frontend. Covers the full lifecycle: EDA → feature engineering → model comparison →
explainability → API → deploy config.

## Problem

Campus placement cells want to know, ahead of interviews, which students are at risk
of not being placed and *why* — so they can target mentoring/prep resources. This is
a binary classification problem (Placed / Not Placed) on a tabular, mixed
categorical+numeric dataset, where a black-box "risk score" isn't actionable without
an explanation attached to it.

## Approach

1. **EDA** (`eda/eda.py`) — class balance, CGPA/aptitude-score distributions by
   outcome, feature correlations, and placement rate by internships / backlogs /
   branch. Figures saved to `eda/figures/`.
2. **Feature engineering** (`model/train.py`) — one derived feature (`skill_avg`,
   the mean of coding/communication/soft-skills ratings), median/most-frequent
   imputation (documented below), one-hot encoding for categoricals, standard
   scaling for numerics.
3. **Model comparison** — Logistic Regression (baseline), Random Forest, XGBoost,
   evaluated on a held-out 20% test split. Selected by **F1**, not accuracy (see
   Tradeoffs).
4. **Explainability** (`model/explain.py`) — SHAP `TreeExplainer`/`LinearExplainer`
   depending on the winning model; a global summary plot and one individual
   waterfall plot, saved to `results/shap/`.
5. **API** (`api/main.py`) — FastAPI `POST /predict`, Pydantic-validated input,
   returns prediction + probability + top-3 SHAP-ranked contributing features in
   plain language.
6. **Frontend** (`frontend/`) — single `index.html` + `script.js`, no build step,
   posts to `/predict` and renders the result and contributing factors.

## Dataset

Uses the real **"Student Placement Dataset"**
([Kaggle, sonalshinde123](https://www.kaggle.com/datasets/sonalshinde123/student-placement-dataset)),
downloaded by the user and merged from its `train.csv` (45,000 rows) +
`test.csv` (5,000 rows, disjoint student IDs, identical schema) into a single
50,000-row modeling file via `data/prepare_real_data.py`. Columns: `Age, Gender,
Degree, Branch, CGPA, Internships, Projects, Coding_Skills,
Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications,
Backlogs, Placement_Status`. No missing values in either file.

`data/generate_synthetic_data.py` is kept as a **fallback only** (a different,
smaller schema modeled on the classic Kaggle "Campus Recruitment" dataset) for
cases where no real dataset is available — it is not part of the active
pipeline now that real data is in use.

**Reproduce:**
```bash
# put the Kaggle download's extracted train.csv / test.csv into data/raw/, then:
python data/prepare_real_data.py
```

## Results

Real numbers from `model/train.py` on a 20%-held-out test split, 50,000 rows total
(see `results/model_comparison.md`):

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|:--------------------|---------:|----------:|-------:|-------:|--------:|
| Random Forest        |   1.0000 |    1.0000 | 1.0000 | 1.0000 |  1.0000 |
| XGBoost               |   1.0000 |    1.0000 | 1.0000 | 1.0000 |  1.0000 |
| Logistic Regression   |   0.8668 |    0.8240 | 0.8044 | 0.8141 |  0.9380 |

**Selected: Random Forest** (tied for highest F1 with XGBoost; Random Forest is
listed first).

**Read this with real caution — a perfect 1.0 F1 is a red flag, not a triumph.**
Fitting a depth-3 decision tree to just the numeric features already recovers
94.3% accuracy with a handful of clean threshold splits (e.g. `Projects <= 3.5`,
`Coding_Skills <= 4.5`, `CGPA <= 6.49`, `Communication_Skills <= 4.5`, `Backlogs
<= 1.5`). That kind of sharp, fully-deterministic decision boundary — recovered
almost exactly by an unpruned Random Forest / XGBoost — is the signature of a
**rule-generated label**, not organic recruiter decisions. In other words: this
"real" Kaggle dataset is very likely itself synthetically labeled by its
uploader via a fixed formula over these features, with little or no noise. The
100% test score confirms the tree ensembles found that formula exactly; it does
not mean student placement is a solved, deterministic real-world problem —
messy human hiring decisions almost never separate this cleanly. Logistic
Regression's 0.81 F1 is the more informative number here: it shows the true
boundary is non-linear (the threshold/AND-OR structure above), which a linear
model structurally can't fully capture, while tree ensembles can.

If you swap in a dataset with genuinely noisy, human-judgment-driven labels
(the original small Kaggle "Campus Recruitment" set behaves this way), expect
F1/ROC-AUC to land well below 1.0 for every model — that would be the more
realistic result to trust.

## Design Decisions

- **Missing values**: none exist in this dataset, but the pipeline still applies
  median imputation (numeric) and most-frequent imputation (categorical) via
  `SimpleImputer`, so it's robust if a dataset with real gaps is swapped in.
- **`Student_ID` dropped** from features: it's a row identifier with no
  predictive meaning (dropped in `data/prepare_real_data.py`).
- **Binary one-hot columns use `drop='if_binary'`**: without it, a binary column
  like `Gender` produces two one-hot columns (`Gender_Male`, `Gender_Female`)
  that carry identical information, so SHAP would (correctly, but confusingly)
  surface both as separate "top contributing factors" for the same underlying
  fact.
- **Per-prediction SHAP → probability-impact conversion is an approximation**:
  SHAP values are additive in log-odds space; converting a single feature's
  contribution to a probability-percentage-point delta
  (`sigmoid(base+contribution) - sigmoid(base)`) ignores interaction effects
  with the other features. It's a reasonable approximation for a short
  human-readable explanation, not a precise decomposition.
- **SHAP plots computed on a 500-row sample** of the 10,000-row test split
  (`model/explain.py`), not the full set — plenty for a stable summary plot,
  and meaningfully faster than running `TreeExplainer` over all 10k rows.
- **Single-container deploy**: the FastAPI app also mounts `frontend/` as static
  files, so one process/port serves both the API and the demo UI — simpler for a
  free-tier Render service or a single HF Space.
- **Input bounds match the real training data's range** (see `api/schema.py`),
  not arbitrary caps (e.g. `Internships` is capped at 5, since training data
  only had 0–3) — the model has never seen inputs outside this range, so
  rejecting them at the API boundary is more honest than silently
  extrapolating and returning a confident-looking prediction anyway.
- **`requirements.txt` is pinned to exact versions** (`pip freeze`), not loose
  `>=` bounds, because a committed `model/best_model.pkl` is a scikit-learn/
  xgboost pickle — those aren't guaranteed compatible across library (or even
  Python) versions, so "whatever installs later" can silently break loading it.

## Why F1 over accuracy

The dataset has real class imbalance (~64% Not Placed / ~36% Placed). Accuracy on
an imbalanced set rewards a model that just leans toward the majority class; F1
balances precision and recall on the *positive* (Placed) class, which is the one a
placement cell actually cares about identifying correctly — both to catch students
likely to succeed and to flag students who need extra prep. ROC-AUC is reported
alongside as a threshold-independent view. (In this particular run F1 and accuracy
happen to agree at the top of the table, precisely because the underlying labels
are cleanly separable — see the caveat above.)

## What SHAP reveals that plain feature importance doesn't

A global feature-importance ranking (e.g. Random Forest's `.feature_importances_`)
tells you the model leans heavily on, say, `CGPA` — but it's the same number for
every student. It can't tell an individual "your CGPA helped you" vs. "your CGPA
hurt you", because raw importances have no sign and no per-instance value. SHAP
decomposes *each individual prediction* into signed, per-feature contributions
relative to a baseline, which is exactly what "why did I get this prediction"
requires — the API's `/predict` response returns those per-instance SHAP values
(top 3, with direction and an approximate probability-point impact), not the
static global ranking.

## How to run

Requires **Python 3.13** — `requirements.txt` is pinned to exact versions (via
`pip freeze`) that match the packages `model/best_model.pkl` was trained and
pickled with, since scikit-learn/xgboost pickles aren't guaranteed compatible
across library (or Python) versions.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
# (requirements-dev.txt = requirements.txt + pytest/httpx for testing.
#  Deploy configs install plain requirements.txt only, so the deployed
#  container doesn't carry test tooling.)

# 1. Prepare the real dataset (data/raw/train.csv + test.csv already included in this repo)
python data/prepare_real_data.py

# 2. EDA -> eda/figures/*.png
python eda/eda.py

# 3. Train + compare models -> results/model_comparison.md, model/best_model.pkl
python model/train.py

# 4. SHAP explainability -> results/shap/*.png
python model/explain.py

# 5. Serve the API (also serves the frontend at the same URL)
uvicorn api.main:app --reload --port 8000
# open http://127.0.0.1:8000
# API docs at http://127.0.0.1:8000/docs

# Tests (tests/test_api.py hits /health and /predict via FastAPI's TestClient;
# tests/test_explain_utils.py covers the SHAP-feature-name parsing logic)
pytest
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Age": 22, "Gender": "Male", "Degree": "B.Tech", "Branch": "CSE",
  "CGPA": 8.1, "Internships": 2, "Projects": 4, "Coding_Skills": 8,
  "Communication_Skills": 7, "Aptitude_Test_Score": 82, "Soft_Skills_Rating": 7,
  "Certifications": 2, "Backlogs": 0
}'
```

## Deployment

- **Render**: copy `deploy/render.yaml` to the repo root before connecting the repo
  (Render's Blueprint auto-detect looks for `render.yaml` at root), or manually create
  a Web Service with build command `pip install -r requirements.txt` and start
  command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- **Hugging Face Spaces (Docker SDK)**: copy `deploy/Dockerfile` to the repo root as
  `Dockerfile` before pushing to a Space (Spaces builds whatever `Dockerfile` sits at
  the repo root). It serves on port 7860 as Spaces expects.

Both are deploy-ready but untested against a live account in this environment — no
Render/HF credentials were available here.

## Structure

```
/data      real dataset (raw/ + merge script) and a synthetic-data fallback generator
/eda       EDA script + figures/
/model     training, model comparison, SHAP explainability, saved artifacts
/results   model_comparison.md, shap/ plots
/api       FastAPI app, Pydantic schema, SHAP-to-plain-language layer
/frontend  index.html + script.js (also served by the API at "/")
/deploy    render.yaml, Dockerfile
```
