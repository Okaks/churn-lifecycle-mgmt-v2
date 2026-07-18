"""
Batch Inference and Scoring.

Applies the winning model to every customer in latest_snapshot (each
customer's ACTUAL current state, not a training-time cutoff):

  1. Load latest_snapshot from Neon (small - 10,000 rows, fast).
  2. Apply the SAME preprocessing as training (src/feature_prep.py): drop
     admin/leakage columns, impute, one-hot encode, align to the training
     feature list, standardise using the saved scaler. No re-splitting -
     this is scoring, not training.
  3. Predict risk_tier for every customer - the model's direct 3-class
     prediction (Low/Medium/High Risk), NOT a probability-threshold bucket.
     churn_probability stores a WEIGHTED risk score (0 x P(Low) + 0.5 x
     P(Medium) + 1.0 x P(High)), not P(High Risk) alone - using P(High Risk)
     alone meant Medium Risk customers always scored near 0 (since the
     model correctly knows they're not High Risk), making them visually
     indistinguishable from genuinely safe Low Risk customers. The weighted
     version places confident Medium Risk customers near 0.5, reflecting
     where they actually sit on the risk spectrum.
  4. Compute SHAP values for every customer, for THEIR SPECIFIC PREDICTED
     CLASS (not an arbitrary fixed class) - matches src/explainability.py's
     per-class approach.
  5. Truncate the predictions and shap_values tables.
  6. Insert one row per customer into each.

This populates the data the dashboard reads for the Portfolio Overview,
Customer Risk Explorer, and SHAP Explainability tabs.

Run standalone (after modeling has been executed):
    python -m src.inference
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
from sqlalchemy import insert, text

from src.config import (
    ADMINISTRATIVE_COLUMNS,
    BEST_MODEL_FILE,
    CONTINUOUS_COLUMNS,
    FEATURE_LIST_FILE,
    LEAKAGE_COLUMNS,
    ONE_HOT_COLUMNS,
    RISK_LABEL_CLASSES,
    SCALER_FILE,
    TABLE_LATEST_SNAPSHOT,
    TABLE_PREDICTIONS,
    TABLE_SHAP_VALUES,
    TARGET_COLUMN,
)
from src.database import engine, predictions
from src.database import shap_values as shap_values_table
from src.feature_prep import align_to_feature_list, cast_booleans, encode_categoricals, impute_missing

HIGH_RISK_CLASS_IDX = RISK_LABEL_CLASSES.index("High Risk")

# Weighted risk score: 0 x P(Low) + 0.5 x P(Medium) + 1.0 x P(High), NOT just
# P(High Risk) alone. Using P(High Risk) alone meant Medium Risk customers -
# whom the model correctly knows are NOT High Risk - always scored near 0,
# indistinguishable from genuinely safe Low Risk customers. This weighted
# version places a confident Medium Risk customer near 0.5, a confident Low
# Risk customer near 0.0, and a confident High Risk customer near 1.0 -
# reflecting where they actually sit on the risk spectrum, not just "are
# they the worst category or not."
RISK_WEIGHTS_BY_CLASS = {"Low Risk": 0.0, "Medium Risk": 0.5, "High Risk": 1.0}
RISK_WEIGHTS = np.array([RISK_WEIGHTS_BY_CLASS[c] for c in RISK_LABEL_CLASSES])


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_artifacts() -> tuple:
    """Load winning model, fitted scaler, and feature name list."""
    model = joblib.load(BEST_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(FEATURE_LIST_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    print(
        f"Loaded {type(model).__name__} model, fitted scaler, "
        f"and {len(feature_names)} feature names."
    )
    return model, scaler, feature_names


def load_all_customers() -> pd.DataFrame:
    """Read every customer from latest_snapshot - each customer's actual
    current state, not a fixed training-time cutoff (small - fast)."""
    df = pd.read_sql(f'SELECT * FROM "{TABLE_LATEST_SNAPSHOT}"', engine)
    print(f"Loaded {len(df):,} customers from '{TABLE_LATEST_SNAPSHOT}'.")
    return df


# ---------------------------------------------------------------------------
# Preprocess for inference (no split, no re-fitting the scaler)
# ---------------------------------------------------------------------------
def preprocess_for_inference(
    df: pd.DataFrame, scaler, feature_names: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the same encoding + scaling as training (src/feature_prep.py),
    without splitting or refitting anything.

    Returns (customer_ids, X) where X is a DataFrame in canonical feature order.
    """
    customer_ids = df["customer_id"].to_numpy()

    X = df.drop(columns=ADMINISTRATIVE_COLUMNS + LEAKAGE_COLUMNS, errors="ignore")
    if TARGET_COLUMN in X.columns:
        X = X.drop(columns=[TARGET_COLUMN])

    X = impute_missing(X)
    X = cast_booleans(X)
    X = encode_categoricals(X, ONE_HOT_COLUMNS)
    X = align_to_feature_list(X, feature_names)

    X = X.copy()
    X[CONTINUOUS_COLUMNS] = scaler.transform(X[CONTINUOUS_COLUMNS])

    return customer_ids, X


# ---------------------------------------------------------------------------
# Risk tier assignment - the model's direct 3-class prediction, no thresholds
# ---------------------------------------------------------------------------
def predicted_tier_names(predicted_indices: np.ndarray) -> list[str]:
    return [RISK_LABEL_CLASSES[i] for i in predicted_indices]


# ---------------------------------------------------------------------------
# SHAP values for the full customer set
# ---------------------------------------------------------------------------
def compute_shap(model, X: pd.DataFrame, background_size: int = 200):
    """Compute SHAP values. Uses a small background sample for efficiency.
    Keeps the full (n_samples, n_features, 3) structure - each customer's
    SHAP contributions get extracted for THEIR specific predicted class
    when writing to the database, not one fixed class for everyone.

    Falls back through known SHAP/model compatibility quirks rather than
    assuming one - the winning model changes between runs (Algorithm 3.2
    picks whichever has the best AUC), so this needs to work whether
    LightGBM, XGBoost, or Random Forest wins.
    """
    background = X.sample(min(background_size, len(X)), random_state=42)
    try:
        explainer = shap.Explainer(model, background)
        return explainer(X)
    except shap.utils._exceptions.ExplainerError:
        # floating-point additivity mismatch (seen with LightGBM) - safe to disable
        return explainer(X, check_additivity=False)
    except NotImplementedError:
        # XGBoost's TreeExplainer default perturbation mode doesn't support
        # categorical splits - tree_path_dependent does
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        return explainer(X)


# ---------------------------------------------------------------------------
# Write to Postgres
# ---------------------------------------------------------------------------
def write_predictions(
    customer_ids: np.ndarray,
    probas: np.ndarray,
    tiers: list[str],
    model_version: str,
    timestamp: datetime,
) -> None:
    """Truncate and repopulate the predictions table."""
    rows = [
        {
            "customer_id": str(cid),
            "churn_probability": float(prob),
            "risk_tier": tier,
            "model_version": model_version,
            "predicted_at": timestamp,
        }
        for cid, prob, tier in zip(customer_ids, probas, tiers)
    ]
    with engine.begin() as conn:
        conn.execute(text(f'TRUNCATE "{TABLE_PREDICTIONS}" RESTART IDENTITY CASCADE'))
        for start in range(0, len(rows), 1000):
            conn.execute(insert(predictions), rows[start : start + 1000])
    print(f"Wrote {len(rows):,} rows to '{TABLE_PREDICTIONS}'.")


def write_shap_values(
    customer_ids: np.ndarray,
    shap_output,
    predicted_indices: np.ndarray,
    feature_names: list[str],
    timestamp: datetime,
) -> None:
    """Truncate and repopulate the shap_values table (JSONB per customer).
    Each customer's stored contributions are for THEIR OWN predicted class -
    a High Risk customer's SHAP explains why they're High Risk, a Low Risk
    customer's explains why they're Low Risk, not the same fixed class for
    everyone."""
    base_values = np.array(shap_output.base_values)  # shape varies by model: (n, 3), (3,), or scalar-ish

    def base_value_for(i: int, class_idx: int) -> float:
        if base_values.ndim == 2:
            return float(base_values[i, class_idx])
        if base_values.ndim == 1 and len(base_values) == len(RISK_LABEL_CLASSES):
            return float(base_values[class_idx])
        return float(base_values) if base_values.ndim == 0 else float(base_values[i])

    rows = []
    for i, cid in enumerate(customer_ids):
        class_idx = int(predicted_indices[i])
        contributions = {
            fname: float(sval)
            for fname, sval in zip(feature_names, shap_output.values[i, :, class_idx])
        }
        rows.append({
            "customer_id": str(cid),
            "feature_contributions": contributions,
            "base_value": base_value_for(i, class_idx),
            "computed_at": timestamp,
        })

    with engine.begin() as conn:
        conn.execute(text(f'TRUNCATE "{TABLE_SHAP_VALUES}" RESTART IDENTITY'))
        for start in range(0, len(rows), 500):
            conn.execute(insert(shap_values_table), rows[start : start + 500])
    print(f"Wrote {len(rows):,} rows to '{TABLE_SHAP_VALUES}'.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_tier_summary(risk_scores: np.ndarray, tiers: list[str]) -> None:
    df = pd.DataFrame({"prob": risk_scores, "tier": tiers})
    order = ["High Risk", "Medium Risk", "Low Risk"]
    print("\nRisk tier distribution:")
    print(f"  {'Tier':<12} {'Count':>7}  {'Share':>7}  {'Avg Risk Score':>18}")
    print("  " + "-" * 53)
    for tier in order:
        subset = df[df["tier"] == tier]
        n = len(subset)
        share = n / len(df) * 100
        avg_p = subset["prob"].mean() if n else 0.0
        print(f"  {tier:<12} {n:>7,}  {share:>6.2f}%  {avg_p:>18.4f}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 78)
    print("Batch Inference and Scoring")
    print("=" * 78)

    model, scaler, feature_names = load_artifacts()
    df = load_all_customers()

    print("\nPreprocessing for inference (no split, no refitting)...")
    customer_ids, X = preprocess_for_inference(df, scaler, feature_names)
    print(f"Prepared feature matrix: {X.shape}")

    print("\nPredicting risk tier for all customers...")
    proba_matrix = model.predict_proba(X)  # shape (n, 3)
    predicted_indices = proba_matrix.argmax(axis=1)
    tiers = predicted_tier_names(predicted_indices)
    risk_scores = proba_matrix @ RISK_WEIGHTS  # weighted across all 3 classes, not just P(High Risk)
    print_tier_summary(risk_scores, tiers)

    print("\nComputing SHAP values for all customers (this may take a few minutes)...")
    shap_output = compute_shap(model, X)
    print(f"SHAP values shape: {shap_output.values.shape}")

    timestamp = datetime.now(timezone.utc)
    model_version = f"{type(model).__name__}-{timestamp.strftime('%Y%m%d-%H%M%S')}"

    print(f"\nWriting to Neon Postgres (model_version={model_version})...")
    write_predictions(customer_ids, risk_scores, tiers, model_version, timestamp)
    write_shap_values(customer_ids, shap_output, predicted_indices, feature_names, timestamp)

    print("=" * 78)
    print("Batch inference complete.")
    print("=" * 78)


if __name__ == "__main__":
    run()
