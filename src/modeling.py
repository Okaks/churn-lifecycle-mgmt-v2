"""
Algorithm 3.2 — Model Construction and Selection Pipeline.

Implements the model training, hyperparameter tuning, evaluation, and
selection procedure specified in Chapter 3, sections 3.3.5 through 3.4.2:

  1. Load the preprocessed arrays produced by Algorithm 3.1.
  2. Train four candidate models on the 3-class risk_label target
     (Low Risk / Medium Risk / High Risk):
     - Logistic Regression (baseline, fixed hyperparameters)
     - Random Forest       (baseline, fixed hyperparameters)
     - XGBoost             (GridSearchCV, cv=5, scoring='f1_macro')
     - LightGBM            (GridSearchCV, cv=5, scoring='f1_macro')
  3. Evaluate every model on the held-out test set using five metrics,
     each macro-averaged across the 3 classes:
     Accuracy, Precision, Recall, F1, AUC-ROC (one-vs-rest).
  4. Select the winner using Chapter 3's rule: argmax(AUC) subject to
     F1 >= 0.75. Falls back to argmax(AUC) alone if no model qualifies.
  5. Persist the winning model to models/churn_model.joblib.
  6. Write per-model metrics and hyperparameters to Neon (model_metadata).

Run standalone (after preprocessing has been executed):
    python -m src.modeling
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sqlalchemy import insert
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_FILE,
    FEATURE_LIST_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SELECTION_F1_FLOOR,
    TABLE_MODEL_METADATA,
    TARGET_COLUMN,
)
from src.database import engine, model_metadata_table


# ---------------------------------------------------------------------------
# GridSearchCV grids — 3^3 = 27 combinations per model, ~135 fits at cv=5
# ---------------------------------------------------------------------------
GRID_XGBOOST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.10, 0.20],
}

GRID_LIGHTGBM = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.10, 0.20],
}


# ---------------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------------
def load_preprocessed() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]
]:
    """Load the parquet files produced by src.preprocessing."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train = pd.read_parquet(
        PROCESSED_DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]
    y_test = pd.read_parquet(
        PROCESSED_DATA_DIR / "y_test.parquet"
    )[TARGET_COLUMN]

    with open(FEATURE_LIST_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(
        f"Loaded preprocessed data: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"{len(feature_names)} features."
    )
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Model builders — Chapter 3, sections 3.3.5.1 to 3.3.5.4
# ---------------------------------------------------------------------------
def build_logistic_regression() -> LogisticRegression:
    """Chapter 3, section 3.3.5.1: LogisticRegression(max_iter=1000, random_state=42)."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def build_random_forest() -> RandomForestClassifier:
    """Chapter 3, section 3.3.5.2: RandomForestClassifier(n_estimators=200, random_state=42)."""
    return RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    )


def build_xgboost_base() -> XGBClassifier:
    """Chapter 3, section 3.3.5.3: XGBClassifier before hyperparameter tuning.

    Note: Chapter 3 specifies `use_label_encoder=False`. This parameter was
    deprecated in XGBoost 1.6 and removed in 2.0+. Since we are on XGBoost 3.x,
    it is omitted; `eval_metric='logloss'` is retained.
    """
    return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)


def build_lightgbm_base() -> LGBMClassifier:
    """Chapter 3, section 3.3.5.4: LGBMClassifier(random_state=42, verbose=-1)."""
    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)


# ---------------------------------------------------------------------------
# GridSearchCV wrapper
# ---------------------------------------------------------------------------
def tune(
    estimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple:
    """Run 5-fold GridSearchCV with F1 scoring (Chapter 3, section 3.3.5)."""
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Evaluation — Chapter 3, section 3.4.2
# ---------------------------------------------------------------------------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute the five metrics specified in Chapter 3, section 3.4.2, each
    macro-averaged across the 3 classes (Low/Medium/High Risk) - the model
    predicts a genuine 3-way risk_label now, not a binary churned flag, so
    there's no single "positive" class for pos_label/column-1 extraction."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # shape (n_samples, 3) - all 3 classes, not just column 1
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "auc_roc": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")),
    }


# ---------------------------------------------------------------------------
# Selection rule — Chapter 3, Algorithm 3.2, step 4
# ---------------------------------------------------------------------------
def select_best(results: dict[str, dict]) -> str:
    """argmax(AUC) subject to F1 >= 0.75; fall back to argmax(AUC) otherwise."""
    qualifying = {
        name: metrics
        for name, metrics in results.items()
        if metrics["f1"] >= SELECTION_F1_FLOOR
    }

    if qualifying:
        winner = max(qualifying, key=lambda k: qualifying[k]["auc_roc"])
        print(
            f"\nSelection: {winner} "
            f"(AUC={qualifying[winner]['auc_roc']:.4f}, "
            f"F1={qualifying[winner]['f1']:.4f}) — "
            f"argmax AUC among models with F1 >= {SELECTION_F1_FLOOR}."
        )
    else:
        winner = max(results, key=lambda k: results[k]["auc_roc"])
        print(
            f"\nWARNING: No model met the F1 >= {SELECTION_F1_FLOOR} floor. "
            f"Falling back to argmax AUC. Selected: {winner} "
            f"(AUC={results[winner]['auc_roc']:.4f}, "
            f"F1={results[winner]['f1']:.4f})."
        )
    return winner


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model, model_name: str) -> None:
    """Serialise the winning model to models/churn_model.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_MODEL_FILE)
    print(f"Saved winning model ({model_name}) to {BEST_MODEL_FILE}")


def save_metrics_to_db(
    results: dict[str, dict],
    hyperparameters: dict[str, dict],
    winner_name: str,
    run_timestamp: datetime,
) -> None:
    """Insert one row per candidate model into the model_metadata table.

    Runs accumulate over time — filter by training_timestamp DESC to get the
    most recent. To reset history manually:
        TRUNCATE model_metadata RESTART IDENTITY;
    """
    rows = [
        {
            "model_name": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
            "hyperparameters": hyperparameters.get(name, {}),
            "training_timestamp": run_timestamp,
            "is_selected": name == winner_name,
        }
        for name, metrics in results.items()
    ]

    with engine.begin() as conn:
        conn.execute(insert(model_metadata_table), rows)

    print(f"Wrote {len(rows)} rows to '{TABLE_MODEL_METADATA}' in Neon.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_comparison_table(
    results: dict[str, dict], winner_name: str
) -> None:
    """Print the model-comparison table for Chapter 4."""
    print("\n" + "=" * 82)
    print("Model Comparison — Chapter 4")
    print("=" * 82)
    header = (
        f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}"
        f"{'Recall':>10}{'F1':>10}{'AUC-ROC':>10}"
    )
    print(header)
    print("-" * 82)
    for name, metrics in sorted(
        results.items(), key=lambda kv: -kv[1]["auc_roc"]
    ):
        marker = "  <-- SELECTED" if name == winner_name else ""
        print(
            f"{name:<22}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{metrics['auc_roc']:>10.4f}"
            f"{marker}"
        )
    print("=" * 82)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> None:
    """Execute Algorithm 3.2 end-to-end."""
    print("=" * 82)
    print("Algorithm 3.2 — Model Construction and Selection Pipeline")
    print("=" * 82)

    X_train, X_test, y_train, y_test, _ = load_preprocessed()
    run_timestamp = datetime.now(timezone.utc)

    results: dict[str, dict] = {}
    hyperparameters: dict[str, dict] = {}

    # ---- Logistic Regression ----
    print("\n[1/4] Training Logistic Regression...")
    lr = build_logistic_regression()
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate(lr, X_test, y_test)
    hyperparameters["Logistic Regression"] = {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }
    print(
        f"      AUC={results['Logistic Regression']['auc_roc']:.4f}, "
        f"F1={results['Logistic Regression']['f1']:.4f}"
    )

    # ---- Random Forest ----
    print("\n[2/4] Training Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate(rf, X_test, y_test)
    hyperparameters["Random Forest"] = {
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
    }
    print(
        f"      AUC={results['Random Forest']['auc_roc']:.4f}, "
        f"F1={results['Random Forest']['f1']:.4f}"
    )

    # ---- XGBoost with GridSearchCV ----
    print("\n[3/4] Tuning XGBoost via GridSearchCV (27 combinations, cv=5)...")
    xgb_best, xgb_params = tune(
        build_xgboost_base(), GRID_XGBOOST, X_train, y_train
    )
    results["XGBoost"] = evaluate(xgb_best, X_test, y_test)
    hyperparameters["XGBoost"] = xgb_params
    print(f"      Best params: {xgb_params}")
    print(
        f"      AUC={results['XGBoost']['auc_roc']:.4f}, "
        f"F1={results['XGBoost']['f1']:.4f}"
    )

    # ---- LightGBM with GridSearchCV ----
    print("\n[4/4] Tuning LightGBM via GridSearchCV (27 combinations, cv=5)...")
    lgbm_best, lgbm_params = tune(
        build_lightgbm_base(), GRID_LIGHTGBM, X_train, y_train
    )
    results["LightGBM"] = evaluate(lgbm_best, X_test, y_test)
    hyperparameters["LightGBM"] = lgbm_params
    print(f"      Best params: {lgbm_params}")
    print(
        f"      AUC={results['LightGBM']['auc_roc']:.4f}, "
        f"F1={results['LightGBM']['f1']:.4f}"
    )

    # ---- Select ----
    winner_name = select_best(results)
    winning_model = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb_best,
        "LightGBM": lgbm_best,
    }[winner_name]

    # ---- Persist ----
    save_model(winning_model, winner_name)
    save_metrics_to_db(
        results, hyperparameters, winner_name, run_timestamp
    )

    # ---- Report ----
    print_comparison_table(results, winner_name)


if __name__ == "__main__":
    run()
