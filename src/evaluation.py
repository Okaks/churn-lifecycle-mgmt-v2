"""
Evaluation Visualisations for Chapter 4.

Generates the visual outputs referenced in Chapter 3, section 3.4.2:
  - Confusion matrix per model (4 figures)
  - Overlaid ROC curves for all four models
  - Bar chart comparing all five metrics across models

Retrains the four models with the exact hyperparameters logged in
model_metadata during the modeling run, so results are deterministic
against the same random_state=42.

Outputs saved to outputs/figures/:
  confusion_matrix_logistic_regression.png
  confusion_matrix_random_forest.png
  confusion_matrix_xgboost.png
  confusion_matrix_lightgbm.png
  roc_curves.png
  metric_comparison.png

Run standalone (after modeling has been executed):
    python -m src.evaluation
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sqlalchemy import text
from xgboost import XGBClassifier

from src.config import (
    FEATURE_LIST_FILE,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RISK_LABEL_CLASSES,
    TABLE_MODEL_METADATA,
    TARGET_COLUMN,
)
from src.database import engine


# ---------------------------------------------------------------------------
# Dark matplotlib theme — matches the dashboard's charcoal + teal palette
# ---------------------------------------------------------------------------
_DARK_BG = "#111827"
_PANEL_BG = "#1F2937"
_GRID = "#374151"
_TEXT = "#F9FAFB"
_MUTED = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor": _DARK_BG,
    "savefig.facecolor": _DARK_BG,
    "axes.facecolor": _PANEL_BG,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _TEXT,
    "axes.titlecolor": _TEXT,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "text.color": _TEXT,
    "legend.facecolor": _PANEL_BG,
    "legend.edgecolor": _GRID,
    "legend.labelcolor": _TEXT,
    "grid.color": _GRID,
    "grid.alpha": 0.4,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Load preprocessed data and best hyperparameters
# ---------------------------------------------------------------------------
def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train = pd.read_parquet(
        PROCESSED_DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]
    y_test = pd.read_parquet(
        PROCESSED_DATA_DIR / "y_test.parquet"
    )[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def load_best_hyperparameters() -> dict[str, dict]:
    """Read the latest hyperparameters per model from model_metadata."""
    query = text(f"""
        SELECT DISTINCT ON (model_name)
            model_name, hyperparameters
        FROM "{TABLE_MODEL_METADATA}"
        ORDER BY model_name, training_timestamp DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).all()
    return {name: dict(hp) for name, hp in rows}


def load_metrics_snapshot() -> pd.DataFrame:
    """Latest metrics per model from model_metadata."""
    query = text(f"""
        SELECT DISTINCT ON (model_name)
            model_name, accuracy, precision, recall, f1_score, auc_roc
        FROM "{TABLE_MODEL_METADATA}"
        ORDER BY model_name, training_timestamp DESC
    """)
    df = pd.read_sql(query, engine)
    return df


# ---------------------------------------------------------------------------
# Retrain the four models (deterministic thanks to random_state)
# ---------------------------------------------------------------------------
def rebuild_models(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict[str, dict]
) -> dict:
    print("Rebuilding models with saved hyperparameters...")
    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ).fit(X_train, y_train)

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ).fit(X_train, y_train)

    xgb_params = best_params.get("XGBoost", {})
    models["XGBoost"] = XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, **xgb_params
    ).fit(X_train, y_train)

    lgbm_params = best_params.get("LightGBM", {})
    models["LightGBM"] = LGBMClassifier(
        random_state=RANDOM_STATE, verbose=-1, **lgbm_params
    ).fit(X_train, y_train)

    return models


# ---------------------------------------------------------------------------
# Confusion matrix per model
# ---------------------------------------------------------------------------
def plot_confusion_matrix(model, name: str, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=RISK_LABEL_CLASSES  # ["Low Risk", "Medium Risk", "High Risk"]
    )
    # "cividis" reads well on dark backgrounds; text is auto-contrasted.
    disp.plot(ax=ax, cmap="cividis", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {name}")
    # Force high-contrast text on cells
    for text_obj in ax.texts:
        text_obj.set_color("#F9FAFB")
        text_obj.set_fontsize(14)
        text_obj.set_fontweight("bold")
    ax.grid(False)
    plt.tight_layout()

    slug = name.lower().replace(" ", "_")
    out_path = FIGURES_DIR / f"confusion_matrix_{slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# ROC curves (one-vs-rest, one subplot per class - 4 models x 3 classes on
# a single plot would be unreadable, so each class gets its own panel)
# ---------------------------------------------------------------------------
def plot_roc_curves(models: dict, X_test, y_test) -> None:
    from sklearn.preprocessing import label_binarize

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # one column per class

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for class_idx, class_name in enumerate(RISK_LABEL_CLASSES):
        ax = axes[class_idx]
        for name, model in models.items():
            y_proba = model.predict_proba(X_test)[:, class_idx]
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_proba)
            auc = roc_auc_score(y_test_bin[:, class_idx], y_proba)
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

        ax.plot([0, 1], [0, 1], color=_MUTED, linestyle="--", alpha=0.6, label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{class_name} (one-vs-rest)")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("ROC Curves — Test Set (one-vs-rest per class)")
    plt.tight_layout()

    out_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Metric comparison bar chart
# ---------------------------------------------------------------------------
def plot_metric_comparison(metrics_df: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    metrics_df = metrics_df.set_index("model_name")
    metrics_df = metrics_df.sort_values("auc_roc", ascending=False)

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, row) in enumerate(metrics_df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - len(metrics_df) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "metric_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 78)
    print("Evaluation Visualisations for Chapter 4")
    print("=" * 78)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_test_data()
    print(f"Loaded X_test={X_test.shape}, y_test={y_test.shape}")

    print("\nFetching best hyperparameters from model_metadata...")
    best_params = load_best_hyperparameters()
    for name, params in best_params.items():
        print(f"  {name}: {params}")

    print()
    models = rebuild_models(X_train, y_train, best_params)

    print("\nGenerating confusion matrices...")
    for name, model in models.items():
        plot_confusion_matrix(model, name, X_test, y_test)

    print("\nGenerating ROC curves...")
    plot_roc_curves(models, X_test, y_test)

    print("\nGenerating metric comparison chart...")
    metrics_df = load_metrics_snapshot()
    plot_metric_comparison(metrics_df)

    print("\n" + "=" * 78)
    print("Evaluation figures complete.")
    print("=" * 78)


if __name__ == "__main__":
    run()
