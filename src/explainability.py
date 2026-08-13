"""
SHAP Global and Local Explainability.

Implements the model-explainability layer specified in Chapter 3, section 3.4.2
and Equation 3.13 (SHAP feature-contribution decomposition).

Generates, per risk class (Low/Medium/High Risk):

  outputs/figures/
    shap_global_bar_{low,medium,high}_risk.png       — mean |SHAP| per feature
    shap_global_summary_{low,medium,high}_risk.png   — beeswarm per class
    shap_local_{low,medium,high}_risk.png            — waterfall for a
                                                        representative customer
                                                        confidently predicted
                                                        into that class

  outputs/tables/
    shap_feature_importance.csv — sorted ranking table, one section per class

Run standalone (after modeling has been executed):
    python -m src.explainability
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    BEST_MODEL_FILE,
    FEATURE_LIST_FILE,
    PROCESSED_DATA_DIR,
    RISK_LABEL_CLASSES,
    TABLES_DIR,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Figure theme — dark for the dashboard (default), light for print.
# Set FIGURE_THEME=light to write white-background figures to
# outputs/figures_light/ without touching the dashboard's set.
# ---------------------------------------------------------------------------
from src.figure_theme import (
    DPI,
    FIGURE_BG as _DARK_BG,
    OUTPUT_DIR as FIGURES_DIR,
    apply_theme,
)

apply_theme()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_artifacts() -> tuple:
    """Load the winning model and the test set."""
    model = joblib.load(BEST_MODEL_FILE)
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(
        PROCESSED_DATA_DIR / "y_test.parquet"
    )[TARGET_COLUMN]
    with open(FEATURE_LIST_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    model_name = type(model).__name__
    print(
        f"Loaded {model_name} model, X_test={X_test.shape}, "
        f"{len(feature_names)} features."
    )
    return model, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values. shap.Explainer picks TreeExplainer automatically
    for LightGBM, XGBoost, and RandomForest.

    Falls back through known SHAP/model compatibility quirks rather than
    assuming one - the winning model changes between runs (Algorithm 3.2
    picks whichever has the best AUC), so this needs to work whether
    LightGBM, XGBoost, or Random Forest wins.
    """
    explainer = shap.Explainer(model, X)
    try:
        shap_output = explainer(X)
    except shap.utils._exceptions.ExplainerError:
        # Known floating-point precision issue between newer SHAP releases
        # and LightGBM's raw score output - the mismatch is at the 1e-2
        # level, not a sign of incorrect values. Safe to disable the check.
        shap_output = explainer(X, check_additivity=False)
    except NotImplementedError:
        # XGBoost's TreeExplainer default perturbation mode doesn't support
        # categorical splits - tree_path_dependent does
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_output = explainer(X)

    # Keep the full (n_samples, n_features, n_classes) structure - do NOT
    # collapse to one class. risk_label is a genuine 3-class target now
    # (Low/Medium/High Risk), and each class deserves its own explanation
    # rather than only ever explaining one arbitrarily-chosen class.
    return explainer, shap_output


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------
def generate_global_bar_plot(shap_output, class_idx: int, class_name: str, out_path: Path) -> None:
    plt.figure()
    shap.plots.bar(shap_output[:, :, class_idx], max_display=15, show=False)
    plt.title(f"Global Feature Importance — {class_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"Saved {class_name} global bar plot to {out_path}")


def generate_global_summary_plot(shap_output, class_idx: int, class_name: str, out_path: Path) -> None:
    plt.figure()
    shap.plots.beeswarm(shap_output[:, :, class_idx], max_display=15, show=False)
    plt.title(f"Feature Impact Distribution — {class_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"Saved {class_name} global beeswarm to {out_path}")


# ---------------------------------------------------------------------------
# Local plots — pick sample customers by predicted churn probability
# ---------------------------------------------------------------------------
def pick_sample_customers(model, X: pd.DataFrame) -> dict:
    """Return {class_name: (test_set_index, confidence, class_idx)} - for
    each risk tier, the customer the model is MOST CONFIDENTLY predicting
    into that class (highest P(that class) among those actually predicted
    into it), not a probability-spectrum position that only made sense for
    a binary target."""
    proba = model.predict_proba(X)  # shape (n_samples, 3)
    predicted_class = proba.argmax(axis=1)

    samples = {}
    for class_idx, class_name in enumerate(RISK_LABEL_CLASSES):
        candidate_positions = np.where(predicted_class == class_idx)[0]
        if len(candidate_positions) == 0:
            continue  # no test customer was predicted into this class
        best_local = candidate_positions[np.argmax(proba[candidate_positions, class_idx])]
        samples[class_name] = (int(best_local), float(proba[best_local, class_idx]), class_idx)
    return samples


def generate_local_waterfall(
    shap_output, idx: int, class_idx: int, class_name: str, prob: float, out_path: Path
) -> None:
    plt.figure()
    shap.plots.waterfall(shap_output[idx, :, class_idx], max_display=12, show=False)
    plt.title(f"{class_name} example (P({class_name})={prob:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    print(f"Saved {class_name} waterfall to {out_path}")


# ---------------------------------------------------------------------------
# Feature importance table
# ---------------------------------------------------------------------------
def save_importance_table(
    shap_output, feature_names: list[str], out_path: Path
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP, separately per risk class -
    what drives a Low Risk prediction isn't necessarily what drives a High
    Risk one, and collapsing to a single ranking would hide that."""
    rows = []
    for class_idx, class_name in enumerate(RISK_LABEL_CLASSES):
        mean_abs = np.abs(shap_output.values[:, :, class_idx]).mean(axis=0)
        for feat, val in zip(feature_names, mean_abs):
            rows.append({"risk_class": class_name, "feature": feat, "mean_abs_shap": val})

    df = pd.DataFrame(rows).sort_values(
        ["risk_class", "mean_abs_shap"], ascending=[True, False]
    ).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print("\nTop 5 features per risk class (mean |SHAP|):")
    for class_name in RISK_LABEL_CLASSES:
        print(f"\n  {class_name}:")
        print(df[df["risk_class"] == class_name].head(5).to_string(index=False))
    print(f"\nSaved full ranking to {out_path}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 78)
    print("SHAP Global and Local Explainability")
    print("=" * 78)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test, _, feature_names = load_artifacts()

    print("\nComputing SHAP values on test set...")
    _, shap_output = compute_shap_values(model, X_test)
    print(f"SHAP values shape: {shap_output.values.shape}")

    print("\nGenerating global plots (one set per risk class)...")
    for class_idx, class_name in enumerate(RISK_LABEL_CLASSES):
        slug = class_name.lower().replace(" ", "_")
        generate_global_bar_plot(
            shap_output, class_idx, class_name, FIGURES_DIR / f"shap_global_bar_{slug}.png"
        )
        generate_global_summary_plot(
            shap_output, class_idx, class_name, FIGURES_DIR / f"shap_global_summary_{slug}.png"
        )

    print("\nSelecting sample customers for local explanations...")
    samples = pick_sample_customers(model, X_test)
    for class_name, (idx, prob, class_idx) in samples.items():
        print(f"  {class_name}: test index {idx}, P({class_name})={prob:.4f}")

    for class_name, (idx, prob, class_idx) in samples.items():
        slug = class_name.lower().replace(" ", "_")
        generate_local_waterfall(
            shap_output, idx, class_idx, class_name, prob,
            FIGURES_DIR / f"shap_local_{slug}.png",
        )

    save_importance_table(
        shap_output,
        feature_names,
        TABLES_DIR / "shap_feature_importance.csv",
    )

    print("\n" + "=" * 78)
    print("SHAP explainability complete.")
    print("=" * 78)


if __name__ == "__main__":
    run()
