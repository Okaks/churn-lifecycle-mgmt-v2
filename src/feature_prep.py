"""
Shared feature-preparation logic used by both src/prepare_training_data.py
(training) and src/inference.py (scoring). Centralised here on purpose - if
training and inference imputed/encoded differently, predictions would be
silently wrong.
"""

import pandas as pd

from src.config import BOOLEAN_COLUMNS, LIFECYCLE_RECENCY_COLUMNS, LIFECYCLE_RECENCY_SENTINEL


def impute_missing(X: pd.DataFrame) -> pd.DataFrame:
    """
    NaNs in this feature set are structural, not random:
      - months_since_* : event never happened -> LIFECYCLE_RECENCY_SENTINEL
        (a large number, so "never" reads as "a very long time ago", not
        "just happened", which 0 would wrongly imply)
      - avg_monthly_salary_naira : non-salaried segment (SME/Business/
        Corporate/Student) -> 0 (genuinely no personal salary on file)
      - *_pct_change, *_vs_trailing3_pct : not enough prior history to
        compute a trend -> 0 ("no measurable change detected")
    """
    X = X.copy()
    for col in LIFECYCLE_RECENCY_COLUMNS:
        if col in X.columns:
            X[col] = X[col].fillna(LIFECYCLE_RECENCY_SENTINEL)
    X = X.fillna(0)
    return X


def cast_booleans(X: pd.DataFrame) -> pd.DataFrame:
    """customer_monthly's boolean flag columns come back from Postgres as
    True/False (or 't'/'f' strings depending on driver) - cast to 0/1 int
    explicitly so they behave predictably through scaling/modeling."""
    X = X.copy()
    for col in BOOLEAN_COLUMNS:
        if col in X.columns:
            X[col] = X[col].map({True: 1, False: 0, "True": 1, "False": 0, "t": 1, "f": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    return X


def encode_categoricals(X: pd.DataFrame, one_hot_columns: list[str]) -> pd.DataFrame:
    present = [c for c in one_hot_columns if c in X.columns]
    return pd.get_dummies(X, columns=present, drop_first=False, dtype=int)


def align_to_feature_list(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reindex to the exact training-time column set/order, filling any
    one-hot category missing from this batch with 0."""
    return X.reindex(columns=feature_names, fill_value=0)
