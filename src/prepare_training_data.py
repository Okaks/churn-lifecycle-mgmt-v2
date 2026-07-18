"""
Prepares training data for src/modeling.py from training_snapshots.

Reads the local training_snapshots.csv (fast - no Neon network calls,
it's small and already sitting on disk from src/preprocessing.py).

CRITICAL: training_snapshots has ~3 rows per customer (multiple snapshots
across their trajectory). The train/test split MUST happen at the
CUSTOMER level first - split customer_ids into train/test, THEN take all
of each customer's snapshot rows into whichever side they landed on.
Splitting by row instead would let the same customer's snapshots end up on
both sides, letting the model partly memorize that customer and inflating
test performance dishonestly.

Steps:
  1. Split customer_ids into train/test (stratified by each customer's own
     churned label - constant across their snapshots).
  2. Take all snapshot rows for customers in each split.
  3. Drop ADMINISTRATIVE_COLUMNS and LEAKAGE_COLUMNS.
  4. Impute (lifecycle recency -> sentinel, everything else -> 0) and cast
     boolean flag columns to 0/1.
  5. One-hot encode categoricals.
  6. Standardise continuous columns (fit scaler on train only).
  7. Save X_train/X_test/y_train/y_test as parquet, feature list as JSON,
     scaler as joblib - the exact artifacts src/modeling.py,
     src/explainability.py, and src/evaluation.py expect.

Run:
    python -m src.prepare_training_data
"""

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    ADMINISTRATIVE_COLUMNS,
    CONTINUOUS_COLUMNS,
    FEATURE_LIST_FILE,
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    ONE_HOT_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RISK_LABEL_CLASSES,
    SCALER_FILE,
    STRATIFY,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.database import engine
from src.feature_prep import align_to_feature_list, cast_booleans, encode_categoricals, impute_missing


def load_training_snapshots() -> pd.DataFrame:
    local_path = PROCESSED_DATA_DIR / "training_snapshots.csv"
    if local_path.exists():
        print(f"Loading {local_path} (local, no network call)...")
        df = pd.read_csv(local_path)
    else:
        print("Local training_snapshots.csv not found - pulling from Neon...")
        df = pd.read_sql("SELECT * FROM training_snapshots", engine)

    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(
            f"'{TARGET_COLUMN}' not found in training_snapshots (columns present: {list(df.columns)}). "
            f"This usually means {local_path} is STALE - left over from a previous run of "
            f"src/preprocessing.py that didn't finish or ran before a schema change. "
            f"Re-run `python src/preprocessing.py` to completion first, then retry this script."
        )
    return df


def split_by_customer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the customer level - never split one customer's snapshots
    across train and test."""
    customer_labels = df.drop_duplicates("customer_id")[["customer_id", TARGET_COLUMN]]
    train_ids, test_ids = train_test_split(
        customer_labels["customer_id"],
        test_size=TEST_SIZE,
        stratify=customer_labels[TARGET_COLUMN] if STRATIFY else None,
        random_state=RANDOM_STATE,
    )
    train_df = df[df["customer_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["customer_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def prepare(train_df: pd.DataFrame, test_df: pd.DataFrame):
    label_to_int = {label: i for i, label in enumerate(RISK_LABEL_CLASSES)}
    y_train = train_df[TARGET_COLUMN].map(label_to_int)
    y_test = test_df[TARGET_COLUMN].map(label_to_int)

    drop_cols = [c for c in ADMINISTRATIVE_COLUMNS + LEAKAGE_COLUMNS + [TARGET_COLUMN]]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    X_train = cast_booleans(impute_missing(X_train))
    X_test = cast_booleans(impute_missing(X_test))

    X_train = encode_categoricals(X_train, ONE_HOT_COLUMNS)
    X_test = encode_categoricals(X_test, ONE_HOT_COLUMNS)

    feature_names = X_train.columns.tolist()
    X_test = align_to_feature_list(X_test, feature_names)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train[CONTINUOUS_COLUMNS])
    X_test[CONTINUOUS_COLUMNS] = scaler.transform(X_test[CONTINUOUS_COLUMNS])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)

    return X_train, X_test, y_train, y_test, feature_names


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_snapshots()
    print(f"Loaded {len(df):,} snapshot rows across {df['customer_id'].nunique():,} customers, {df.shape[1]} raw columns.")

    train_df, test_df = split_by_customer(df)
    train_customers = train_df["customer_id"].nunique()
    test_customers = test_df["customer_id"].nunique()
    overlap = set(train_df["customer_id"]) & set(test_df["customer_id"])
    print(f"\nCustomer split: {train_customers:,} train customers, {test_customers:,} test customers.")
    print(f"Row split: {len(train_df):,} train rows, {len(test_df):,} test rows.")
    print(f"Customer overlap between train and test: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "Customer overlap between train/test - this must never happen!"

    X_train, X_test, y_train, y_test, feature_names = prepare(train_df, test_df)

    X_train.to_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test.to_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train.to_frame(TARGET_COLUMN).to_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
    y_test.to_frame(TARGET_COLUMN).to_parquet(PROCESSED_DATA_DIR / "y_test.parquet")

    with open(FEATURE_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    label_map_file = MODELS_DIR / "risk_label_classes.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(RISK_LABEL_CLASSES, f, indent=2)

    print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")
    print("y_train distribution:", y_train.value_counts().sort_index().to_dict(), "(0=Low, 1=Medium, 2=High)")
    print("y_test distribution:", y_test.value_counts().sort_index().to_dict())
    print(f"{len(feature_names)} features saved to {FEATURE_LIST_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")
    print(f"Label classes saved to {label_map_file}")
    print("\nReady for: python -m src.modeling")


if __name__ == "__main__":
    main()
