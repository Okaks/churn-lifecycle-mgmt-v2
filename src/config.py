"""
Project configuration - schema v3 (multi-snapshot, 2-table core schema).

Centralises every constant used across the pipeline, API, and dashboard.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

BEST_MODEL_FILE = MODELS_DIR / "churn_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
FEATURE_LIST_FILE = MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "")

TABLE_CUSTOMER_PROFILES = "customer_profiles"
TABLE_CUSTOMER_MONTHLY = "customer_monthly"

# Built by src/preprocessing.py (SQL, computed in Neon)
TABLE_TRAINING_SNAPSHOTS = "training_snapshots"  # ~3 rows/customer, multi-snapshot training data
TABLE_LATEST_SNAPSHOT = "latest_snapshot"        # 1 row/customer, actual current state - for live scoring

# Written to by later pipeline stages
TABLE_MODEL_METADATA = "model_metadata"     # written by src/modeling.py
TABLE_PREDICTIONS = "predictions"           # written by src/inference.py
TABLE_SHAP_VALUES = "shap_values"           # written by src/explainability.py
TABLE_RECOMMENDATIONS = "recommendations"   # written by src/recommender.py

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
# risk_label is a genuine 3-class ground truth (computed in SQL from
# final_cohort), NOT a probability-threshold bucket:
#   Low Risk    = Healthy, Growing
#   Medium Risk = Early Drift, New-Unconverted (showing decline signals, or
#                 too new to have transacted yet - neither is "safe" nor
#                 "already gone")
#   High Risk   = Inactive, Dormant-or-Churned (already disengaged)
# This replaces an earlier binary "churned" label + probability-threshold
# tiering: a binary label could never honestly place Early Drift/
# New-Unconverted anywhere but "not churned", since in this data they never
# reach official Inactive/Dormant status - a model trained on that label
# correctly learned to score them as safe, which defeated the whole point
# of early identification. Making the 3-way distinction the actual training
# target (not a probability cutoff) forces the model to learn what
# "Medium" genuinely looks like as its own behavioural signature.
#
# churn_pathway (Never Activated / Declined After Engagement / Not Churned)
# is NOT a model feature - dashboard/reporting only, orthogonal to risk_label
# (e.g. New-Unconverted is Medium Risk AND Never Activated).
TARGET_COLUMN = "risk_label"
RISK_LABEL_CLASSES = ["Low Risk", "Medium Risk", "High Risk"]  # fixed order for encoding

# ---------------------------------------------------------------------------
# Data schema - training_snapshots / latest_snapshot columns
# ---------------------------------------------------------------------------
# Identifiers and sampling bookkeeping: never used as model features.
# has_txn/txn_grp are SQL intermediates with no standalone meaning.
# total_months_for_customer/row_position are artifacts of HOW we sampled
# snapshots, not something available at real inference time - including
# them would let the model learn a shortcut tied to our sampling procedure
# rather than genuine customer behaviour.
ADMINISTRATIVE_COLUMNS = [
    "id", "customer_id", "year_month", "surname", "first_name",
    "has_txn", "txn_grp", "total_months_for_customer", "row_position",
]

# LEAKAGE: final_cohort determines churned directly, churn_pathway is
# derived from final_cohort, and account_status is excluded because - even
# computed point-in-time - it's too close to a direct label restatement
# (no reactivation is modeled, so once a customer shows a gap it never
# closes). Always drop these before training - see src/prepare_training_data.py.
LEAKAGE_COLUMNS = ["final_cohort", "churn_pathway", "account_status"]

# Categorical - one-hot encoded
ONE_HOT_COLUMNS = [
    "gender", "location", "account_type", "starting_segment",
    "customer_value_segment", "primary_digital_channel",
]

# Already binary (0/1) by nature - cast to int, NOT standardised (a 0/1
# flag doesn't need z-score scaling the way a naira amount does).
BOOLEAN_COLUMNS = [
    "dormancy_warning_sent", "password_reset", "kyc_update", "card_issued",
    "loan_approved", "loan_completed", "new_product_opened",
]

# Continuous - standardised via z-score
CONTINUOUS_COLUMNS = [
    "age", "tenure_years", "avg_monthly_salary_naira",
    # raw monthly activity (as of the snapshot month)
    "login_count", "login_morning_count", "login_afternoon_count",
    "login_night_count", "login_weekend_count", "days_active",
    "transaction_count", "failed_transaction_count", "inflow_naira",
    "outflow_naira", "largest_outflow_naira", "outflow_first_3days_pct",
    "unique_transfer_recipients", "branch_transaction_pct", "products_held",
    "month_end_balance_naira", "complaints_count",
    # 7 consolidated transaction categories
    "transfer_count", "bills_count", "card_pos_count", "cash_withdrawal_count",
    "salary_credit_count", "loan_repayment_count", "investment_count",
    # 6 channels
    "app_count", "ussd_count", "pos_channel_count", "atm_count",
    "internet_banking_count", "branch_count",
    # rolling / trend features (src/preprocessing.py, point-in-time correct)
    "months_since_onboarding", "months_since_last_transaction",
    "transaction_count_roll3_avg", "login_count_roll3_avg", "inflow_naira_roll3_avg",
    "outflow_naira_roll3_avg", "month_end_balance_naira_roll3_avg",
    "transaction_count_volatility_roll3", "transaction_count_pct_change",
    "login_count_pct_change", "month_end_balance_naira_pct_change",
    "bills_count_vs_trailing3_pct", "unique_transfer_recipients_vs_trailing3_pct",
    "transfer_count_vs_trailing3_pct", "card_pos_count_vs_trailing3_pct",
    "outflow_first_3days_pct_roll3_avg", "largest_outflow_share",
    "largest_outflow_share_roll3_avg", "products_held_change_from_start",
    # category / channel diversity
    "active_category_count", "discretionary_txn_count", "channel_count", "branch_share",
    # lifecycle-event recency (NULL = event never happened; imputed with a
    # large sentinel value in prepare_training_data.py, not zero)
    "months_since_complaint", "months_since_dormancy_warning",
    "months_since_password_reset", "months_since_kyc_update", "months_since_loan_approved",
]

LIFECYCLE_RECENCY_COLUMNS = [
    "months_since_complaint", "months_since_dormancy_warning",
    "months_since_password_reset", "months_since_kyc_update", "months_since_loan_approved",
]

# Sentinel used to impute months_since_* columns where the event never
# happened (NULL in SQL) - large enough to read as "a very long time ago /
# never", without using NaN (which breaks most sklearn estimators).
LIFECYCLE_RECENCY_SENTINEL = 99

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
# CRITICAL: because training_snapshots has multiple rows per customer,
# splitting must happen at the CUSTOMER level first (src/prepare_training_data.py),
# then all of a customer's snapshot rows go to whichever side they landed on.
# Splitting by row would let the same customer appear in both train and
# test, inflating results dishonestly.
TEST_SIZE = 0.20
STRATIFY = True

# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
# No probability thresholds needed anymore - risk_tier is the model's direct
# 3-class prediction (argmax of predict_proba), not a cutoff applied to a
# binary probability. RISK_TIERS is kept as the canonical naming reference
# used by src/recommender.py and the dashboard.
RISK_TIERS = {
    "HIGH": "High Risk",
    "MEDIUM": "Medium Risk",
    "LOW": "Low Risk",
}

# ---------------------------------------------------------------------------
# Recommender - engagement banding
# ---------------------------------------------------------------------------
ENGAGEMENT_SCORE_COLUMN = "transaction_count_roll3_avg"

# ---------------------------------------------------------------------------
# Model-selection criterion (Algorithm 3.2, step 4)
# ---------------------------------------------------------------------------
SELECTION_PRIMARY_METRIC = "auc_roc"
SELECTION_F1_FLOOR = 0.75

# ---------------------------------------------------------------------------
# Evaluation metrics reported for every model
# ---------------------------------------------------------------------------
EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
