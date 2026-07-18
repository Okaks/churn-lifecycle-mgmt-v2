"""
Preprocessing pipeline for churn-lifecycle-mgmt-v2 (schema v3, multi-snapshot).

v3 replaces the single fixed-cutoff snapshot (every customer scored at
month 12, one row each) with MULTIPLE snapshots per customer sampled across
their own trajectory (~30%, ~65%, and 100%/latest of their active months).

Why: a single fixed cutoff meant the model never saw what "far along in
decline" numerically looks like during training (the worst case any
customer could show by month 12 was capped), so at inference time - when
scoring customers on their actual latest data - genuinely long-dormant
customers were scoring as LOW risk, because the model had no training
examples resembling them. Multiple snapshots per customer, spanning early
to late in their trajectory, gives the model real examples across the full
range of decline severity, each correctly paired with the eventual outcome.

This is leakage-safe by construction, not by picking a "safe" cutoff date:
every rolling average and trend feature is computed with SQL window
functions that only ever look backward from each row
(ROWS BETWEEN N PRECEDING AND CURRENT ROW), so a snapshot at any position
never sees its own future - there's no separate cutoff logic to get wrong.

IMPORTANT: because multiple rows can belong to the same customer, the
train/test split (src/prepare_training_data.py) MUST split by customer_id
first, then take all of a customer's snapshot rows into whichever side they
landed on. Splitting by row would let the same customer appear in both
train and test, which inflates results dishonestly.

Produces 2 small tables in Neon:
    training_snapshots   ~6 rows per customer (~55-60K rows) - for modeling
    latest_snapshot       1 row per customer (~10K rows) - for live inference/dashboard scoring

Run:
    python src/preprocessing.py
"""

import io
import os
import sys
import time

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import engine  # noqa: E402

SNAPSHOT_FRACTIONS = [1/6, 2/6, 3/6, 4/6, 5/6, 1.0]  # one per quarter of each customer's own history
ROLLING_WINDOW = 3

CATEGORY_COLUMNS = ["transfer_count", "bills_count", "card_pos_count", "cash_withdrawal_count",
                    "salary_credit_count", "loan_repayment_count", "investment_count"]
CHANNEL_COLUMNS = ["app_count", "ussd_count", "pos_channel_count", "atm_count",
                   "internet_banking_count", "branch_count"]
EVENT_COLUMNS = {
    "complaint": "complaints_count > 0",
    "dormancy_warning": "dormancy_warning_sent",
    "password_reset": "password_reset",
    "kyc_update": "kyc_update",
    "loan_approved": "loan_approved",
}

SQL_FULL_FEATURES = """
DROP TABLE IF EXISTS stg_full_features;
CREATE TABLE stg_full_features AS
WITH base AS (
    SELECT *, (transaction_count > 0)::int AS has_txn
    FROM customer_monthly
),
grouped AS (
    SELECT *,
        SUM(has_txn) OVER (PARTITION BY customer_id ORDER BY year_month
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS txn_grp
    FROM base
)
SELECT
    g.*,

    ROW_NUMBER() OVER w - 1 AS months_since_onboarding,
    COUNT(*) OVER (PARTITION BY g.customer_id) AS total_months_for_customer,
    ROW_NUMBER() OVER w AS row_position,

    ROW_NUMBER() OVER (PARTITION BY g.customer_id, txn_grp ORDER BY year_month) - 1
        AS months_since_last_transaction,

    AVG(transaction_count) OVER w3 AS transaction_count_roll3_avg,
    AVG(login_count) OVER w3 AS login_count_roll3_avg,
    AVG(inflow_naira) OVER w3 AS inflow_naira_roll3_avg,
    AVG(outflow_naira) OVER w3 AS outflow_naira_roll3_avg,
    AVG(month_end_balance_naira) OVER w3 AS month_end_balance_naira_roll3_avg,
    COALESCE(STDDEV_SAMP(transaction_count) OVER w3, 0) AS transaction_count_volatility_roll3,

    CASE WHEN LAG(transaction_count) OVER w > 0
         THEN (transaction_count - LAG(transaction_count) OVER w)::numeric / LAG(transaction_count) OVER w
         ELSE NULL END AS transaction_count_pct_change,
    CASE WHEN LAG(login_count) OVER w > 0
         THEN (login_count - LAG(login_count) OVER w)::numeric / LAG(login_count) OVER w
         ELSE NULL END AS login_count_pct_change,
    CASE WHEN LAG(month_end_balance_naira) OVER w > 0
         THEN (month_end_balance_naira - LAG(month_end_balance_naira) OVER w) / LAG(month_end_balance_naira) OVER w
         ELSE NULL END AS month_end_balance_naira_pct_change,

    CASE WHEN AVG(bills_count) OVER w3_prior > 0
         THEN (bills_count - AVG(bills_count) OVER w3_prior) / AVG(bills_count) OVER w3_prior
         ELSE NULL END AS bills_count_vs_trailing3_pct,
    CASE WHEN AVG(unique_transfer_recipients) OVER w3_prior > 0
         THEN (unique_transfer_recipients - AVG(unique_transfer_recipients) OVER w3_prior) / AVG(unique_transfer_recipients) OVER w3_prior
         ELSE NULL END AS unique_transfer_recipients_vs_trailing3_pct,
    CASE WHEN AVG(transfer_count) OVER w3_prior > 0
         THEN (transfer_count - AVG(transfer_count) OVER w3_prior) / AVG(transfer_count) OVER w3_prior
         ELSE NULL END AS transfer_count_vs_trailing3_pct,
    CASE WHEN AVG(card_pos_count) OVER w3_prior > 0
         THEN (card_pos_count - AVG(card_pos_count) OVER w3_prior) / AVG(card_pos_count) OVER w3_prior
         ELSE NULL END AS card_pos_count_vs_trailing3_pct,

    AVG(outflow_first_3days_pct) OVER w3 AS outflow_first_3days_pct_roll3_avg,
    CASE WHEN outflow_naira > 0 THEN largest_outflow_naira / outflow_naira ELSE 0 END AS largest_outflow_share,
    AVG(CASE WHEN outflow_naira > 0 THEN largest_outflow_naira / outflow_naira ELSE 0 END) OVER w3
        AS largest_outflow_share_roll3_avg,

    products_held - FIRST_VALUE(products_held) OVER w AS products_held_change_from_start,

    -- active category / channel diversity, computed directly from columns
    (CASE WHEN transfer_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN bills_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN card_pos_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN cash_withdrawal_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN salary_credit_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN loan_repayment_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN investment_count > 0 THEN 1 ELSE 0 END) AS active_category_count,
    card_pos_count AS discretionary_txn_count,
    (CASE WHEN app_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN ussd_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN pos_channel_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN atm_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN internet_banking_count > 0 THEN 1 ELSE 0 END +
     CASE WHEN branch_count > 0 THEN 1 ELSE 0 END) AS channel_count,
    CASE WHEN (app_count + ussd_count + pos_channel_count + atm_count + internet_banking_count + branch_count) > 0
         THEN branch_count::numeric / (app_count + ussd_count + pos_channel_count + atm_count + internet_banking_count + branch_count)
         ELSE 0 END AS branch_share,

    -- lifecycle recency: months since each event last happened (NULL = never yet)
    (EXTRACT(YEAR FROM g.year_month) * 12 + EXTRACT(MONTH FROM g.year_month))::int
        - (EXTRACT(YEAR FROM MAX(CASE WHEN complaints_count > 0 THEN g.year_month END) OVER w) * 12
           + EXTRACT(MONTH FROM MAX(CASE WHEN complaints_count > 0 THEN g.year_month END) OVER w))::int
        AS months_since_complaint,
    (EXTRACT(YEAR FROM g.year_month) * 12 + EXTRACT(MONTH FROM g.year_month))::int
        - (EXTRACT(YEAR FROM MAX(CASE WHEN dormancy_warning_sent THEN g.year_month END) OVER w) * 12
           + EXTRACT(MONTH FROM MAX(CASE WHEN dormancy_warning_sent THEN g.year_month END) OVER w))::int
        AS months_since_dormancy_warning,
    (EXTRACT(YEAR FROM g.year_month) * 12 + EXTRACT(MONTH FROM g.year_month))::int
        - (EXTRACT(YEAR FROM MAX(CASE WHEN password_reset THEN g.year_month END) OVER w) * 12
           + EXTRACT(MONTH FROM MAX(CASE WHEN password_reset THEN g.year_month END) OVER w))::int
        AS months_since_password_reset,
    (EXTRACT(YEAR FROM g.year_month) * 12 + EXTRACT(MONTH FROM g.year_month))::int
        - (EXTRACT(YEAR FROM MAX(CASE WHEN kyc_update THEN g.year_month END) OVER w) * 12
           + EXTRACT(MONTH FROM MAX(CASE WHEN kyc_update THEN g.year_month END) OVER w))::int
        AS months_since_kyc_update,
    (EXTRACT(YEAR FROM g.year_month) * 12 + EXTRACT(MONTH FROM g.year_month))::int
        - (EXTRACT(YEAR FROM MAX(CASE WHEN loan_approved THEN g.year_month END) OVER w) * 12
           + EXTRACT(MONTH FROM MAX(CASE WHEN loan_approved THEN g.year_month END) OVER w))::int
        AS months_since_loan_approved

FROM grouped g
WINDOW
    w AS (PARTITION BY g.customer_id ORDER BY year_month),
    w3 AS (PARTITION BY g.customer_id ORDER BY year_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
    w3_prior AS (PARTITION BY g.customer_id ORDER BY year_month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING);

CREATE INDEX ON stg_full_features (customer_id, year_month);
"""

SQL_TRAINING_SNAPSHOTS = """
DROP TABLE IF EXISTS training_snapshots;
CREATE TABLE training_snapshots AS
WITH targets AS (
    SELECT customer_id, total_months_for_customer,
        {position_exprs}
    FROM stg_full_features
    GROUP BY customer_id, total_months_for_customer
)
SELECT f.*, p.surname, p.first_name, p.age, p.gender, p.location, p.tenure_years,
    p.account_type, p.starting_segment, p.customer_value_segment, p.avg_monthly_salary_naira, p.final_cohort,
    CASE
        WHEN p.final_cohort IN ('Healthy', 'Growing') THEN 'Low Risk'
        WHEN p.final_cohort IN ('Early Drift', 'New-Unconverted') THEN 'Medium Risk'
        WHEN p.final_cohort IN ('Inactive', 'Dormant-or-Churned') THEN 'High Risk'
    END AS risk_label,
    CASE
        WHEN p.final_cohort = 'New-Unconverted' THEN 'Never Activated'
        WHEN p.final_cohort IN ('Inactive', 'Dormant-or-Churned') THEN 'Declined After Engagement'
        ELSE 'Not Churned'
    END AS churn_pathway
FROM stg_full_features f
JOIN targets t ON t.customer_id = f.customer_id AND f.row_position IN ({position_cols})
JOIN customer_profiles p ON p.customer_id = f.customer_id;

CREATE INDEX ON training_snapshots (customer_id);
"""

SQL_LATEST_SNAPSHOT = """
DROP TABLE IF EXISTS latest_snapshot;
CREATE TABLE latest_snapshot AS
SELECT f.*, p.surname, p.first_name, p.age, p.gender, p.location, p.tenure_years,
    p.account_type, p.starting_segment, p.customer_value_segment, p.avg_monthly_salary_naira, p.final_cohort,
    CASE
        WHEN p.final_cohort IN ('Healthy', 'Growing') THEN 'Low Risk'
        WHEN p.final_cohort IN ('Early Drift', 'New-Unconverted') THEN 'Medium Risk'
        WHEN p.final_cohort IN ('Inactive', 'Dormant-or-Churned') THEN 'High Risk'
    END AS risk_label,
    CASE
        WHEN p.final_cohort = 'New-Unconverted' THEN 'Never Activated'
        WHEN p.final_cohort IN ('Inactive', 'Dormant-or-Churned') THEN 'Declined After Engagement'
        ELSE 'Not Churned'
    END AS churn_pathway
FROM stg_full_features f
JOIN customer_profiles p ON p.customer_id = f.customer_id
WHERE f.row_position = f.total_months_for_customer;

CREATE INDEX ON latest_snapshot (customer_id);
"""

SQL_CLEANUP = "DROP TABLE IF EXISTS stg_full_features;"


def build_training_snapshot_sql():
    position_exprs = ",\n        ".join(
        f"GREATEST(1, ROUND(total_months_for_customer * {frac})) AS pos_{i}"
        for i, frac in enumerate(SNAPSHOT_FRACTIONS)
    )
    position_cols = ", ".join(f"t.pos_{i}" for i in range(len(SNAPSHOT_FRACTIONS)))
    return SQL_TRAINING_SNAPSHOTS.format(position_exprs=position_exprs, position_cols=position_cols)


def run_sql(conn, sql, label):
    t0 = time.time()
    for statement in sql.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(text(statement))
    print(f"  {label}: {time.time() - t0:.1f}s", flush=True)


def fetch_table(table_name: str) -> pd.DataFrame:
    """Chunked fetch with visible progress, on a proper SQLAlchemy Connection
    (not a raw DBAPI connection - pandas warns/can fail on those). No
    statement_timeout: a fixed timeout guesses at "too slow" vs "genuinely
    stuck", and on slower connections a normal, working fetch can legitimately
    take longer than any short timeout - cancelling it makes things worse,
    not safer. The progress print every 2,000 rows is what actually answers
    "is this still working", without an artificial cutoff that can fire on
    a fetch that would have succeeded if left alone."""
    with engine.connect() as conn:
        chunks = []
        total = 0
        for chunk in pd.read_sql(f"SELECT * FROM {table_name}", conn, chunksize=2000):
            chunks.append(chunk)
            total += len(chunk)
            print(f"    ...{total:,} rows fetched", flush=True)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    print("Computing point-in-time features for every customer_monthly row (window functions - leakage-safe by construction)...")
    with engine.begin() as conn:
        run_sql(conn, SQL_FULL_FEATURES, "full feature computation")

    print("\nSampling multi-snapshot training data (1 per quarter of each customer's trajectory)...")
    with engine.begin() as conn:
        run_sql(conn, build_training_snapshot_sql(), "building training_snapshots")

    print("\nBuilding latest_snapshot (each customer's actual current state, for live inference)...")
    with engine.begin() as conn:
        run_sql(conn, SQL_LATEST_SNAPSHOT, "building latest_snapshot")

    with engine.begin() as conn:
        run_sql(conn, SQL_CLEANUP, "dropping staging table")

    print("\nPulling both tables to local CSVs (small - no bulk transfer)...")
    training_df = fetch_table("training_snapshots")
    training_df.to_csv("data/processed/training_snapshots.csv", index=False)
    print(f"  -> training_snapshots.csv ({len(training_df):,} rows, {training_df.shape[1]} cols)")
    print(f"     avg snapshots per customer: {len(training_df) / training_df['customer_id'].nunique():.2f}")

    latest_df = fetch_table("latest_snapshot")
    latest_df.to_csv("data/processed/latest_snapshot.csv", index=False)
    print(f"  -> latest_snapshot.csv ({len(latest_df):,} rows, {latest_df.shape[1]} cols)")

    print("\nrisk_label distribution in training_snapshots:")
    print(training_df["risk_label"].value_counts(normalize=True).round(3))
    print("\nrisk_label distribution in latest_snapshot:")
    print(latest_df["risk_label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    main()
