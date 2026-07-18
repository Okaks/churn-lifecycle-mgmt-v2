"""
Cached database access for the dashboard.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CACHE_TTL = 300


@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        st.error("DATABASE_URL is not set. Check your .env file.")
        st.stop()
    return create_engine(database_url, pool_pre_ping=True, pool_recycle=300)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading customer predictions...")
def load_predictions() -> pd.DataFrame:
    """Every customer, joined with their full snapshot features + prediction.
    This is the single richest table the dashboard works from - equivalent
    to v1's load_predictions() but carrying the full v2 feature set."""
    query = """
        SELECT
            s.customer_id, cp.surname, cp.first_name,
            s.age, s.gender, s.location, s.tenure_years, s.account_type,
            s.starting_segment, s.customer_value_segment, s.avg_monthly_salary_naira,
            s.products_held, s.account_status, s.month_end_balance_naira,
            s.transaction_count_roll3_avg, s.months_since_last_transaction,
            s.months_since_onboarding, s.channel_count, s.branch_share,
            s.discretionary_txn_count, s.bill_payment_count_vs_trailing3_pct,
            s.unique_transfer_recipients_vs_trailing3_pct, s.transfer_count_vs_trailing3_pct,
            s.outflow_first_3days_pct_roll3_avg, s.largest_outflow_share,
            s.months_since_complaint_raised, s.months_since_dormancy_warning_sent,
            s.final_cohort, s.churn_pathway, s.churned,
            p.churn_probability, p.risk_tier, p.model_version, p.predicted_at
        FROM features_snapshot s
        JOIN predictions p ON p.customer_id = s.customer_id
        LEFT JOIN customer_profiles cp ON cp.customer_id = s.customer_id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading recommendations...")
def load_recommendations() -> pd.DataFrame:
    query = """
        SELECT
            r.customer_id, cp.surname, cp.first_name, r.risk_tier, r.churn_pathway,
            r.recommendation_type, r.recommendation_text, r.rationale, r.generated_at
        FROM recommendations r
        LEFT JOIN customer_profiles cp ON cp.customer_id = r.customer_id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading model metadata...")
def load_model_metadata() -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (model_name)
            model_name, accuracy, precision, recall, f1_score, auc_roc,
            hyperparameters, training_timestamp, is_selected
        FROM model_metadata
        ORDER BY model_name, training_timestamp DESC
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)
    return df.sort_values("auc_roc", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def load_shap_for_customer(customer_id: str) -> dict | None:
    query = text("SELECT feature_contributions, base_value FROM shap_values WHERE customer_id = :cid")
    with get_engine().connect() as conn:
        row = conn.execute(query, {"cid": customer_id}).mappings().first()
    return dict(row) if row else None


@st.cache_data
def load_shap_feature_importance() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "outputs" / "tables" / "shap_feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading customer explanations...")
def load_explanations_table() -> pd.DataFrame:
    """One row per customer: full snapshot features + prediction + SHAP + recommendation."""
    query = """
        SELECT
            s.customer_id, cp.surname, cp.first_name,
            s.age, s.gender, s.location, s.tenure_years, s.products_held,
            s.account_status, s.customer_value_segment, s.avg_monthly_salary_naira,
            s.transaction_count_roll3_avg, s.months_since_last_transaction,
            s.months_since_onboarding, s.channel_count, s.branch_share,
            s.discretionary_txn_count, s.bill_payment_count_vs_trailing3_pct,
            s.unique_transfer_recipients_vs_trailing3_pct, s.transfer_count_vs_trailing3_pct,
            s.outflow_first_3days_pct_roll3_avg, s.largest_outflow_share,
            s.months_since_complaint_raised, s.months_since_dormancy_warning_sent,
            s.final_cohort, s.churn_pathway, s.churned,
            p.churn_probability, p.risk_tier,
            sv.feature_contributions,
            r.recommendation_type, r.recommendation_text, r.rationale
        FROM features_snapshot s
        JOIN predictions p ON p.customer_id = s.customer_id
        LEFT JOIN customer_profiles cp ON cp.customer_id = s.customer_id
        LEFT JOIN shap_values sv ON sv.customer_id = s.customer_id
        LEFT JOIN recommendations r ON r.customer_id = s.customer_id
    """
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL)
def get_filter_options() -> dict:
    with get_engine().connect() as conn:
        pathways = [r[0] for r in conn.execute(text("SELECT DISTINCT churn_pathway FROM features_snapshot")).fetchall()]
        cohorts = [r[0] for r in conn.execute(text("SELECT DISTINCT final_cohort FROM features_snapshot")).fetchall()]
        locations = [r[0] for r in conn.execute(text("SELECT DISTINCT location FROM features_snapshot")).fetchall()]
        segments = [r[0] for r in conn.execute(text("SELECT DISTINCT customer_value_segment FROM features_snapshot")).fetchall()]
    return {
        "pathways": sorted(pathways), "cohorts": sorted(cohorts),
        "locations": sorted(locations), "segments": sorted(segments),
    }
