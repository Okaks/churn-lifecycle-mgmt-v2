"""
Database layer for churn-lifecycle-mgmt-v2 (schema v3).

Core schema, 2 tables (replaces the v2 5-table design):
    customer_profiles   one row per customer, stable attributes
    customer_monthly    one comprehensive row per customer per active month
                         (24 months where the customer has full tenure, fewer
                         for late joiners - no row exists before a customer's
                         join month). Everything that used to be split across
                         monthly_behaviour + transaction_categories +
                         channel_usage + lifecycle_events now lives here as
                         columns: activity, 7 consolidated transaction
                         categories, 6 channels, and 8 lifecycle-event flags
                         (complaints_count is a count, not a flag - a
                         customer can log more than one complaint in a month).

Why 2 tables instead of 5: the old long-format category/channel tables
(1 row per customer per month per category) were the direct cause of the
storage problems fought earlier - 7 categories x ~24 months x 10,000
customers as ROWS is 1M+ rows; the same data as COLUMNS on one row per
customer-month is ~180-240K rows total. This also makes multi-snapshot
training (src/preprocessing.py) and per-customer trajectory charts (the
dashboard) simpler - one table, one customer, no joins.

Also supports the model pipeline: model_metadata, predictions, shap_values,
recommendations (written by later pipeline stages).
"""

import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Numeric,
    Date,
    DateTime,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Point it at the churn-lifecycle-mgmt-v2 Neon project."
    )

# Neon auto-suspends idle compute; pool_pre_ping + pool_recycle avoid stale
# connections after a suspend/resume cycle, and connect_args sets a short
# server-side timeout so a cold start doesn't hang the app.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"connect_timeout": 10},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    customer_id = Column(String(20), primary_key=True)
    surname = Column(String(100), nullable=False)
    first_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    location = Column(String(100), nullable=False)
    tenure_years = Column(Numeric(4, 1), nullable=False)
    account_type = Column(String(50), nullable=False)
    starting_segment = Column(String(50), nullable=False)
    customer_value_segment = Column(String(50), nullable=False)  # Salary/Business/Student/SME/Corporate/Mass/Affluent/Private
    avg_monthly_salary_naira = Column(Numeric(14, 2), nullable=True)  # null for non-salaried segments
    final_cohort = Column(String(50), nullable=False)


class CustomerMonthly(Base):
    """One comprehensive row per customer per active month. No row exists for
    months before a customer's join month (a customer who joined Sep 2025
    has zero rows for 2024, not zero-value rows)."""
    __tablename__ = "customer_monthly"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), ForeignKey("customer_profiles.customer_id"), nullable=False)
    year_month = Column(Date, nullable=False)

    # --- core activity ---
    login_count = Column(Integer, nullable=False)
    login_morning_count = Column(Integer, nullable=False, default=0)
    login_afternoon_count = Column(Integer, nullable=False, default=0)
    login_night_count = Column(Integer, nullable=False, default=0)
    login_weekend_count = Column(Integer, nullable=False, default=0)
    days_active = Column(Integer, nullable=False)
    transaction_count = Column(Integer, nullable=False)
    failed_transaction_count = Column(Integer, nullable=False)
    inflow_naira = Column(Numeric(16, 2), nullable=False)
    outflow_naira = Column(Numeric(16, 2), nullable=False)
    largest_outflow_naira = Column(Numeric(16, 2), nullable=False)
    outflow_first_3days_pct = Column(Numeric(5, 2), nullable=False)
    unique_transfer_recipients = Column(Integer, nullable=False)
    branch_transaction_pct = Column(Numeric(5, 2), nullable=False)
    primary_digital_channel = Column(Integer, nullable=False)  # 0=none,1=POS,2=ATM,3=App,4=USSD
    products_held = Column(Integer, nullable=False)
    month_end_balance_naira = Column(Numeric(16, 2), nullable=False)
    account_status = Column(String(20), nullable=False)  # Active / Inactive / Dormant

    # --- transaction categories (7, consolidated) ---
    transfer_count = Column(Integer, nullable=False, default=0)
    bills_count = Column(Integer, nullable=False, default=0)  # includes airtime + data
    card_pos_count = Column(Integer, nullable=False, default=0)  # QR + POS + Card merged
    cash_withdrawal_count = Column(Integer, nullable=False, default=0)
    salary_credit_count = Column(Integer, nullable=False, default=0)
    loan_repayment_count = Column(Integer, nullable=False, default=0)
    investment_count = Column(Integer, nullable=False, default=0)

    # --- channels (6) ---
    app_count = Column(Integer, nullable=False, default=0)
    ussd_count = Column(Integer, nullable=False, default=0)
    pos_channel_count = Column(Integer, nullable=False, default=0)
    atm_count = Column(Integer, nullable=False, default=0)
    internet_banking_count = Column(Integer, nullable=False, default=0)
    branch_count = Column(Integer, nullable=False, default=0)

    # --- lifecycle-event flags, fired in the specific month they happened ---
    complaints_count = Column(Integer, nullable=False, default=0)  # count, not flag - can be >1
    dormancy_warning_sent = Column(Boolean, nullable=False, default=False)
    password_reset = Column(Boolean, nullable=False, default=False)
    kyc_update = Column(Boolean, nullable=False, default=False)
    card_issued = Column(Boolean, nullable=False, default=False)
    loan_approved = Column(Boolean, nullable=False, default=False)
    loan_completed = Column(Boolean, nullable=False, default=False)
    new_product_opened = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("customer_id", "year_month", name="uq_customer_monthly"),
        Index("ix_customer_monthly_customer_id", "customer_id"),
        Index("ix_customer_monthly_year_month", "year_month"),
    )


class ModelMetadata(Base):
    """One row per candidate model per training run (Algorithm 3.2, step 6).
    Runs accumulate over time - filter by training_timestamp DESC for the latest."""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    accuracy = Column(Numeric(6, 4), nullable=False)
    precision = Column(Numeric(6, 4), nullable=False)
    recall = Column(Numeric(6, 4), nullable=False)
    f1_score = Column(Numeric(6, 4), nullable=False)
    auc_roc = Column(Numeric(6, 4), nullable=False)
    hyperparameters = Column(JSONB, nullable=True)
    training_timestamp = Column(DateTime(timezone=True), nullable=False)
    is_selected = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_model_metadata_training_timestamp", "training_timestamp"),
        Index("ix_model_metadata_is_selected", "is_selected"),
    )


class Prediction(Base):
    """Latest churn probability + risk tier per customer, scored on each
    customer's LATEST actual month (not a fixed training cutoff). Truncated
    and fully repopulated by src/inference.py on every run."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), ForeignKey("customer_profiles.customer_id"), nullable=False, unique=True)
    churn_probability = Column(Numeric(6, 5), nullable=False)
    risk_tier = Column(String(20), nullable=False)
    model_version = Column(String(100), nullable=False)
    predicted_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_predictions_customer_id", "customer_id"),
        Index("ix_predictions_risk_tier", "risk_tier"),
    )


class ShapValue(Base):
    """Per-customer SHAP feature contributions (JSONB), truncated and
    repopulated by src/inference.py alongside predictions."""
    __tablename__ = "shap_values"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), ForeignKey("customer_profiles.customer_id"), nullable=False, unique=True)
    feature_contributions = Column(JSONB, nullable=False)
    base_value = Column(Numeric(10, 6), nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_shap_values_customer_id", "customer_id"),
    )


class Recommendation(Base):
    """Per-customer retention recommendation, truncated and repopulated by
    src/recommender.py after each inference run."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(20), ForeignKey("customer_profiles.customer_id"), nullable=False, unique=True)
    risk_tier = Column(String(20), nullable=False)
    churn_pathway = Column(String(50), nullable=False)
    recommendation_type = Column(String(50), nullable=False)
    recommendation_text = Column(String(1000), nullable=False)
    rationale = Column(String(1000), nullable=False)
    generated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_recommendations_customer_id", "customer_id"),
        Index("ix_recommendations_risk_tier", "risk_tier"),
    )


# aliases matching the import style used elsewhere in the pipeline:
# `from src.database import engine, model_metadata_table`
# `from src.database import engine, predictions`
# `from src.database import shap_values as shap_values_table`
# `from src.database import engine, recommendations as recommendations_table`
model_metadata_table = ModelMetadata.__table__
predictions = Prediction.__table__
shap_values = ShapValue.__table__
recommendations = Recommendation.__table__


def _drop_legacy_v2_tables():
    """Drop any tables left over from the old 5-table v2 schema
    (monthly_behaviour, transaction_categories, channel_usage,
    lifecycle_events, and any stray features_long/features_snapshot from
    the old SQL-computed preprocessing) if they still exist."""
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    legacy = {"monthly_behaviour", "transaction_categories", "channel_usage",
              "lifecycle_events", "features_long", "features_snapshot"}
    for table_name in legacy & existing:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
        print(f"Dropped legacy table `{table_name}`.")


def _connect_with_retry(retries=3, delay=2):
    """Neon cold starts can drop the first connection attempt; retry briefly."""
    import time as _time
    for attempt in range(1, retries + 1):
        try:
            with engine.connect():
                return
        except OperationalError:
            if attempt == retries:
                raise
            _time.sleep(delay)


def init_db(force: bool = False):
    """
    Create customer_profiles, customer_monthly, and the model-pipeline tables.

    force=True drops all v3 tables first for a clean reset (used by
    `scripts/load_data.py --force`). Also always cleans up any leftover
    v2-schema tables regardless of force, since they're never valid to keep
    around once this schema is in place.
    """
    _connect_with_retry()
    _drop_legacy_v2_tables()

    if force:
        for table in (Recommendation, ShapValue, Prediction, ModelMetadata,
                      CustomerMonthly, CustomerProfile):
            table.__table__.drop(engine, checkfirst=True)
        print("Dropped existing v3 tables (force reset).")

    Base.metadata.create_all(engine)
    print("All v3 tables ready: customer_profiles, customer_monthly, "
          "model_metadata, predictions, shap_values, recommendations.")


def get_session():
    return SessionLocal()


if __name__ == "__main__":
    import sys

    init_db(force="--force" in sys.argv)
