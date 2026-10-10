"""
Customer Churn Prediction & Lifecycle Management Dashboard (v3).

Five tabs:
  1. Portfolio Overview        - aggregate metrics, cohort/pathway/risk distributions
  2. Customer Risk Explorer    - searchable per-customer table + trajectory charts
  3. Customer Explainability   - plain-English SHAP explanations + decline-rate charts
  4. Retention Recommendations - filterable list of pathway-aware retention actions
  5. Model Performance         - metrics comparison + confusion matrices + ROC

risk_tier is now the model's genuine direct 3-class prediction (Low/Medium/
High Risk), not a probability-threshold bucket - see src/inference.py.
churn_probability is a WEIGHTED risk score (0 x P(Low) + 0.5 x P(Medium) +
1.0 x P(High)), not P(High Risk) alone - so Medium Risk customers score
near 0.5, distinct from Low Risk (near 0.0), not indistinguishable from it.

Reads live data from Neon (latest_snapshot, customer_monthly, predictions,
shap_values, recommendations, model_metadata) and static figures from
outputs/figures/.

Run (venv active, cwd is project root):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src.config import RISK_LABEL_CLASSES
from src.database import engine

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .js-plotly-plot {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 8px;
        background: rgba(255, 255, 255, 0.015);
    }
    .stImage img {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        max-height: 420px;
        width: 100%;
        object-fit: contain;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 14px 16px;
        border-radius: 8px;
    }
    span[data-baseweb="tag"] {
        background-color: #374151 !important;
        color: #F9FAFB !important;
        border: 1px solid #5FDFDC !important;
    }
    span[data-baseweb="tag"] span { color: #F9FAFB !important; }
    ul[data-baseweb="menu"], div[data-baseweb="popover"] {
        background-color: #1F2937 !important;
        color: #F9FAFB !important;
    }
    ul[data-baseweb="menu"] li { color: #F9FAFB !important; }
    ul[data-baseweb="menu"] li:hover { background-color: #374151 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Colour palettes - consistent across every chart
# ---------------------------------------------------------------------------
TIER_COLOURS = {"High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#14B8A6"}
PATHWAY_COLOURS = {
    "Never Activated": "#F59E0B",
    "Declined After Engagement": "#EF4444",
    "Not Churned": "#14B8A6",
}
TIER_EMOJI = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}

# Ground-truth mapping (matches src/preprocessing.py's risk_label SQL exactly) -
# used to show whether the model's prediction matches what the customer's
# actual cohort implies, a genuine per-customer accuracy check.
COHORT_TO_EXPECTED_TIER = {
    "Healthy": "Low Risk", "Growing": "Low Risk",
    "Early Drift": "Medium Risk", "New-Unconverted": "Medium Risk",
    "Inactive": "High Risk", "Dormant-or-Churned": "High Risk",
}


def metric_card(label: str, value: str, value_colour: str, delta: str | None = None) -> str:
    delta_html = (
        f'<div style="color:{value_colour}; font-size:12px; margin-top:6px; font-weight:500;">{delta}</div>'
        if delta else
        '<div style="font-size:12px; margin-top:6px; visibility:hidden;">·</div>'
    )
    return f"""
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid {value_colour}55;
            padding: 14px 16px; border-radius: 8px; min-height: 108px;
            display: flex; flex-direction: column; justify-content: space-between;">
    <div>
        <div style="color:#94A3B8; font-size:13px; font-weight:400;">{label}</div>
        <div style="color:{value_colour}; font-size:28px; font-weight:500; margin-top:6px; line-height:1.1;">{value}</div>
    </div>
    {delta_html}
</div>
"""


def format_count_label(v):
    if v >= 1000:
        return f"{v/1000:.1f}k"
    return f"{int(v)}"


def sorted_bar(df, x, y, color=None, color_discrete_map=None, label_format="count", **kwargs):
    """Bar chart with categories sorted highest-to-lowest by value - used
    everywhere per the house rule that charts read in descending order.
    label_format: 'count' -> whole numbers, 'k' notation over 1000.
                  'score' -> 2 decimal places (for probabilities/scores 0-1)."""
    df = df.sort_values(y, ascending=False).copy()
    if label_format == "score":
        df["_label"] = df[y].apply(lambda v: f"{v:.2f}")
    else:
        df["_label"] = df[y].apply(format_count_label)
    fig = px.bar(df, x=x, y=y, color=color, color_discrete_map=color_discrete_map, text="_label", **kwargs)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders - each is a single query, no per-row Python loops over
# large frames, and NO pandas .style.format() on large tables (that's what
# caused the hang before - Styler is extremely slow past a few hundred rows;
# st.column_config formatting achieves the same look at native speed).
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading customer predictions...")
def load_predictions() -> pd.DataFrame:
    query = """
        SELECT
            s.customer_id, s.age, s.gender, s.location, s.tenure_years,
            s.account_type, s.starting_segment, s.customer_value_segment,
            s.avg_monthly_salary_naira, s.products_held, s.month_end_balance_naira,
            s.account_status, s.transaction_count_roll3_avg, s.months_since_last_transaction,
            s.channel_count, s.branch_share, s.final_cohort, s.churn_pathway,
            p.churn_probability, p.risk_tier, p.model_version, p.predicted_at
        FROM latest_snapshot s
        JOIN predictions p ON p.customer_id = s.customer_id
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300, show_spinner="Loading recommendations...")
def load_recommendations() -> pd.DataFrame:
    query = "SELECT customer_id, risk_tier, churn_pathway, recommendation_type, recommendation_text, rationale, generated_at FROM recommendations"
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300, show_spinner="Loading model metadata...")
def load_model_metadata() -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (model_name)
            model_name, accuracy, precision, recall, f1_score, auc_roc,
            hyperparameters, training_timestamp, is_selected
        FROM model_metadata ORDER BY model_name, training_timestamp DESC
    """
    df = pd.read_sql(query, engine)
    return df.sort_values("auc_roc", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_shap_feature_importance() -> pd.DataFrame | None:
    path = _PROJECT_ROOT / "outputs" / "tables" / "shap_feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300, show_spinner="Loading customer explanations...")
def load_explanations_summary() -> pd.DataFrame:
    """Lean version - everything needed to filter/sort/display the table,
    WITHOUT the feature_contributions JSONB blob (that's fetched separately,
    on demand, only for the single customer being viewed in detail).
    Pulling all 10,000 customers' full SHAP JSONB upfront when at most 200
    rows are ever shown was the single biggest cost in this tab."""
    query = """
        SELECT
            s.customer_id, s.age, s.tenure_years, s.months_since_onboarding,
            s.products_held, s.account_status, s.account_type, s.customer_value_segment,
            s.transaction_count_pct_change, s.login_count_pct_change,
            s.bills_count_vs_trailing3_pct, s.unique_transfer_recipients_vs_trailing3_pct,
            s.transfer_count_vs_trailing3_pct, s.card_pos_count_vs_trailing3_pct,
            s.outflow_first_3days_pct_roll3_avg,
            s.largest_outflow_share, s.months_since_last_transaction, s.channel_count,
            s.branch_share, s.discretionary_txn_count, s.months_since_complaint,
            s.months_since_dormancy_warning, s.final_cohort, s.churn_pathway,
            p.churn_probability, p.risk_tier,
            r.recommendation_type, r.recommendation_text, r.rationale
        FROM latest_snapshot s
        JOIN predictions p ON p.customer_id = s.customer_id
        LEFT JOIN recommendations r ON r.customer_id = s.customer_id
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def load_shap_for_customers(customer_ids: tuple[str, ...]) -> dict[str, dict]:
    """Batched fetch for exactly the customer IDs currently on screen (e.g. the
    top 200 filtered rows) - one query, not 200 round trips and not all 10,000."""
    if not customer_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT customer_id, feature_contributions FROM shap_values WHERE customer_id = ANY(:ids)"),
            {"ids": list(customer_ids)},
        ).fetchall()
    return {cid: contribs for cid, contribs in rows}


@st.cache_data(ttl=300)
def load_customer_trajectory(customer_id: str) -> pd.DataFrame:
    """Full month-by-month history for one customer, straight from
    customer_monthly - the actual behavioural story (logins, transfers,
    bills, complaints, account status) over their whole tenure, not just a
    single snapshot. This is what customer_monthly staying in Neon
    (unlike the old long-format tables we had to drop for storage) makes
    possible - one simple query, no joins."""
    query = """
        SELECT year_month, login_count, transaction_count, transfer_count,
               bills_count, card_pos_count, cash_withdrawal_count,
               unique_transfer_recipients, complaints_count, account_status,
               month_end_balance_naira, products_held,
               dormancy_warning_sent, card_issued, loan_approved,
               loan_completed, new_product_opened
        FROM customer_monthly
        WHERE customer_id = :cid
        ORDER BY year_month
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params={"cid": customer_id})


# ---------------------------------------------------------------------------
# Plain-English explanation engine, mapped to v2's actual behavioural
# features (the four drift patterns + lifecycle recency signals), replacing
# v1's Kaggle-schema version (Balance, CreditScore, etc. don't exist here).
# ---------------------------------------------------------------------------
_EXCLUDED_PREFIXES = ("gender_", "location_", "account_type_", "starting_segment_",
                      "customer_value_segment_", "primary_digital_channel_")


def explain_feature(feature: str, shap_value: float, row: dict) -> str | None:
    if any(feature.startswith(p) for p in _EXCLUDED_PREFIXES):
        return None
    is_risk = shap_value > 0

    if feature in ("unique_transfer_recipients_vs_trailing3_pct", "unique_transfer_recipients_vs_trailing3_pct_avg_over_history"):
        return "Sending money to far fewer people than their own normal pattern (transfer diversity narrowing)" if is_risk else "Maintains a diverse, stable network of transfer recipients"
    if feature in ("transfer_count_vs_trailing3_pct", "transfer_count_vs_trailing3_pct_avg_over_history"):
        return "Transfer activity has dropped compared to their recent normal" if is_risk else "Transfer activity holding steady or growing"
    if feature in ("bills_count_vs_trailing3_pct", "bills_count_vs_trailing3_pct_avg_over_history"):
        return "Bill payments have declined or stopped (possible bill abandonment)" if is_risk else "Consistently paying bills through the app"
    if feature in ("card_pos_count_vs_trailing3_pct", "card_pos_count_vs_trailing3_pct_avg_over_history"):
        return "Card/POS spending has dropped compared to their normal pattern" if is_risk else "Steady card/POS spending activity"
    if feature == "outflow_first_3days_pct_roll3_avg":
        return "Spending is concentrated in the first few days after funds arrive (rapid spend-down)" if is_risk else "Spending is spread evenly across the month"
    if feature in ("largest_outflow_share", "largest_outflow_share_roll3_avg"):
        return "A single large withdrawal dominates their outflow (possible funds moving elsewhere)" if is_risk else "No unusually large single withdrawals"
    if feature == "months_since_last_transaction":
        n = row.get("months_since_last_transaction", 0)
        return f"Has not transacted in {int(n)} month(s)" if is_risk else "Transacted recently"
    if feature in ("transaction_count_pct_change", "transaction_count_pct_change_avg_over_history"):
        return "Transaction volume trending downward" if is_risk else "Transaction volume stable or growing"
    if feature in ("login_count_pct_change", "login_count_pct_change_avg_over_history"):
        return "Logging in less often than before" if is_risk else "Login activity stable or increasing"
    if feature == "channel_count":
        return "Using fewer digital channels than typical (channel narrowing)" if is_risk else "Actively uses multiple channels (app, USSD, POS, etc.)"
    if feature == "branch_share":
        return "Increasingly relying on branch visits over digital channels" if is_risk else "Fully digital-first transaction behaviour"
    if feature in ("discretionary_txn_count", "discretionary_txn_count_avg_over_history"):
        return "Little day-to-day spending activity (airtime, data, POS, card)" if is_risk else "Regular everyday spending activity"
    if feature == "products_held":
        n = int(row.get("products_held", 1))
        if is_risk:
            return "Holds only one product with the bank" if n <= 1 else None
        return "Deep multi-product relationship" if n >= 3 else ("Well-diversified product relationship" if n == 2 else None)
    if feature in ("tenure_years", "months_since_onboarding"):
        return "Newer customer, relationship still developing" if is_risk else "Long-established customer relationship"
    if feature == "months_since_complaint":
        n = row.get("months_since_complaint", 99)
        return f"Raised a complaint {int(n)} month(s) ago" if is_risk and n < 6 else None
    if feature == "months_since_dormancy_warning":
        n = row.get("months_since_dormancy_warning", 99)
        return "Already flagged with a dormancy warning" if is_risk and n < 12 else None
    if feature == "month_end_balance_naira_pct_change":
        return "Account balance trending down" if is_risk else "Account balance stable or growing"
    return None


def build_customer_explanation(row: pd.Series, contribs: dict | None) -> tuple[list[str], list[str]]:
    if not isinstance(contribs, dict):
        return [], []
    row_dict = row.to_dict()
    ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    risk, protective = [], []
    for feat, val in ranked:
        txt = explain_feature(feat, val, row_dict)
        if txt is None:
            continue
        if val > 0 and len(risk) < 3 and txt not in risk:
            risk.append(txt)
        elif val < 0 and len(protective) < 2 and txt not in protective:
            protective.append(txt)
        if len(risk) >= 3 and len(protective) >= 2:
            break
    return risk, protective


def format_risk_narrative(risk_factors: list[str]) -> str:
    if not risk_factors:
        return "No specific risk drivers stood out - but overall pattern still elevates churn probability."
    return " · ".join(risk_factors)


def format_protective_narrative(protective_factors: list[str], products_held: int, is_active: bool) -> str:
    if protective_factors:
        return " · ".join(protective_factors)
    hints = []
    if not is_active:
        hints.append("not currently active")
    if products_held <= 1:
        hints.append("shallow product relationship")
    if hints:
        return "No strong protective factors - " + ", ".join(hints) + ". Priority for retention outreach."
    return "No strong protective factors identified - minimal buffer against churn."


def render_trajectory_chart(customer_id: str, key: str) -> None:
    """Renders the lifecycle progression chart for one customer - shared
    between the Customer Risk Explorer and Customer Explainability tabs.
    `key` must be unique per call site since this gets called from two
    different tabs in the same script run."""
    trajectory = load_customer_trajectory(customer_id)
    if len(trajectory) < 2:
        st.info("Not enough history for this customer to chart a trajectory (very recent join).")
        return

    fig = go.Figure()
    for col, label, color in [
        ("login_count", "Logins", "#5FDFDC"),
        ("transfer_count", "Transfers", "#F59E0B"),
        ("bills_count", "Bill Payments", "#EF4444"),
        ("card_pos_count", "Card/POS", "#A78BFA"),
    ]:
        fig.add_trace(go.Scatter(x=trajectory["year_month"], y=trajectory[col], mode="lines+markers",
                                  name=label, line=dict(color=color, width=2)))

    non_active = trajectory[trajectory["account_status"] != "Active"]
    for _, r in non_active.iterrows():
        fig.add_vrect(x0=r["year_month"], x1=r["year_month"], fillcolor="#EF4444", opacity=0.08, line_width=0)

    event_cols = {"card_issued": "Card Issued", "loan_approved": "Loan Approved",
                 "loan_completed": "Loan Completed", "new_product_opened": "New Product",
                 "dormancy_warning_sent": "Dormancy Warning"}
    y_max = trajectory[["login_count", "transfer_count", "bills_count", "card_pos_count"]].values.max()
    for col, label in event_cols.items():
        hits = trajectory[trajectory[col] == True]
        for _, r in hits.iterrows():
            fig.add_annotation(x=r["year_month"], y=y_max * 1.05, text=label, showarrow=True,
                               arrowhead=1, font=dict(size=9, color="#F9FAFB"))

    fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Monthly count",
                      legend_title_text="Metric", hovermode="x unified")
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Churn Lifecycle Framework")
st.sidebar.markdown(
    "**A Framework for Customer Churn Prediction and Lifecycle Management "
    "to Curtail Customer Attrition in Nigerian Digital Banks**"
)
st.sidebar.markdown("---")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Connectivity check only - actual data loading happens inside each tab
# function below, so only the SELECTED tab's queries ever run.
# ---------------------------------------------------------------------------
try:
    with engine.connect() as _conn:
        _conn.execute(text("SELECT 1"))
except Exception as e:
    st.error(f"Could not connect to the database: {e}")
    st.info("Check DATABASE_URL in .env, and that you've run the full pipeline "
            "(preprocessing -> modeling -> inference -> recommender).")
    st.stop()

st.markdown(
    """
    <style>
    div[role="radiogroup"] {
        gap: 2px;
        flex-wrap: nowrap !important;
        overflow-x: auto;
        overflow-y: hidden;
        padding-bottom: 2px;
    }
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.15);
        border-radius: 8px 8px 0 0; padding: 6px 12px; margin-right: 2px;
        white-space: nowrap; flex-shrink: 0;
    }
    div[role="radiogroup"] label p { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
NAV_OPTIONS = [
    "Portfolio Overview", "Customer Risk Explorer", "Customer Explainability",
    "Retention Recommendations", "Model Performance",
]
selected_tab = st.radio(
    "Navigation", NAV_OPTIONS,
    horizontal=True, label_visibility="collapsed",
)
st.markdown("---")

# ===========================================================================
# TAB 1 - Portfolio Overview
# ===========================================================================
def render_overview():
    predictions_df = load_predictions()
    st.header("Portfolio Overview")

    total = len(predictions_df)
    high = (predictions_df["risk_tier"] == "High Risk").sum()
    medium = (predictions_df["risk_tier"] == "Medium Risk").sum()
    low = (predictions_df["risk_tier"] == "Low Risk").sum()
    avg_prob = predictions_df["churn_probability"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("Total Customers", f"{total:,}", "#94A3B8"), unsafe_allow_html=True)
    c2.markdown(metric_card("High Risk", f"{high:,}", TIER_COLOURS["High Risk"], f"{high/total*100:.1f}% of portfolio"), unsafe_allow_html=True)
    c3.markdown(metric_card("Medium Risk", f"{medium:,}", TIER_COLOURS["Medium Risk"], f"{medium/total*100:.1f}% of portfolio"), unsafe_allow_html=True)
    c4.markdown(metric_card("Low Risk", f"{low:,}", TIER_COLOURS["Low Risk"], f"{low/total*100:.1f}% of portfolio"), unsafe_allow_html=True)
    c5.markdown(metric_card("Avg Risk Score", f"{avg_prob:.3f}", "#5FDFDC"), unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Risk Tier Distribution")
        tier_counts = predictions_df["risk_tier"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"])
        fig = px.pie(values=tier_counts.values, names=tier_counts.index, color=tier_counts.index,
                     color_discrete_map=TIER_COLOURS, hole=0.4)
        fig.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Average Risk Score by Tier")
        avg_by_tier = predictions_df.groupby("risk_tier")["churn_probability"].mean().reset_index()
        avg_by_tier.columns = ["risk_tier", "avg_score"]
        fig = sorted_bar(avg_by_tier, x="risk_tier", y="avg_score", color="risk_tier", color_discrete_map=TIER_COLOURS, label_format="score")
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Average Risk Score", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Cohort Distribution")
    cohort_counts = predictions_df["final_cohort"].value_counts().reset_index()
    cohort_counts.columns = ["final_cohort", "n"]
    fig = sorted_bar(cohort_counts, x="final_cohort", y="n", color="final_cohort",
                      color_discrete_sequence=px.colors.sequential.Teal_r)
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Churn Pathway Breakdown")
    pathway_counts = predictions_df["churn_pathway"].value_counts().reset_index()
    pathway_counts.columns = ["churn_pathway", "n"]
    fig = sorted_bar(pathway_counts, x="churn_pathway", y="n", color="churn_pathway", color_discrete_map=PATHWAY_COLOURS)
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Cohort vs. Risk Tier")
    cohort_tier = predictions_df.groupby(["final_cohort", "risk_tier"]).size().reset_index(name="count")
    cohort_tier["_label"] = cohort_tier["count"].apply(format_count_label)
    cohort_order = predictions_df.groupby("final_cohort").size().sort_values(ascending=False).index.tolist()
    fig = px.bar(cohort_tier, x="final_cohort", y="count", color="risk_tier", text="_label",
                 color_discrete_map=TIER_COLOURS, barmode="group",
                 category_orders={"final_cohort": cohort_order})
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, xaxis_title="", yaxis_title="Number of customers", legend_title_text="Predicted Tier")
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    with st.expander("Deeper cuts - risk by segment, and population patterns", expanded=False):
        st.markdown("**Predicted risk tiers by segment**")
        col_seg, col_type = st.columns(2)
        with col_seg:
            seg_tier = predictions_df.groupby(["customer_value_segment", "risk_tier"]).size().reset_index(name="count")
            seg_tier["_label"] = seg_tier["count"].apply(format_count_label)
            order = predictions_df.groupby("customer_value_segment").size().sort_values(ascending=False).index.tolist()
            fig = px.bar(seg_tier, x="customer_value_segment", y="count", color="risk_tier", text="_label",
                         color_discrete_map=TIER_COLOURS, barmode="group",
                         category_orders={"customer_value_segment": order})
            fig.update_traces(textposition="outside")
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
        with col_type:
            type_tier = predictions_df.groupby(["account_type", "risk_tier"]).size().reset_index(name="count")
            type_tier["_label"] = type_tier["count"].apply(format_count_label)
            order = predictions_df.groupby("account_type").size().sort_values(ascending=False).index.tolist()
            fig = px.bar(type_tier, x="account_type", y="count", color="risk_tier", text="_label",
                         color_discrete_map=TIER_COLOURS, barmode="group",
                         category_orders={"account_type": order})
            fig.update_traces(textposition="outside")
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")
        st.markdown("**High Risk rate by segment**")
        pat1, pat2 = st.columns(2)
        with pat1:
            predictions_df["is_high_risk"] = (predictions_df["risk_tier"] == "High Risk").astype(int)
            seg_stats = predictions_df.groupby("customer_value_segment")["is_high_risk"].agg(rate="mean", total="count").reset_index()
            seg_stats["High Risk rate (%)"] = (seg_stats["rate"] * 100).round(1)
            seg_stats["_label"] = seg_stats["High Risk rate (%)"].apply(lambda v: f"{v:.1f}%")
            seg_stats = seg_stats.sort_values("High Risk rate (%)", ascending=False)
            fig = px.bar(seg_stats, x="customer_value_segment", y="High Risk rate (%)", text="_label",
                         color="customer_value_segment", color_discrete_sequence=px.colors.sequential.Teal_r)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, showlegend=False, margin=dict(t=40, b=40),
                              xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig, width="stretch")
        with pat2:
            loc_stats = predictions_df.groupby("location")["is_high_risk"].agg(rate="mean", total="count").reset_index()
            loc_stats["High Risk rate (%)"] = (loc_stats["rate"] * 100).round(1)
            loc_stats["_label"] = loc_stats["High Risk rate (%)"].apply(lambda v: f"{v:.1f}%")
            loc_stats = loc_stats.sort_values("High Risk rate (%)", ascending=False)
            fig = px.bar(loc_stats, x="location", y="High Risk rate (%)", text="_label",
                         color="location", color_discrete_sequence=px.colors.sequential.Teal_r)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, showlegend=False, margin=dict(t=40, b=40),
                              xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig, width="stretch")

# ===========================================================================
# TAB 2 - Customer Risk Explorer
# ===========================================================================
def render_explorer():
    predictions_df = load_predictions()
    recommendations_df = load_recommendations()
    st.header("Customer Risk Explorer")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        tier_filter = st.multiselect("Risk Tier", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk"])
    with fc2:
        pathway_filter = st.multiselect("Churn Pathway", options=sorted(predictions_df["churn_pathway"].unique()), default=[])
    with fc3:
        cohort_filter = st.multiselect("Cohort", options=sorted(predictions_df["final_cohort"].unique()), default=[])
    with fc4:
        min_prob = st.slider("Min. churn probability", 0.0, 1.0, 0.0, 0.05)

    search_id = st.text_input("Search by Customer ID", placeholder="e.g. CUST00042")

    filtered = predictions_df.copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if pathway_filter:
        filtered = filtered[filtered["churn_pathway"].isin(pathway_filter)]
    if cohort_filter:
        filtered = filtered[filtered["final_cohort"].isin(cohort_filter)]
    if search_id:
        filtered = filtered[filtered["customer_id"].str.contains(search_id, case=False, na=False)]
    filtered = filtered[filtered["churn_probability"] >= min_prob]
    filtered = filtered.sort_values("churn_probability", ascending=False)

    st.markdown(f"**Showing {len(filtered):,} of {len(predictions_df):,} customers**")

    display_filtered = filtered.copy()
    display_filtered["Tier"] = display_filtered["risk_tier"].map(lambda t: f"{TIER_EMOJI.get(t, '⚪')} {t}")

    show_cols = ["customer_id", "location", "gender", "age", "tenure_years", "products_held",
                 "customer_value_segment", "month_end_balance_naira", "churn_probability",
                 "churn_pathway", "Tier"]
    st.dataframe(
        display_filtered[show_cols].head(200),
        width="stretch", hide_index=True,
        column_config={
            "churn_probability": st.column_config.NumberColumn("Risk Score", format="%.3f"),
            "month_end_balance_naira": st.column_config.NumberColumn("Balance (NGN)", format="%.0f"),
            "tenure_years": st.column_config.NumberColumn("Tenure (yrs)", format="%.1f"),
        },
    )
    if len(filtered) > 200:
        st.caption("Showing first 200 rows. Refine filters to see more specific results.")

    st.markdown("---")
    st.subheader("Customer Deep Dive")
    if len(filtered):
        cid_options = filtered["customer_id"].tolist()[:200]
        selected_cid = st.selectbox("Select Customer ID for detail view", options=cid_options)
        row = filtered[filtered["customer_id"] == selected_cid].iloc[0]

        expected_tier = COHORT_TO_EXPECTED_TIER.get(row["final_cohort"], "Unknown")
        matches = row["risk_tier"] == expected_tier

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Predicted Risk Tier", row["risk_tier"])
        dc2.metric("Risk Score", f"{row['churn_probability']:.4f}")
        dc3.metric("Current Status", row["account_status"])
        dc4.metric("Ground Truth Cohort", row["final_cohort"])
        if not matches:
            st.warning(
                f"**Model disagreement:** predicted {row['risk_tier']}, but this customer's cohort "
                f"({row['final_cohort']}) would suggest {expected_tier}. Worth reviewing their SHAP "
                "explanation in the Customer Explainability tab to see what drove this prediction."
            )

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("**Demographics**")
            st.write(
                f"- Age: {int(row['age'])}\n"
                f"- Gender: {row['gender']}\n"
                f"- Location: {row['location']}\n"
                f"- Value segment: {row['customer_value_segment']}\n"
                f"- Churn pathway: {row['churn_pathway']}"
            )
        with detail_cols[1]:
            st.markdown("**Banking Profile**")
            salary = f"{row['avg_monthly_salary_naira']:,.0f}" if pd.notna(row["avg_monthly_salary_naira"]) else "N/A"
            st.write(
                f"- Products: {int(row['products_held'])}\n"
                f"- Tenure: {row['tenure_years']:.1f} years\n"
                f"- Balance: NGN {row['month_end_balance_naira']:,.2f}\n"
                f"- Est. monthly salary: NGN {salary}\n"
                f"- 3-mo avg. transactions: {row['transaction_count_roll3_avg']:.1f}"
            )

        rec_row = recommendations_df[recommendations_df["customer_id"] == selected_cid]
        if len(rec_row):
            r = rec_row.iloc[0]
            st.markdown(f"**Recommendation:** `{r['recommendation_type']}`")
            st.info(r["recommendation_text"])
            st.caption(f"Rationale: {r['rationale']}")

        st.markdown("---")
        st.subheader("Lifecycle Progression")
        render_trajectory_chart(selected_cid, key="explorer_trajectory")
    else:
        st.info("No customers match the current filters.")

# ===========================================================================
# TAB 3 - Customer Explainability
# ===========================================================================
def render_explainability():
    st.header("Customer Explainability")

    explanations_df = load_explanations_summary()

    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        exp_tier = st.multiselect("Risk Tier", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk"], key="exp_tier")
    with sf2:
        exp_min_prob = st.slider("Min. churn probability", 0.0, 1.0, 0.0, 0.05, key="exp_min_prob")
    with sf3:
        exp_search = st.text_input("Search by Customer ID", placeholder="e.g. CUST00042", key="exp_search")

    filtered_exp = explanations_df.copy()
    if exp_tier:
        filtered_exp = filtered_exp[filtered_exp["risk_tier"].isin(exp_tier)]
    filtered_exp = filtered_exp[filtered_exp["churn_probability"] >= exp_min_prob]
    if exp_search:
        filtered_exp = filtered_exp[filtered_exp["customer_id"].str.contains(exp_search, case=False, na=False)]
    filtered_exp = filtered_exp.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    with st.spinner("Building plain-English explanations..."):
        display_rows = filtered_exp.head(200).copy()
        # one batched query for the ~200 visible customers, not all 10,000
        shap_lookup = load_shap_for_customers(tuple(display_rows["customer_id"]))
        risk_list, protective_list = [], []
        for _, row in display_rows.iterrows():
            contribs = shap_lookup.get(row["customer_id"])
            r, p = build_customer_explanation(row, contribs)
            risk_list.append(format_risk_narrative(r))
            protective_list.append(format_protective_narrative(p, int(row.get("products_held", 0)), row.get("account_status") == "Active"))
        display_rows["Why at risk"] = risk_list
        display_rows["What's protecting them"] = protective_list

    st.markdown(f"**Showing {min(len(filtered_exp), 200):,} of {len(explanations_df):,} customers** (top 200 by churn probability; refine filters to narrow further)")

    display_rows["Tier"] = display_rows["risk_tier"].map(lambda t: f"{TIER_EMOJI.get(t, '⚪')} {t}")
    show_cols = ["customer_id", "churn_probability", "Tier", "churn_pathway", "Why at risk", "What's protecting them", "recommendation_type"]
    st.dataframe(
        display_rows[show_cols], width="stretch", hide_index=True,
        column_config={
            "customer_id": "Customer ID",
            "churn_probability": st.column_config.NumberColumn("Risk Score", format="%.3f"),
            "Why at risk": st.column_config.TextColumn("Why at risk", width="large"),
            "What's protecting them": st.column_config.TextColumn("What's protecting them", width="medium"),
            "recommendation_type": st.column_config.TextColumn("Recommended action", width="medium"),
        },
    )

    st.markdown("---")
    st.subheader("Customer Detail")
    if len(display_rows):
        selected_cid = st.selectbox("Select a Customer ID for the full explanation", options=display_rows["customer_id"].tolist(), key="exp_selected_cid")
        detail = display_rows[display_rows["customer_id"] == selected_cid].iloc[0]

        dc1, dc2, dc3 = st.columns([1, 1, 2])
        with dc1:
            st.markdown(f"**{detail['customer_id']}**")
            st.caption(f"Cohort: {detail['final_cohort']}")
            st.metric("Risk Tier", detail["risk_tier"])
            st.metric("Risk Score", f"{detail['churn_probability']:.3f}")
            st.metric("Churn Pathway", detail["churn_pathway"])

        with dc2:
            st.markdown("**Why this customer is flagged**")
            risk_text = detail["Why at risk"]
            if "No specific risk drivers" in risk_text:
                st.caption(risk_text)
            else:
                for factor in risk_text.split(" · "):
                    st.markdown(f"⚠️ {factor}")
            st.markdown("**What's protecting them**")
            prot_text = detail["What's protecting them"]
            if "No strong protective factors" in prot_text:
                st.caption(prot_text)
            else:
                for factor in prot_text.split(" · "):
                    st.markdown(f"✅ {factor}")

        with dc3:
            st.markdown("**Recommended action**")
            st.markdown(f"`{detail['recommendation_type']}`")
            if pd.notna(detail.get("recommendation_text")):
                st.info(detail["recommendation_text"])
            if pd.notna(detail.get("rationale")):
                st.caption(f"Rationale: {detail['rationale']}")

        st.markdown("---")
        st.markdown("**Behavioural Trend**")
        render_trajectory_chart(selected_cid, key="explainability_trajectory")

        with st.expander("Technical SHAP breakdown (for analysts)", expanded=False):
            contribs = shap_lookup.get(detail["customer_id"]) or {}
            if contribs:
                contrib_df = pd.DataFrame([{"feature": k, "shap_value": v} for k, v in contribs.items()])
                contrib_df["abs_value"] = contrib_df["shap_value"].abs()
                contrib_df = contrib_df.sort_values("abs_value", ascending=True).tail(15)
                contrib_df["direction"] = contrib_df["shap_value"].apply(lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk")
                fig = px.bar(contrib_df, x="shap_value", y="feature", orientation="h", color="direction",
                             color_discrete_map={"Increases churn risk": "#EF4444", "Decreases churn risk": "#14B8A6"})
                fig.update_layout(height=500, xaxis_title="SHAP value (log-odds)")
                st.plotly_chart(fig, width="stretch")
    else:
        st.info("No customers match the current filters.")

    st.markdown("---")
    with st.expander("Aggregate feature importance (all customers)", expanded=False):
        selected_class = st.radio("Risk tier", RISK_LABEL_CLASSES, horizontal=True, key="shap_class_selector")
        slug = selected_class.lower().replace(" ", "_")
        global_bar = _PROJECT_ROOT / "outputs" / "figures" / f"shap_global_bar_{slug}.png"
        global_bee = _PROJECT_ROOT / "outputs" / "figures" / f"shap_global_summary_{slug}.png"
        sc1, sc2 = st.columns(2)
        with sc1:
            if global_bar.exists():
                st.image(str(global_bar), caption=f"Mean absolute SHAP per feature - {selected_class}")
            else:
                st.warning("Run `python -m src.explainability` first.")
        with sc2:
            if global_bee.exists():
                st.image(str(global_bee), caption=f"Feature contribution distribution - {selected_class}")
            else:
                st.warning("Global beeswarm plot not found.")
        fi = load_shap_feature_importance()
        if fi is not None:
            if "risk_class" in fi.columns:
                fi = fi[fi["risk_class"] == selected_class]
            st.dataframe(fi, width="stretch", hide_index=True)

# ===========================================================================
# TAB 4 - Retention Recommendations
# ===========================================================================
def render_recommendations():
    recommendations_df = load_recommendations()
    st.header("Retention Recommendations")

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        rec_tier = st.multiselect("Filter by tier", options=["High Risk", "Medium Risk", "Low Risk"], default=["High Risk"], key="rec_tier")
    with rc2:
        rec_pathway = st.multiselect("Filter by pathway", options=sorted(recommendations_df["churn_pathway"].unique()), default=[], key="rec_pathway")
    with rc3:
        rec_types = st.multiselect("Filter by recommendation type", options=sorted(recommendations_df["recommendation_type"].unique()), default=[])
    with rc4:
        search_id = st.text_input("Search Customer ID", placeholder="e.g. CUST00042", key="rec_search")

    filtered_rec = recommendations_df.copy()
    if rec_tier:
        filtered_rec = filtered_rec[filtered_rec["risk_tier"].isin(rec_tier)]
    if rec_pathway:
        filtered_rec = filtered_rec[filtered_rec["churn_pathway"].isin(rec_pathway)]
    if rec_types:
        filtered_rec = filtered_rec[filtered_rec["recommendation_type"].isin(rec_types)]
    if search_id:
        filtered_rec = filtered_rec[filtered_rec["customer_id"].str.contains(search_id, case=False, na=False)]

    st.markdown(f"**Showing {len(filtered_rec):,} of {len(recommendations_df):,} recommendations**")

    st.subheader("Recommendation Mix")
    mix = filtered_rec.groupby(["churn_pathway", "recommendation_type"]).size().reset_index(name="count")
    mix["_label"] = mix["count"].apply(format_count_label)
    mix = mix.sort_values("count", ascending=True)
    show_legend = mix["churn_pathway"].nunique() > 1
    fig = px.bar(mix, x="count", y="recommendation_type", color="churn_pathway", color_discrete_map=PATHWAY_COLOURS,
                 orientation="h", text="_label", labels={"recommendation_type": "", "count": "Number of customers", "churn_pathway": "Pathway"})
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_layout(height=max(320, 45 * len(mix) + 80), showlegend=show_legend, margin=dict(t=30, b=40, l=10, r=60))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recommendation List")
    st.dataframe(
        filtered_rec[["customer_id", "risk_tier", "churn_pathway", "recommendation_type", "recommendation_text"]].head(500),
        width="stretch", hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer ID", width="small"),
            "risk_tier": st.column_config.TextColumn("Tier", width="small"),
            "churn_pathway": st.column_config.TextColumn("Pathway", width="medium"),
            "recommendation_type": st.column_config.TextColumn("Type", width="medium"),
            "recommendation_text": st.column_config.TextColumn("Recommendation", width="large"),
        },
    )
    if len(filtered_rec) > 500:
        st.caption("Showing first 500 rows. Refine filters to see specific segments.")

    if len(filtered_rec):
        st.markdown("---")
        st.subheader("Recommendation Detail")
        selected_cid = st.selectbox("Select Customer ID", options=filtered_rec["customer_id"].tolist()[:500])
        detail = filtered_rec[filtered_rec["customer_id"] == selected_cid].iloc[0]
        st.markdown(f"**{detail['recommendation_type']}**")
        st.info(detail["recommendation_text"])
        st.caption(f"Rationale: {detail['rationale']}")

# ===========================================================================
# TAB 5 - Model Performance
# ===========================================================================
def render_performance():
    model_metadata_df = load_model_metadata()
    st.header("Model Performance")

    st.subheader("Metrics Comparison")
    if len(model_metadata_df):
        display_df = model_metadata_df.copy()
        display_df["Selected"] = display_df["is_selected"].map({True: "✅", False: ""})
        metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        for c in metric_cols:
            display_df[c] = display_df[c].apply(lambda v: f"{v:.2f}")
        st.dataframe(
            display_df[["model_name", "accuracy", "precision", "recall", "f1_score", "auc_roc", "Selected"]],
            width="stretch", hide_index=True,
        )
    else:
        st.info("No model metadata found - run `python -m src.modeling` first.")

    st.markdown("---")
    fig_dir = _PROJECT_ROOT / "outputs" / "figures"
    st.subheader("Metric Comparison Chart")
    p = fig_dir / "metric_comparison.png"
    if p.exists():
        st.image(str(p))
    else:
        st.warning("Run `python -m src.evaluation` to generate this figure.")

    st.markdown("---")
    st.subheader("ROC Curves")
    p = fig_dir / "roc_curves.png"
    if p.exists():
        st.image(str(p))
    else:
        st.warning("Run `python -m src.evaluation` to generate this figure.")

    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_files = {
        "Logistic Regression": fig_dir / "confusion_matrix_logistic_regression.png",
        "Random Forest": fig_dir / "confusion_matrix_random_forest.png",
        "XGBoost": fig_dir / "confusion_matrix_xgboost.png",
        "LightGBM": fig_dir / "confusion_matrix_lightgbm.png",
    }
    cm_cols = st.columns(2)
    for i, (name, path) in enumerate(cm_files.items()):
        with cm_cols[i % 2]:
            st.markdown(f"**{name}**")
            if path.exists():
                st.image(str(path))
            else:
                st.warning(f"Missing: {path.name}")

    st.markdown("---")
    st.subheader("Best Hyperparameters")
    for _, row in model_metadata_df.iterrows():
        selected_flag = " ✅ SELECTED" if row["is_selected"] else ""
        st.markdown(f"**{row['model_name']}{selected_flag}**")
        st.json(row["hyperparameters"])


# ---------------------------------------------------------------------------
# Dispatch - only the SELECTED tab's function runs, so only its queries and
# computations execute. This is the actual fix for the slow-load issue:
# Streamlit's native st.tabs() runs every tab's code on every single page
# load regardless of which one is visible; a manual radio + function-per-tab
# means unselected tabs cost nothing at all.
# ---------------------------------------------------------------------------
if selected_tab == "Portfolio Overview":
    render_overview()
elif selected_tab == "Customer Risk Explorer":
    render_explorer()
elif selected_tab == "Customer Explainability":
    render_explainability()
elif selected_tab == "Retention Recommendations":
    render_recommendations()
elif selected_tab == "Model Performance":
    render_performance()


# ---------------------------------------------------------------------------
# Making Update to the fileS
# ---------------------------------------------------------------------------