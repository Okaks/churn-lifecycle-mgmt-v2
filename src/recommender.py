"""
Retention Recommendation Rule Engine.

Implements the per-customer retention recommendation step (Algorithm 3.2,
step 7). v2's rule engine is organised primarily around churn_pathway, not
just risk_tier - this is the actual lifecycle-management contribution of
the framework: a "Never Activated" customer and a "Declined After
Engagement" customer at the same risk_tier need completely different
interventions (onboarding nudge vs. win-back), and a customer who is
still technically "Not Churned" but has a high model-predicted probability
(an Early Drift customer the model caught before official churn) needs a
third, distinct tone entirely: proactive retention, not win-back.

Inputs per customer: {risk_tier, churn_pathway, engagement_band,
products_held}. Engagement band is percentile-based on
transaction_count_roll3_avg (see config.ENGAGEMENT_SCORE_COLUMN) across the
entire customer set:
    LOW    = bottom 25%
    HIGH   = top 25%
    MEDIUM = middle 50%

Rules are designed for Nigerian digital-first banks (Opay, Kuda, PalmPay,
Moniepoint, Sterling, Carbon, etc.): interventions route through in-app
messaging, USSD codes, airtime rewards, cashback, fee waivers, and referral
programs.

Priority-based rule matching: first matching rule per (pathway, tier) wins.

Run standalone (after inference has been executed):
    python -m src.recommender
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import insert, text

from src.config import (
    ENGAGEMENT_SCORE_COLUMN,
    RISK_TIERS,
    TABLE_LATEST_SNAPSHOT,
    TABLE_PREDICTIONS,
    TABLE_RECOMMENDATIONS,
)
from src.database import engine, recommendations as recommendations_table

PATHWAY_NEVER_ACTIVATED = "Never Activated"
PATHWAY_DECLINED = "Declined After Engagement"
PATHWAY_NOT_CHURNED = "Not Churned"


# ---------------------------------------------------------------------------
# Engagement bands (percentile-based over the whole customer set)
# ---------------------------------------------------------------------------
def compute_engagement_bands(scores: pd.Series) -> tuple[float, float]:
    """Return (low_cut, high_cut) - the 25th and 75th percentiles."""
    return float(scores.quantile(0.25)), float(scores.quantile(0.75))


def band_of(score: float, low_cut: float, high_cut: float) -> str:
    if score <= low_cut:
        return "LOW"
    if score >= high_cut:
        return "HIGH"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Recommendation catalogue
# ---------------------------------------------------------------------------
# Each entry: (recommendation_type, customer_facing_text, rationale)

# --- Never Activated: the goal is activation, not retention. These
# customers are Medium Risk (not High) - they're active, just haven't
# transacted yet - but the copy is deliberately urgent: every month they
# stay unconverted is a month closer to becoming a true Inactive/Dormant
# case, and urgency is what actually drives someone to act on a "make your
# first transaction" nudge.
REC_ACTIVATION_BONUS = (
    "ACTIVATION_FIRST_TXN_BONUS",
    "Your account is set up but still inactive - and inactive accounts get "
    "closed. Make ANY transaction in the next 7 days and get N1,000 "
    "credited instantly, no minimum amount required.",
    "Never-activated customers need a low-friction reason to make their "
    "first transaction, and a real deadline. An open-ended nudge is easy "
    "to ignore; a 7-day window with a concrete reward is not.",
)

REC_ACTIVATION_GUIDED_SETUP = (
    "ACTIVATION_GUIDED_SETUP",
    "You're one step away from a fully active account. Complete your "
    "profile in the app now (2 minutes) to unlock transfers, bill "
    "payments, and your virtual card before your setup window closes.",
    "Customer has some product depth but hasn't transacted meaningfully; "
    "the blocker is likely an incomplete setup step, not lack of "
    "interest - urgency plus a concrete next step closes that gap.",
)

REC_ACTIVATION_USSD_FIRST_TXN = (
    "ACTIVATION_USSD_FIRST_TXN",
    "Act now: dial *966*000# today to buy airtime or check your balance - "
    "no app or internet needed. Your first USSD transaction this week "
    "earns N200 airtime, but the offer expires in 7 days.",
    "Low-engagement never-activated customers may be less comfortable "
    "with the app; USSD works on any phone and a real deadline converts "
    "passive interest into action.",
)

# --- Declined After Engagement: win-back, they were active before ---
REC_HIGH_VALUE_WINBACK = (
    "HIGH_VALUE_WINBACK",
    "We've noticed you've stepped back, and we'd like to understand why. "
    "As one of our valued customers, expect a call from our Priority Care "
    "team within 48 hours with an offer tailored to you.",
    "High-risk customer with 3+ products represented significant share of "
    "wallet before declining; personalised human contact is worth the cost "
    "to preserve the multi-product relationship.",
)

REC_USSD_REENGAGE = (
    "USSD_REENGAGE",
    "We miss you. Dial *966*456# from your registered mobile number this "
    "week to claim N500 free airtime credit on your first transaction back.",
    "Customer has gone quiet after a period of low engagement; USSD works "
    "on any phone and airtime is a universally valued reward for a "
    "low-friction return.",
)

REC_PRODUCT_DIVERSIFY_WINBACK = (
    "PRODUCT_DIVERSIFY_WINBACK",
    "Open our high-yield Savings account with zero maintenance fees for "
    "the next 6 months. Get started in under 2 minutes in the app.",
    "Single-product customers who've declined are the highest-churn "
    "segment; deepening the relationship on their return strengthens "
    "retention against a repeat decline.",
)

REC_PRIORITY_WINBACK_OUTREACH = (
    "PRIORITY_WINBACK_OUTREACH",
    "A senior retention specialist will reach out via in-app chat within "
    "48 hours with a personalised offer to welcome you back.",
    "High-risk fallback for declined customers not covered by the more "
    "specific rules - direct human contact.",
)

REC_ACTIVITY_CASHBACK = (
    "ACTIVITY_CASHBACK",
    "Complete any 3 transactions this month and earn 1% cashback on your "
    "next N10,000 of bill payments and airtime purchases.",
    "Medium-risk declining customer needs a reason to transact again; "
    "cashback ties the reward to the exact behaviour we want to reinforce.",
)

REC_FEE_WAIVER = (
    "FEE_WAIVER",
    "Enjoy zero transfer fees and free ATM withdrawals for the next 30 "
    "days, on us.",
    "Fee sensitivity is a leading disengagement driver in Nigerian digital "
    "banking; a temporary waiver removes friction and demonstrates value "
    "as they reconsider the relationship.",
)

REC_PRIORITY_SUPPORT_WINBACK = (
    "PRIORITY_SUPPORT_WINBACK",
    "You've been upgraded to Priority Support: dedicated support line, "
    "reduced wait times, and faster dispute resolution.",
    "Medium-risk fallback for declined customers - service quality "
    "improvements retain customers not motivated by cash-based rewards.",
)

# --- Not Churned but elevated risk: proactive, BEFORE they leave ---
REC_EARLY_INTERVENTION_CHECKIN = (
    "EARLY_INTERVENTION_CHECKIN",
    "How's your experience with us been lately? Tell us in a 30-second "
    "in-app survey and get N300 airtime as a thank-you - your feedback "
    "shapes what we build next.",
    "Customer shows early drift signals (declining bills/transfers, "
    "narrowing channel use) but has NOT yet officially churned - this is "
    "the highest-leverage moment to intervene, before disengagement "
    "hardens into departure.",
)

REC_EARLY_LITERACY_NUDGE = (
    "EARLY_LITERACY_NUDGE",
    "New to investing? Our 5-minute in-app guide shows you how to start "
    "with as little as N1,000, with zero opening fees this month.",
    "Single-product customer showing early drift responds well to "
    "educational content paired with a low-friction next step, deepening "
    "the relationship before risk escalates.",
)

REC_EARLY_PRIORITY_SUPPORT = (
    "EARLY_PRIORITY_SUPPORT",
    "You've been upgraded to Priority Support: dedicated support line and "
    "faster response times, on us.",
    "Elevated-risk fallback for customers still active but showing "
    "warning signs - a service-quality improvement addresses friction "
    "before it becomes a reason to leave.",
)

# --- Not Churned, low risk: standard loyalty/growth plays ---
REC_REFERRAL_INVITE = (
    "REFERRAL_INVITE",
    "Refer a friend and you both earn N2,500 when they complete their "
    "first transaction. Share your referral code from the app in one tap.",
    "Engaged multi-product customers are the ideal advocacy segment; "
    "referral rewards deepen loyalty and grow acquisition simultaneously.",
)

REC_SAVINGS_AUTO_ENROLL = (
    "SAVINGS_AUTO_ENROLL",
    "Set a savings goal in 60 seconds and we'll auto-save a small amount "
    "from every transaction. Reach your goal faster with our 12% p.a. rate.",
    "Active, engaged low-risk customers benefit from a nudge toward "
    "goal-based savings, which increases stickiness without asking them "
    "to change their transacting behaviour.",
)

REC_LOYALTY_REWARDS = (
    "LOYALTY_REWARDS",
    "Enjoy 1% cashback on all bill payments and airtime purchases this "
    "month, plus N200 free airtime every time you pay your electricity bill.",
    "Low-risk fallback - a steady stream of small rewards on habitual "
    "transactions maintains satisfaction without heavy operational cost.",
)


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------
def match_recommendation(
    risk_tier: str,
    churn_pathway: str,
    engagement_band: str,
    products_held: int,
) -> tuple[str, str, str]:
    """Return (type, message, rationale) for a customer's inputs."""

    if churn_pathway == PATHWAY_NEVER_ACTIVATED:
        if products_held <= 1 and engagement_band == "LOW":
            return REC_ACTIVATION_BONUS
        if engagement_band == "LOW":
            return REC_ACTIVATION_USSD_FIRST_TXN
        return REC_ACTIVATION_GUIDED_SETUP

    if churn_pathway == PATHWAY_DECLINED:
        if risk_tier == RISK_TIERS["HIGH"]:
            if products_held >= 3:
                return REC_HIGH_VALUE_WINBACK
            if engagement_band == "LOW":
                return REC_USSD_REENGAGE
            if products_held == 1:
                return REC_PRODUCT_DIVERSIFY_WINBACK
            return REC_PRIORITY_WINBACK_OUTREACH
        if risk_tier == RISK_TIERS["MEDIUM"]:
            if engagement_band == "LOW":
                return REC_ACTIVITY_CASHBACK
            if products_held == 1:
                return REC_FEE_WAIVER
            return REC_PRIORITY_SUPPORT_WINBACK
        return REC_LOYALTY_REWARDS  # rare edge case: declined but scored low risk

    # PATHWAY_NOT_CHURNED
    if risk_tier in (RISK_TIERS["HIGH"], RISK_TIERS["MEDIUM"]):
        # elevated model risk despite no official churn yet - Early Drift caught early
        if products_held == 1:
            return REC_EARLY_LITERACY_NUDGE
        if engagement_band == "LOW":
            return REC_EARLY_INTERVENTION_CHECKIN
        return REC_EARLY_PRIORITY_SUPPORT

    # low risk, not churned
    if engagement_band == "HIGH" and products_held >= 2:
        return REC_REFERRAL_INVITE
    if engagement_band == "HIGH":
        return REC_SAVINGS_AUTO_ENROLL
    return REC_LOYALTY_REWARDS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_customers_with_predictions() -> pd.DataFrame:
    """Join predictions to their latest_snapshot row."""
    query = f"""
        SELECT
            s.customer_id,
            s.products_held,
            s.churn_pathway,
            s."{ENGAGEMENT_SCORE_COLUMN}" AS engagement_score,
            p.risk_tier
        FROM "{TABLE_LATEST_SNAPSHOT}" s
        JOIN "{TABLE_PREDICTIONS}" p ON p.customer_id = s.customer_id
    """
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} customers with predictions.")
    return df


# ---------------------------------------------------------------------------
# Write recommendations
# ---------------------------------------------------------------------------
def write_recommendations(rows: list[dict]) -> None:
    """Truncate and repopulate the recommendations table."""
    with engine.begin() as conn:
        conn.execute(text(f'TRUNCATE "{TABLE_RECOMMENDATIONS}" RESTART IDENTITY'))
        for start in range(0, len(rows), 1000):
            conn.execute(insert(recommendations_table), rows[start : start + 1000])
    print(f"Wrote {len(rows):,} rows to '{TABLE_RECOMMENDATIONS}'.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("\nRecommendation distribution:")
    print(f"  {'Pathway':<28} {'Type':<28} {'Count':>7}  {'Share':>7}")
    print("  " + "-" * 75)
    total = len(df)
    for pathway in [PATHWAY_NEVER_ACTIVATED, PATHWAY_DECLINED, PATHWAY_NOT_CHURNED]:
        subset = df[df["churn_pathway"] == pathway]
        counts = subset["recommendation_type"].value_counts()
        for rec_type, count in counts.items():
            share = count / total * 100
            print(f"  {pathway:<28} {rec_type:<28} {count:>7,}  {share:>6.2f}%")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 78)
    print("Retention Recommendation Rule Engine")
    print("=" * 78)

    df = load_customers_with_predictions()

    print("\nComputing engagement bands (percentile-based)...")
    low_cut, high_cut = compute_engagement_bands(df["engagement_score"])
    print(
        f"  LOW    = {ENGAGEMENT_SCORE_COLUMN} <= {low_cut:.2f} (bottom 25%)\n"
        f"  MEDIUM = {low_cut:.2f} < {ENGAGEMENT_SCORE_COLUMN} < {high_cut:.2f}\n"
        f"  HIGH   = {ENGAGEMENT_SCORE_COLUMN} >= {high_cut:.2f} (top 25%)"
    )

    print("\nApplying rules to every customer...")
    timestamp = datetime.now(timezone.utc)
    rows = []
    df_summary = []
    for _, r in df.iterrows():
        band = band_of(r["engagement_score"], low_cut, high_cut)
        rec_type, message, rationale = match_recommendation(
            r["risk_tier"], r["churn_pathway"], band, int(r["products_held"])
        )
        rows.append({
            "customer_id": r["customer_id"],
            "risk_tier": r["risk_tier"],
            "churn_pathway": r["churn_pathway"],
            "recommendation_type": rec_type,
            "recommendation_text": message,
            "rationale": rationale,
            "generated_at": timestamp,
        })
        df_summary.append({"churn_pathway": r["churn_pathway"], "recommendation_type": rec_type})

    print(f"Generated {len(rows):,} recommendations.")

    print("\nWriting to Neon Postgres...")
    write_recommendations(rows)

    print_summary(pd.DataFrame(df_summary))

    print("\n" + "=" * 78)
    print("Recommendation generation complete.")
    print("=" * 78)


if __name__ == "__main__":
    run()
