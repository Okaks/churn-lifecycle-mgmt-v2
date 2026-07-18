"""
Generates synthetic behavioural data for churn-lifecycle-mgmt-v2 (schema v3).

Produces 2 CSVs in data/raw/:
    customer_profiles.csv   10,000 rows
    customer_monthly.csv    ~180-240K rows - one comprehensive row per
                             customer per active month (24 months for
                             full-tenure customers, fewer for late joiners -
                             NO row exists before a customer's join month).
Run:
    python scripts/generate_synthetic_data.py
"""

import os
import random

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_CUSTOMERS = 10_000
MONTHS = list(pd.date_range("2024-01-01", "2025-12-01", freq="MS").date)
N_MONTHS = len(MONTHS)  # 24

OUT_DIR = os.path.join("data", "raw")

# ---------------------------------------------------------------------------
# Reference distributions
# ---------------------------------------------------------------------------

COHORTS = ["Healthy", "Early Drift", "Inactive", "Dormant-or-Churned", "Growing", "New-Unconverted"]
COHORT_WEIGHTS = [0.48, 0.21, 0.13, 0.11, 0.04, 0.03]

AGE_BANDS = [(18, 29, 0.35), (30, 44, 0.40), (45, 59, 0.20), (60, 75, 0.05)]

LOCATIONS = [
    "Nigeria", "Ghana", "Benin Republic", "Togo",
    "Cote d'Ivoire", "Senegal", "Sierra Leone", "Liberia",
]
LOCATION_WEIGHTS = [0.85, 0.05, 0.03, 0.02, 0.02, 0.015, 0.0075, 0.0075]

NAIRA_TIERS = {
    "Heavy": (500_000, 5_000_000),
    "Regular": (50_000, 500_000),
    "Occasional": (5_000, 50_000),
    "Business": (2_000_000, 100_000_000),
}
TIER_NAMES = list(NAIRA_TIERS.keys())
TIER_WEIGHTS = [0.15, 0.55, 0.25, 0.05]

ACCOUNT_TYPES = ["Savings", "Current", "Business", "Loan", "Investment"]

VALUE_SEGMENTS_BY_TIER = {
    "Business": {"SME": 0.45, "Corporate": 0.25, "Business": 0.30},
    "Heavy": {"Affluent": 0.5, "Private": 0.2, "Salary": 0.3},
    "Regular": {"Salary": 0.6, "Mass": 0.3, "SME": 0.1},
    "Occasional": {"Student": 0.4, "Mass": 0.5, "Salary": 0.1},
}
SALARIED_SEGMENTS = {"Salary", "Corporate", "Private", "Affluent", "Mass"}

MALE_FIRST_NAMES = [
    "Chinedu", "Emeka", "Obinna", "Tunde", "Ayodele", "Segun", "Kunle", "Femi",
    "Musa", "Ibrahim", "Aliyu", "Suleiman", "Chukwuemeka", "Uche", "Ikenna",
    "Adewale", "Olumide", "Bayo", "Yusuf", "Kelechi", "Nnamdi", "Chidi",
    "Babatunde", "Godwin", "Emmanuel", "Kwame", "Kofi", "Ama", "Kojo", "Sani",
]
FEMALE_FIRST_NAMES = [
    "Adaeze", "Ngozi", "Amaka", "Chiamaka", "Funke", "Bisi", "Yetunde", "Aisha",
    "Fatima", "Zainab", "Halima", "Ijeoma", "Chinwe", "Blessing", "Grace",
    "Omolara", "Folake", "Kemi", "Abimbola", "Rita", "Comfort", "Patience",
    "Adaora", "Efe", "Amina", "Ama", "Efua", "Abena", "Rahama", "Hadiza",
]
SURNAMES = [
    "Okafor", "Eze", "Okonkwo", "Adeyemi", "Balogun", "Okoro", "Nwosu",
    "Abubakar", "Mohammed", "Bello", "Yakubu", "Adebayo", "Ogunleye",
    "Chukwu", "Nnaji", "Uzor", "Afolabi", "Oyelaran", "Danjuma", "Usman",
    "Owusu", "Mensah", "Asante", "Boateng", "Diallo", "Toure", "Camara",
    "Ojo", "Fashola", "Igwe",
]

DIGITAL_CHANNEL_CODES = {"App": 3, "USSD": 4, "POS": 1, "ATM": 2}  # matches customer_monthly.primary_digital_channel
CHANNEL_TO_COLUMN = {
    "App": "app_count", "USSD": "ussd_count", "POS": "pos_channel_count",
    "ATM": "atm_count", "Internet Banking": "internet_banking_count", "Branch": "branch_count",
}
ALL_CHANNELS = list(CHANNEL_TO_COLUMN.keys())
CHANNEL_BASE_WEIGHTS = {"App": 0.50, "USSD": 0.15, "POS": 0.15, "ATM": 0.10, "Internet Banking": 0.05, "Branch": 0.05}
CHANNEL_DIVERSITY_BY_TIER = {"Heavy": 5, "Business": 5, "Regular": 3, "Occasional": 2}

# 7 consolidated categories -> customer_monthly columns.
# bills_count now includes airtime + data (per explicit instruction: "airtime
# and data is merged as bill payment"). card_pos_count merges QR + POS + Card.
CATEGORY_BASE_BY_TIER = {
    "transfer_count":        {"Heavy": 15, "Regular": 8, "Occasional": 3, "Business": 20},
    "bills_count":           {"Heavy": 14, "Regular": 10, "Occasional": 4, "Business": 4},
    "card_pos_count":        {"Heavy": 17, "Regular": 9, "Occasional": 3.3, "Business": 17},
    "cash_withdrawal_count": {"Heavy": 3,  "Regular": 2, "Occasional": 2, "Business": 4},
}
# salary_credit_count, loan_repayment_count, investment_count are gated by
# segment/product, handled separately (not tier-based like the above four).


def weighted_choice(options, weights, size=None):
    return np.random.choice(options, size=size, p=weights)


def sample_age():
    band = random.choices(AGE_BANDS, weights=[b[2] for b in AGE_BANDS])[0]
    return random.randint(band[0], band[1])


# ---------------------------------------------------------------------------
# Step 1: customer_profiles
# ---------------------------------------------------------------------------

def assign_value_segment(tier, age):
    if age <= 23 and tier in ("Occasional", "Regular") and random.random() < 0.6:
        return "Student"
    options = VALUE_SEGMENTS_BY_TIER[tier]
    return random.choices(list(options.keys()), weights=list(options.values()))[0]


def generate_customer_profiles():
    rows = []
    cohorts = random.choices(COHORTS, weights=COHORT_WEIGHTS, k=NUM_CUSTOMERS)

    for i in range(NUM_CUSTOMERS):
        customer_id = f"CUST{i + 1:05d}"
        gender = random.choice(["Male", "Female"])
        first_name = random.choice(MALE_FIRST_NAMES if gender == "Male" else FEMALE_FIRST_NAMES)
        surname = random.choice(SURNAMES)
        age = sample_age()
        location = weighted_choice(LOCATIONS, LOCATION_WEIGHTS)
        tier = weighted_choice(TIER_NAMES, TIER_WEIGHTS)
        if tier == "Business":
            account_type = random.choices(["Business", "Loan", "Current"], weights=[0.70, 0.15, 0.15])[0]
        else:
            account_type = random.choices(
                ["Savings", "Current", "Loan", "Investment"], weights=[0.50, 0.25, 0.15, 0.10]
            )[0]
        cohort = cohorts[i]
        value_segment = assign_value_segment(tier, age)

        if value_segment in SALARIED_SEGMENTS:
            lo, hi = NAIRA_TIERS[tier]
            avg_monthly_salary = round(random.uniform(lo, hi) * random.uniform(0.5, 0.85), 2)
        else:
            avg_monthly_salary = None

        # tenure_years drives how many of the 24 months this customer has history for.
        if cohort == "New-Unconverted":
            tenure_years = round(random.uniform(0.1, 0.45), 1)
        elif cohort in ("Inactive", "Dormant-or-Churned"):
            tenure_years = round(random.uniform(2.0, 12.0), 1)  # >=24 months -> full window
        else:
            if random.random() < 0.85:
                tenure_years = round(random.uniform(2.0, 12.0), 1)
            else:
                tenure_years = round(random.uniform(0.2, 1.9), 1)

        rows.append({
            "customer_id": customer_id,
            "surname": surname,
            "first_name": first_name,
            "age": age,
            "gender": gender,
            "location": location,
            "tenure_years": tenure_years,
            "account_type": account_type,
            "starting_segment": tier,
            "customer_value_segment": value_segment,
            "avg_monthly_salary_naira": avg_monthly_salary,
            "final_cohort": cohort,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared trajectory helpers
# ---------------------------------------------------------------------------

def active_months_for(tenure_years):
    """Positions (0-indexed into MONTHS) this customer has a row for.
    A customer with short tenure has NO rows before their join month -
    not zero-value rows, genuinely absent rows."""
    months_of_history = min(N_MONTHS, max(1, round(tenure_years * 12)))
    if months_of_history >= N_MONTHS:
        return list(range(N_MONTHS))
    start_idx = N_MONTHS - months_of_history
    return list(range(start_idx, N_MONTHS))


def bank_status(months_since_last_txn):
    if months_since_last_txn is None:
        return "Active"
    if months_since_last_txn <= 6:
        return "Active"
    if months_since_last_txn <= 12:
        return "Inactive"
    return "Dormant"


def churn_points_for(cohort, n_months):
    """
    Returns (decline_start, churn_point). churn_point is the last month with
    real transaction activity; None means the customer never fully stops.

    Decline timing is spread wide: this is a slice of an ongoing customer
    relationship, so decline can plausibly begin as early as month 2,
    and churn_point (the actual full-stop) is pushed toward the LATER part
    of its valid range - giving a genuine multi-month "still transacting,
    but visibly declining" runway before the stop, not an instant collapse.
    """
    if cohort == "Dormant-or-Churned":
        max_gap = n_months - 1
        min_gap = 13  # bank_status() treats a gap of exactly 12 as still "Inactive"
        gap_hi = min(max_gap, min_gap + 5)
        target_gap = random.randint(min_gap, gap_hi) if gap_hi >= min_gap else max_gap
        churn_point = max(0, (n_months - 1) - target_gap)
        decline_start = max(0, churn_point - random.randint(4, 10))
        return decline_start, churn_point
    if cohort == "Inactive":
        target_gap = random.randint(7, 12)  # gap of exactly 6 is still "Active"
        churn_point = max(0, (n_months - 1) - target_gap)
        decline_start = max(0, churn_point - random.randint(4, 10))
        return decline_start, churn_point
    if cohort == "New-Unconverted":
        churn_point = random.randint(0, min(2, n_months - 1))
        return 0, churn_point
    if cohort == "Early Drift":
        lo = min(2, max(0, n_months - 1))
        hi = max(lo, n_months - 4)
        decline_start = random.randint(lo, hi)
        return decline_start, None
    return None, None  # Healthy, Growing


def initial_products_held(tier, account_type, cohort):
    if cohort == "New-Unconverted":
        return 1
    tier_extra_range = {"Business": (1, 4), "Heavy": (1, 3), "Regular": (0, 2), "Occasional": (0, 1)}[tier]
    total = 1 + random.randint(*tier_extra_range)
    if account_type in ("Loan", "Investment") and total < 2:
        total = 2
    return min(5, max(1, total))


# ---------------------------------------------------------------------------
# Category / channel / login sampling
# ---------------------------------------------------------------------------

def sample_category_counts(tier, drift_progress, growth_factor, has_loan, has_investment, is_salaried, quiet_engagement_decay=0.85):
    counts = {}
    for col, by_tier in CATEGORY_BASE_BY_TIER.items():
        base = by_tier[tier] * growth_factor
        if col == "bills_count":
            lam = base * (1 - drift_progress)  # Pattern 4: bill payment abandonment
        elif col == "transfer_count":
            lam = base * (1 - 0.5 * drift_progress)  # Pattern 3 affects diversity, mildly reduces volume too
        else:
            lam = base * (1 - quiet_engagement_decay * drift_progress)
        counts[col] = int(np.random.poisson(max(0.01, lam)))

    counts["salary_credit_count"] = int(np.random.poisson(0.95 * (1 - 0.3 * drift_progress))) if is_salaried else 0
    counts["loan_repayment_count"] = int(np.random.poisson(0.95 * (1 - drift_progress))) if has_loan else 0
    counts["investment_count"] = int(np.random.poisson(0.55 * (1 - 0.6 * drift_progress))) if has_investment else 0
    return counts


def sample_channel_counts(tier, transaction_count, branch_pct, drift_progress):
    diversity = CHANNEL_DIVERSITY_BY_TIER[tier]
    num_active = max(1, round(diversity * (1 - 0.7 * drift_progress)))
    ranked = sorted(ALL_CHANNELS, key=lambda c: -CHANNEL_BASE_WEIGHTS[c])
    active_channels = ranked[:num_active]
    if "Branch" not in active_channels and branch_pct > 10:
        active_channels.append("Branch")

    remaining = transaction_count
    counts = {col: 0 for col in CHANNEL_TO_COLUMN.values()}
    branch_count = round(transaction_count * branch_pct / 100)
    if "Branch" in active_channels:
        counts["branch_count"] = branch_count
        remaining = max(0, remaining - branch_count)
        active_channels = [c for c in active_channels if c != "Branch"]

    weights = np.array([CHANNEL_BASE_WEIGHTS[c] for c in active_channels], dtype=float)
    if len(weights):
        weights = weights / weights.sum()
        for c, w in zip(active_channels, weights):
            counts[CHANNEL_TO_COLUMN[c]] = int(round(remaining * w))

    return counts


def sample_login_daypart(login_count, age):
    if login_count == 0:
        return 0, 0, 0, 0
    weekday_weights = [0.35, 0.40, 0.25] if age < 45 else [0.45, 0.40, 0.15]
    daypart_counts = np.random.multinomial(login_count, weekday_weights)
    weekend_share = random.uniform(0.15, 0.35)
    weekend_count = round(login_count * weekend_share)
    return int(daypart_counts[0]), int(daypart_counts[1]), int(daypart_counts[2]), weekend_count


def dormant_row(customer_id, month_idx, status, balance, products_held):
    """Fully quiet month: no logins, no transactions, no category/channel
    activity, no lifecycle events, flat balance."""
    row = {
        "customer_id": customer_id, "year_month": MONTHS[month_idx],
        "login_count": 0, "login_morning_count": 0, "login_afternoon_count": 0,
        "login_night_count": 0, "login_weekend_count": 0,
        "days_active": 0, "transaction_count": 0, "failed_transaction_count": 0,
        "inflow_naira": 0.0, "outflow_naira": 0.0, "largest_outflow_naira": 0.0,
        "outflow_first_3days_pct": 0.0, "unique_transfer_recipients": 0,
        "branch_transaction_pct": 0.0, "primary_digital_channel": 0,
        "products_held": max(1, products_held), "month_end_balance_naira": round(balance, 2),
        "account_status": status,
        "transfer_count": 0, "bills_count": 0, "card_pos_count": 0, "cash_withdrawal_count": 0,
        "salary_credit_count": 0, "loan_repayment_count": 0, "investment_count": 0,
        "app_count": 0, "ussd_count": 0, "pos_channel_count": 0, "atm_count": 0,
        "internet_banking_count": 0, "branch_count": 0,
        "complaints_count": 0, "dormancy_warning_sent": False, "password_reset": False,
        "kyc_update": False, "card_issued": False, "loan_approved": False,
        "loan_completed": False, "new_product_opened": False,
    }
    return row, balance


# ---------------------------------------------------------------------------
# Step 2: assemble customer_monthly - single pass per customer
# ---------------------------------------------------------------------------

def generate_customer_monthly(profiles: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for _, cust in profiles.iterrows():
        cid = cust["customer_id"]
        cohort = cust["final_cohort"]
        tier = cust["starting_segment"]
        age = cust["age"]
        account_type = cust["account_type"]
        value_segment = cust["customer_value_segment"]
        is_salaried = value_segment in SALARIED_SEGMENTS
        has_loan = account_type == "Loan" or random.random() < 0.15
        has_investment = account_type == "Investment" or value_segment in ("Affluent", "Private")

        months = active_months_for(cust["tenure_years"])
        n = len(months)
        lo, hi = NAIRA_TIERS[tier]
        base_inflow = random.uniform(lo, hi)
        branch_bias = 15.0 if age >= 60 else 3.0
        base_login = {"Heavy": 22, "Regular": 12, "Occasional": 5, "Business": 18}[tier]
        base_recipients = random.randint(5, 9)
        products_held = initial_products_held(tier, account_type, cohort)
        balance = base_inflow * random.uniform(0.5, 2.0)

        decline_start, churn_point = churn_points_for(cohort, n)

        # --- one-off lifecycle event months, chosen once per customer ---
        card_issued_pos = None
        if random.random() < 0.85:
            card_issued_pos = min(n - 1, random.randint(0, min(2, n - 1)))
        loan_approved_pos, loan_completed_pos = None, None
        if has_loan:
            loan_approved_pos = random.randint(0, max(0, (n // 2) - 1)) if n > 1 else 0
            if random.random() < 0.25 and n > 12:
                loan_completed_pos = min(n - 1, random.randint(n // 2, n - 1))
        password_reset_pos = random.randint(0, n - 1) if random.random() < 0.10 else None
        kyc_update_pos = random.randint(0, n - 1) if random.random() < 0.15 else None

        dormancy_warning_fired = False

        for pos, m_idx in enumerate(months):
            growth_factor = 1.0
            new_product_this_month = False
            if cohort == "Growing":
                growth_factor = 1 + 0.015 * (pos / max(1, n - 1))
                if pos > 0 and pos % 6 == 0:
                    old = products_held
                    products_held = min(5, products_held + 1)
                    new_product_this_month = products_held > old

            is_quiet_month = churn_point is not None and pos > churn_point

            if is_quiet_month:
                months_since_last_txn = pos - churn_point
                status = bank_status(months_since_last_txn)
                row, balance = dormant_row(cid, m_idx, status, balance, products_held)
                if status in ("Inactive", "Dormant") and not dormancy_warning_fired:
                    row["dormancy_warning_sent"] = True
                    dormancy_warning_fired = True
                all_rows.append(row)
                continue

            if decline_start is not None and pos >= decline_start:
                span = max(1, (churn_point if churn_point is not None else n - 1) - decline_start)
                drift_progress = min(0.97, (pos - decline_start) / span)
            else:
                drift_progress = 0.0

            status = bank_status(0)

            cats = sample_category_counts(tier, drift_progress, growth_factor, has_loan, has_investment, is_salaried)
            transaction_count = sum(cats.values())

            inflow = base_inflow * growth_factor * random.uniform(0.85, 1.15)
            login_count = round(max(0, base_login * growth_factor * (1 - 0.85 * drift_progress) * random.uniform(0.85, 1.15)))
            days_active = min(30, round(login_count * random.uniform(0.5, 0.9)))
            failed_transaction_count = int(np.random.poisson(0.3 + 0.4 * drift_progress))

            save_rate = random.uniform(0.05, 0.4)
            outflow = inflow * (1 - save_rate)

            base_first3_pct = random.uniform(15, 30)
            outflow_first_3days_pct = min(97, base_first3_pct + 65 * drift_progress)

            if drift_progress > 0.6 and random.random() < drift_progress:
                largest_outflow = outflow * random.uniform(0.75, 0.97)
            else:
                largest_outflow = outflow * random.uniform(0.15, 0.4)

            unique_recipients = max(1, round(base_recipients * (1 - 0.75 * drift_progress)))
            branch_pct = min(100, branch_bias + 20 * drift_progress + random.uniform(0, 5))

            # complaints: a count, not a flag - churn often happens silently
            # (drift_progress low -> baseline rare complaints), but complaints
            # do rise with decline for the subset who do complain.
            complaints_lambda = 0.05 + (0.35 * drift_progress if drift_progress > 0.5 else 0)
            complaints_count = int(np.random.poisson(complaints_lambda))

            lm, la, ln, lw = sample_login_daypart(login_count, age)
            channel_counts = sample_channel_counts(tier, transaction_count, branch_pct, drift_progress)

            if transaction_count == 0 and login_count == 0:
                primary_channel_code = 0
            else:
                dominant_channel = max(
                    ((c, channel_counts[CHANNEL_TO_COLUMN[c]]) for c in DIGITAL_CHANNEL_CODES),
                    key=lambda x: x[1], default=("App", 0),
                )[0]
                primary_channel_code = DIGITAL_CHANNEL_CODES[dominant_channel]

            balance = max(0, balance + inflow - outflow + random.uniform(-0.02, 0.02) * balance)
            products_held = max(1, products_held)

            row = {
                "customer_id": cid, "year_month": MONTHS[m_idx],
                "login_count": login_count, "login_morning_count": lm, "login_afternoon_count": la,
                "login_night_count": ln, "login_weekend_count": lw,
                "days_active": days_active, "transaction_count": transaction_count,
                "failed_transaction_count": failed_transaction_count,
                "inflow_naira": round(inflow, 2), "outflow_naira": round(outflow, 2),
                "largest_outflow_naira": round(largest_outflow, 2),
                "outflow_first_3days_pct": round(outflow_first_3days_pct, 2),
                "unique_transfer_recipients": unique_recipients,
                "branch_transaction_pct": round(branch_pct, 2),
                "primary_digital_channel": primary_channel_code,
                "products_held": products_held, "month_end_balance_naira": round(balance, 2),
                "account_status": status,
                "transfer_count": cats["transfer_count"], "bills_count": cats["bills_count"],
                "card_pos_count": cats["card_pos_count"], "cash_withdrawal_count": cats["cash_withdrawal_count"],
                "salary_credit_count": cats["salary_credit_count"], "loan_repayment_count": cats["loan_repayment_count"],
                "investment_count": cats["investment_count"],
                **channel_counts,
                "complaints_count": complaints_count,
                "dormancy_warning_sent": False,
                "password_reset": password_reset_pos == pos,
                "kyc_update": kyc_update_pos == pos,
                "card_issued": card_issued_pos == pos,
                "loan_approved": loan_approved_pos == pos,
                "loan_completed": loan_completed_pos == pos,
                "new_product_opened": new_product_this_month,
            }
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {NUM_CUSTOMERS:,} customer profiles...")
    profiles = generate_customer_profiles()
    profiles.to_csv(os.path.join(OUT_DIR, "customer_profiles.csv"), index=False)
    print(profiles["final_cohort"].value_counts(normalize=True).round(3))
    print(profiles["customer_value_segment"].value_counts(normalize=True).round(3))

    print(f"\nGenerating customer_monthly across {N_MONTHS} months ({MONTHS[0]} - {MONTHS[-1]})...")
    monthly = generate_customer_monthly(profiles)
    monthly.to_csv(os.path.join(OUT_DIR, "customer_monthly.csv"), index=False)
    print(f"  -> customer_monthly.csv ({len(monthly):,} rows, {monthly.shape[1]} columns)")

    current_status = monthly.sort_values("year_month").groupby("customer_id").tail(1)["account_status"]
    print("\naccount_status, final month only (should match cohort split):")
    print(current_status.value_counts(normalize=True).round(3))

    print("\nLifecycle flag counts:")
    for col in ["dormancy_warning_sent", "password_reset", "kyc_update", "card_issued",
                "loan_approved", "loan_completed", "new_product_opened"]:
        print(f"  {col}: {int(monthly[col].sum()):,}")
    print(f"  complaints_count total: {int(monthly['complaints_count'].sum()):,}")


if __name__ == "__main__":
    main()
