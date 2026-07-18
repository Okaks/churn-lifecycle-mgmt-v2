"""
Loads the 2 generated CSVs (data/raw/*.csv) into their matching Neon tables
(schema v3 - customer_profiles + customer_monthly).

Uses PostgreSQL COPY (not row-by-row INSERT) for speed.

Run:
    python scripts/load_data.py --force
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database import engine, init_db  # noqa: E402

DATA_DIR = os.path.join("data", "raw")

# (csv filename, table name, column order matching the CSV header exactly,
#  excluding the autoincrement id column on customer_monthly)
TABLES = [
    (
        "customer_profiles.csv", "customer_profiles",
        ["customer_id", "surname", "first_name", "age", "gender", "location",
         "tenure_years", "account_type", "starting_segment", "customer_value_segment",
         "avg_monthly_salary_naira", "final_cohort"],
    ),
    (
        "customer_monthly.csv", "customer_monthly",
        ["customer_id", "year_month", "login_count", "login_morning_count",
         "login_afternoon_count", "login_night_count", "login_weekend_count",
         "days_active", "transaction_count", "failed_transaction_count",
         "inflow_naira", "outflow_naira", "largest_outflow_naira",
         "outflow_first_3days_pct", "unique_transfer_recipients",
         "branch_transaction_pct", "primary_digital_channel", "products_held",
         "month_end_balance_naira", "account_status",
         "transfer_count", "bills_count", "card_pos_count", "cash_withdrawal_count",
         "salary_credit_count", "loan_repayment_count", "investment_count",
         "app_count", "ussd_count", "pos_channel_count", "atm_count",
         "internet_banking_count", "branch_count",
         "complaints_count", "dormancy_warning_sent", "password_reset", "kyc_update",
         "card_issued", "loan_approved", "loan_completed", "new_product_opened"],
    ),
]


def load_table_via_copy(csv_path, table_name, columns):
    raw_conn = engine.raw_connection()
    try:
        cur = raw_conn.cursor()
        column_list = ", ".join(columns)
        copy_sql = f"COPY {table_name} ({column_list}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"

        if hasattr(cur, "copy_expert"):
            # psycopg2
            with open(csv_path, "r", encoding="utf-8") as f:
                cur.copy_expert(copy_sql, f)
        else:
            # psycopg v3
            with cur.copy(copy_sql) as copy:
                with open(csv_path, "r", encoding="utf-8") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        copy.write(chunk)

        raw_conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        return count
    finally:
        raw_conn.close()


def main():
    force = "--force" in sys.argv

    print("Initializing schema...")
    init_db(force=force)

    if not force:
        with engine.connect() as conn:
            existing = conn.exec_driver_sql("SELECT COUNT(*) FROM customer_profiles").scalar()
        if existing > 0:
            print(
                f"customer_profiles already has {existing:,} rows. "
                "Run with --force for a clean reset, or truncate manually first."
            )
            sys.exit(1)

    for filename, table_name, columns in TABLES:
        csv_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(csv_path):
            print(f"Missing {csv_path} - run scripts/generate_synthetic_data.py first.")
            sys.exit(1)

        print(f"Loading {filename} -> {table_name} ...")
        t0 = time.time()
        count = load_table_via_copy(csv_path, table_name, columns)
        print(f"  -> {count:,} rows in {table_name} ({time.time() - t0:.1f}s)")

    print("\nBoth tables loaded.")


if __name__ == "__main__":
    main()
