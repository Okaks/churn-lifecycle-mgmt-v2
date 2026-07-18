"""
Verifies the data loaded into Neon actually matches what it's supposed to:
row counts, referential integrity, cohort-vs-status consistency, and a real
sample trajectory per cohort (replaces the old 3-named-customer preview,
since customer_id is now randomized rather than named).

Run:
    python scripts/verify_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text  # noqa: E402
from src.database import engine  # noqa: E402


def section(title):
    print(f"\n{'-' * 60}\n{title}\n{'-' * 60}")


def row_counts(conn):
    section("Row counts")
    for t in ["customer_profiles", "monthly_behaviour", "transaction_categories", "channel_usage", "lifecycle_events"]:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"  {t:<24} {n:,}")


def orphan_check(conn):
    section("Referential integrity (orphan rows - should all be 0)")
    for t in ["monthly_behaviour", "transaction_categories", "channel_usage", "lifecycle_events"]:
        n = conn.execute(text(f"""
            SELECT COUNT(*) FROM {t} c
            LEFT JOIN customer_profiles p ON c.customer_id = p.customer_id
            WHERE p.customer_id IS NULL
        """)).scalar()
        flag = "OK" if n == 0 else "PROBLEM"
        print(f"  {t:<24} {n} orphans  [{flag}]")


def cohort_distribution(conn):
    section("Cohort distribution")
    rows = conn.execute(text("""
        SELECT final_cohort, COUNT(*), ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM customer_profiles GROUP BY final_cohort ORDER BY pct DESC
    """)).fetchall()
    for cohort, n, pct in rows:
        print(f"  {cohort:<22} {n:>6,}  ({pct}%)")


def status_matches_cohort(conn):
    section("Final-month account_status vs cohort (should be a clean 1:1 mapping)")
    rows = conn.execute(text("""
        WITH last_month AS (
            SELECT DISTINCT ON (customer_id) customer_id, account_status
            FROM monthly_behaviour ORDER BY customer_id, year_month DESC
        )
        SELECT p.final_cohort, lm.account_status, COUNT(*)
        FROM last_month lm JOIN customer_profiles p ON p.customer_id = lm.customer_id
        GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """)).fetchall()
    for cohort, status, n in rows:
        print(f"  {cohort:<22} -> {status:<10} {n:>6,}")


def sample_trajectory(conn, cohort):
    section(f"Sample trajectory: one {cohort} customer")
    cust = conn.execute(text("""
        SELECT customer_id FROM customer_profiles WHERE final_cohort = :cohort
        ORDER BY customer_id LIMIT 1 OFFSET 3
    """), {"cohort": cohort}).scalar()
    if cust is None:
        print("  (no customer found in this cohort)")
        return
    print(f"  customer_id = {cust}")
    rows = conn.execute(text("""
        SELECT year_month, transaction_count, bill_payment_count,
               unique_transfer_recipients, account_status
        FROM monthly_behaviour WHERE customer_id = :cid ORDER BY year_month
    """), {"cid": cust}).fetchall()
    print(f"  {'month':<12}{'txn_count':<12}{'bills':<8}{'recipients':<12}{'status'}")
    for ym, txn, bills, recip, status in rows:
        print(f"  {str(ym):<12}{txn:<12}{bills:<8}{recip:<12}{status}")


def lifecycle_event_summary(conn):
    section("Lifecycle events by type")
    rows = conn.execute(text("""
        SELECT event_type, COUNT(*) FROM lifecycle_events
        GROUP BY event_type ORDER BY COUNT(*) DESC
    """)).fetchall()
    for event_type, n in rows:
        print(f"  {event_type:<24} {n:,}")


def main():
    with engine.connect() as conn:
        row_counts(conn)
        orphan_check(conn)
        cohort_distribution(conn)
        status_matches_cohort(conn)
        lifecycle_event_summary(conn)
        for cohort in ["Healthy", "Early Drift", "Dormant-or-Churned", "Inactive", "Growing", "New-Unconverted"]:
            sample_trajectory(conn, cohort)

    print("\nDone. Review the sample trajectories above -")
    print("Healthy: flat and stable. Early Drift: bills/recipients decline mid-to-late window, status stays Active.")
    print("Dormant-or-Churned: transaction_count/bills drop to 0, status Active -> Inactive -> Dormant.")
    print("Growing: gentle upward trend. New-Unconverted: short history, sparse activity throughout.")


if __name__ == "__main__":
    main()
