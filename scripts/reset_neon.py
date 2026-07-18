"""
Drops EVERY table in the Neon database - not just the 9 known ORM tables.

src/database.py's init_db(force=True) only knows about tables defined in
its SQLAlchemy models. It has no idea features_long, features_snapshot, or
the stg_* staging tables exist (those are created by raw SQL in
preprocessing.py), so repeated preprocessing runs across iterations have
been silently accumulating storage that --force never touched. This
discovers every table in the public schema and drops all of them, giving
an actually clean slate.

Run:
    python scripts/reset_neon.py
    python scripts/reset_neon.py --yes    (skip the confirmation prompt)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import inspect, text  # noqa: E402
from src.database import engine  # noqa: E402


def main():
    skip_confirm = "--yes" in sys.argv

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if not tables:
        print("No tables found - database is already empty.")
        return

    print(f"Found {len(tables)} tables in the database:")
    for t in tables:
        print(f"  - {t}")

    if not skip_confirm:
        confirm = input(f"\nDrop all {len(tables)} tables? This cannot be undone. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return

    with engine.begin() as conn:
        for t in tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
            print(f"  Dropped {t}")

    print(f"\nAll {len(tables)} tables dropped. Database is clean.")
    print("Next: python src/database.py  (rebuilds the 9 core tables)")


if __name__ == "__main__":
    main()
