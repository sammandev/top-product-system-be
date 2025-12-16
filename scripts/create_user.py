"""Create a local user in the application's PostgreSQL database.

Usage (from backend_fastapi root):
python scripts/create_user.py --username admin --password secret --admin

This script ensures the project's `src` directory is on sys.path so the
local `app` package can be imported. It calls `init_db()` to prepare the DB
and then creates/updates a user with `create_user`.
"""

import argparse
import os
import sys
from getpass import getpass

# Ensure src is on path (backend_fastapi/src)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app.db import SessionLocal
from app.db.init_db import init_db
from app.utils.auth import create_user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=False)
    parser.add_argument("--password", required=False)
    parser.add_argument("--admin", action="store_true")
    args = parser.parse_args()

    username = args.username or input("username: ")
    password = args.password or getpass("password: ")

    # initialize DB (creates tables if needed)
    init_db()

    db = SessionLocal()
    try:
        user = create_user(db, username, password, is_admin=args.admin)
        print(f"Created/updated user: {user.username} (admin={user.is_admin})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
