#!/usr/bin/env python3
"""Create the SQLite database and apply the schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH
from src.db.connection import init_db


def main():
    init_db()
    print(f"Database ready → {DB_PATH}")


if __name__ == "__main__":
    main()
