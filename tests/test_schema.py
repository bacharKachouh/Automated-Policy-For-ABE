import sqlite3
import tempfile
from pathlib import Path

from src.db.connection import init_db


EXPECTED_TABLES = {
    "authorities",
    "global_params",
    "patients",
    "patient_records",
    "ciphertexts",
    "audit_runs",
}


def test_init_db_creates_all_tables(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    conn = sqlite3.connect(str(db_file))
    try:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()
    assert EXPECTED_TABLES.issubset(names)


def test_init_db_is_idempotent(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    init_db(db_file)  # must not raise
