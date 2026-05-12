"""SQLite connection helpers."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import DB_PATH

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def _apply_schema(conn):
    conn.executescript(SCHEMA_FILE.read_text())


@contextmanager
def get_conn(db_path=None):
    """Yield a sqlite3 connection. Applies schema if the DB file is new."""
    path = Path(db_path or DB_PATH)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if is_new:
            _apply_schema(conn)
            conn.commit()
        yield conn
    finally:
        conn.close()


def init_db(db_path=None):
    """Force-apply schema (idempotent thanks to IF NOT EXISTS)."""
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        _apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
