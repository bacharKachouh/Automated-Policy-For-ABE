"""
Domain-level persistence functions for Automated-ABE.

This module is the only place SQL strings live. Everything else calls these
functions. Charm group elements are converted to/from bytes via
src.abe.serialization before they cross the DB boundary.
"""
import json

from charm.toolbox.pairinggroup import G1

from src.abe.serialization import (
    deserialize_element,
    deserialize_obj,
    serialize_element,
    serialize_obj,
)
from src.db.connection import get_conn


# ---------------------------------------------------------------------------
# Global params
# ---------------------------------------------------------------------------

def save_global_params(group, gp, pairing_group_name):
    """Persist the singleton row containing the generator g."""
    g_bytes = serialize_element(group, gp["g"])
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO global_params (id, pairing_group, g_bytes) "
            "VALUES (1, ?, ?)",
            (pairing_group_name, g_bytes),
        )
        conn.commit()


def load_global_params(group):
    """
    Load global params and reconstruct H.

    Returns a dict {"g": <Element>, "H": <callable>} matching Dabe.setup output.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT g_bytes FROM global_params WHERE id = 1"
        ).fetchone()
    if row is None:
        raise RuntimeError(
            "Global params missing — run scripts/generate_authority_keys.py"
        )
    g = deserialize_element(group, row["g_bytes"])
    H = lambda x: group.hash(x, G1)  # noqa: E731
    return {"g": g, "H": H}


# ---------------------------------------------------------------------------
# Authorities
# ---------------------------------------------------------------------------

def save_authority(group, name, type_, prefix, sk, pk):
    sk_bytes = serialize_obj(group, sk)
    pk_bytes = serialize_obj(group, pk)
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO authorities (name, type, prefix, sk_bytes, pk_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, type_, prefix, sk_bytes, pk_bytes),
        )
        conn.commit()


def load_authority_sk(group, name):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT sk_bytes FROM authorities WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Authority not found: {name}")
    return deserialize_obj(group, row["sk_bytes"])


def load_authority_pk(group, name):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT pk_bytes FROM authorities WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Authority not found: {name}")
    return deserialize_obj(group, row["pk_bytes"])


def load_all_pks(group):
    """
    Merge every authority's PK dict into a single dict, the format
    HybridABEncMA.encrypt expects.
    """
    with get_conn() as conn:
        rows = conn.execute("SELECT name, pk_bytes FROM authorities").fetchall()
    merged = {}
    for row in rows:
        pk = deserialize_obj(group, row["pk_bytes"])
        merged.update(pk)
    return merged


def list_authorities():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name, type, prefix FROM authorities ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def register_patient(hospital_name, patient_id, gid):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO patients (hospital_name, patient_id, gid) "
            "VALUES (?, ?, ?)",
            (hospital_name, patient_id, gid),
        )
        conn.commit()


def get_patient(hospital_name, patient_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT patient_id, hospital_name, gid FROM patients "
            "WHERE hospital_name = ? AND patient_id = ?",
            (hospital_name, patient_id),
        ).fetchone()
    return dict(row) if row else None


def list_patients(hospital_name):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT patient_id FROM patients WHERE hospital_name = ? ORDER BY patient_id",
            (hospital_name,),
        ).fetchall()
    return [r["patient_id"] for r in rows]
