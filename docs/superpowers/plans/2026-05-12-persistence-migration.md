# Persistence Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate ABE authority keys, patient mapping, ciphertexts, and audit runs from filesystem-scattered state into a single SQLite database; fix the multi-authority keygen bug; verify end-to-end via smoke test.

**Architecture:** Thin `src/db/` module wraps `sqlite3` with parameterised queries (no ORM). `src/abe/serialization.py` round-trips Charm group elements using `objectToBytes`/`bytesToObject`. Pipeline loads persisted authority keys and routes user keygen by attribute prefix so cross-authority users actually work.

**Tech Stack:** Python 3.7, Charm-Crypto 0.50 (SS512), sqlite3 (stdlib), pytest, BioClinicalBERT and GPT-2 (existing, unchanged).

**Spec:** `docs/superpowers/specs/2026-05-12-persistence-migration-design.md`

**Test note:** Charm-Crypto is hard to install on Windows; pure unit tests for Charm-dependent code are out of scope. Pure-logic helpers (e.g. `resolve_authority`, schema application, the classifier empty-input guard) get focused pytest coverage. End-to-end correctness comes from the smoke test in Task 10, which uses injectable ML stages so it doesn't depend on trained models.

---

## File Structure

**Create:**
- `src/db/__init__.py`
- `src/db/schema.sql` — all `CREATE TABLE` statements
- `src/db/connection.py` — `get_conn()` context manager, applies schema if DB is new
- `src/db/repo.py` — typed-ish domain functions: authorities, global params, patients, records, ciphertexts, audit
- `src/abe/serialization.py` — Charm element / SK / PK / ciphertext serialise helpers
- `scripts/init_db.py` — creates the SQLite file, applies schema
- `scripts/smoke_test.py` — end-to-end workflow check with injected ML stubs
- `tests/__init__.py` — empty package marker
- `tests/test_resolve_authority.py` — pure-logic test
- `tests/test_classifier_guard.py` — pure-logic test
- `tests/test_schema.py` — checks schema applies cleanly and all tables exist

**Modify:**
- `src/config.py` — add `DB_PATH`, env-overridable
- `src/abe/authority.py` — drop the str-based JSON helpers; replace with Charm-native serialisation calls (or delete the file entirely if no longer used)
- `src/abe/hybrid.py:54-56` — remove dead `if key is False` block
- `src/classification/classifier.py:81-87` — guard empty `section_labels`
- `src/hospital/patient.py` — `register_patient` writes to DB, `list_patients` reads from DB; keep filesystem dir-creation
- `src/hospital/pipeline.py` — load global params + PKs from DB; new `resolve_authority(attr)` helper; per-authority keygen; persist `patient_records`, `ciphertexts`, `audit_runs`; accept optional injected `classifier` and `extractor` for testability
- `scripts/generate_authority_keys.py` — persist authorities to DB via repo
- `scripts/register_patient.py` — pass through to the new DB-backed `register_patient`
- `scripts/run_pipeline.py` — unchanged surface; will work because internals are migrated
- `.gitignore` — add `data/abe.db`, `data/abe.db-journal`, `data/abe.db-wal`

**Retire (delete from working tree):**
- `keys/public_keys_and_GP.json`
- `data/Hospital1/keys/hospital_A_SK.json`
- `data/Hospital2/keys/hospital_B_SK.json`
- `data/Hospital1/patient_mapping.csv` and `data/Hospital2/patient_mapping.csv`

---

## Task 1: DB scaffolding — schema, connection, init script

**Files:**
- Create: `src/db/__init__.py`, `src/db/schema.sql`, `src/db/connection.py`, `scripts/init_db.py`, `tests/__init__.py`, `tests/test_schema.py`
- Modify: `src/config.py`, `.gitignore`

- [ ] **Step 1: Add `DB_PATH` to config**

Edit `src/config.py`, append after the `TRAINING_DATA_DIR` line (around line 18):

```python
DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_DIR / "abe.db")))
```

- [ ] **Step 2: Add DB files to .gitignore**

Append to `.gitignore`:

```
# SQLite state
data/abe.db
data/abe.db-journal
data/abe.db-wal
```

- [ ] **Step 3: Create `src/db/__init__.py`**

```python
"""SQLite-backed persistence for Automated-ABE."""
```

- [ ] **Step 4: Create `src/db/schema.sql`**

```sql
CREATE TABLE IF NOT EXISTS authorities (
    name        TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK(type IN ('hospital','insurance')),
    prefix      TEXT NOT NULL UNIQUE,
    sk_bytes    BLOB NOT NULL,
    pk_bytes    BLOB NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS global_params (
    id              INTEGER PRIMARY KEY CHECK(id = 1),
    pairing_group   TEXT NOT NULL,
    g_bytes         BLOB NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT NOT NULL,
    hospital_name   TEXT NOT NULL REFERENCES authorities(name),
    gid             TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hospital_name, patient_id)
);

CREATE TABLE IF NOT EXISTS patient_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_name        TEXT NOT NULL,
    patient_id           TEXT NOT NULL,
    xml_filename         TEXT NOT NULL,
    document_label       TEXT,
    section_labels_json  TEXT,
    policy_text          TEXT,
    abe_policy_string    TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hospital_name, patient_id) REFERENCES patients(hospital_name, patient_id)
);

CREATE TABLE IF NOT EXISTS ciphertexts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id           INTEGER NOT NULL REFERENCES patient_records(id),
    ciphertext_bytes    BLOB NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id            INTEGER REFERENCES patient_records(id),
    user_gid             TEXT,
    user_attributes_json TEXT,
    outcome              TEXT NOT NULL,
    duration_ms          INTEGER,
    error_message        TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 5: Create `src/db/connection.py`**

```python
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
```

- [ ] **Step 6: Create `scripts/init_db.py`**

```python
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
```

- [ ] **Step 7: Create `tests/__init__.py`**

```python
```

- [ ] **Step 8: Write the failing schema test**

Create `tests/test_schema.py`:

```python
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
```

- [ ] **Step 9: Run the test**

```bash
pytest tests/test_schema.py -v
```

Expected: both tests PASS.

- [ ] **Step 10: Manually init the project DB**

```bash
python scripts/init_db.py
```

Expected output: `Database ready → /…/data/abe.db`. Verify file exists.

- [ ] **Step 11: Commit**

```bash
git add src/config.py src/db/ scripts/init_db.py tests/__init__.py tests/test_schema.py .gitignore
git commit -m "Add SQLite scaffolding: schema, connection helper, init script"
```

---

## Task 2: Charm serialisation helpers

**Files:**
- Create: `src/abe/serialization.py`

- [ ] **Step 1: Create `src/abe/serialization.py`**

```python
"""
Charm-Crypto serialisation helpers.

Round-trips group elements / nested dicts of group elements through bytes
using Charm's native objectToBytes / bytesToObject. Replaces the broken
str(...) JSON dumps that previously lived in src/abe/authority.py.
"""
from charm.core.engine.util import objectToBytes, bytesToObject


def serialize_element(group, element):
    """Serialise a single Charm group element to bytes."""
    return group.serialize(element)


def deserialize_element(group, blob):
    """Deserialise a single Charm group element from bytes."""
    return group.deserialize(blob)


def serialize_obj(group, obj):
    """Serialise an arbitrary Charm-aware object (dict of elements, ciphertext, etc.)."""
    return objectToBytes(obj, group)


def deserialize_obj(group, blob):
    """Inverse of serialize_obj."""
    return bytesToObject(blob, group)
```

- [ ] **Step 2: Commit**

```bash
git add src/abe/serialization.py
git commit -m "Add Charm-native serialisation helpers"
```

---

## Task 3: Authority + global-params repo

**Files:**
- Create: `src/db/repo.py`

- [ ] **Step 1: Create `src/db/repo.py` with authority + global-params functions**

```python
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
```

- [ ] **Step 2: Quick smoke check at the Python REPL (manual)**

```bash
python -c "from src.db.repo import list_authorities; print(list_authorities())"
```

Expected: `[]` (empty list — no authorities yet).

- [ ] **Step 3: Commit**

```bash
git add src/db/repo.py
git commit -m "Add repo functions for authorities and global params"
```

---

## Task 4: Persist authority keys (rewrite `generate_authority_keys.py`)

**Files:**
- Modify: `scripts/generate_authority_keys.py` (full rewrite)
- Modify: `src/abe/authority.py` (delete `save_secret_key`, `save_public_keys_and_gp`, `load_json` — they're obsolete; keep only the `AuthorityGeneration` class if anything still imports it)

- [ ] **Step 1: Check what imports `authority.py`**

```bash
grep -rn "from src.abe.authority" src scripts
```

Note any remaining imports.

- [ ] **Step 2: Trim `src/abe/authority.py`**

Replace the full file content with:

```python
"""
Authority key generation (decoupled from persistence).

Persistence of the resulting SK/PK is handled by src.db.repo.
"""
from charm.toolbox.pairinggroup import G1, pair


class AuthorityGeneration:
    """Generates global parameters and per-authority key pairs."""

    def __init__(self, groupObj):
        self._group = groupObj

    def setup(self):
        g = self._group.random(G1)
        H = lambda x: self._group.hash(x, G1)  # noqa: E731
        return {"g": g, "H": H}

    def authsetup(self, GP, attributes):
        SK, PK = {}, {}
        for attr in attributes:
            alpha_i, y_i = self._group.random(), self._group.random()
            SK[attr] = {"alpha_i": alpha_i, "y_i": y_i}
            PK[attr] = {
                "e(gg)^alpha_i": pair(GP["g"], GP["g"]) ** alpha_i,
                "g^y_i": GP["g"] ** y_i,
            }
        return SK, PK
```

- [ ] **Step 3: Rewrite `scripts/generate_authority_keys.py`**

```python
#!/usr/bin/env python3
"""
Generate the global pairing parameters and per-authority key pairs and
persist them to the SQLite database. Idempotent: re-running replaces
existing rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from charm.toolbox.pairinggroup import PairingGroup

from src import config
from src.abe.dabe import Dabe
from src.db import repo
from src.db.connection import init_db


def main():
    init_db()  # idempotent

    print(f"Initialising pairing group ({config.PAIRING_GROUP})...")
    group = PairingGroup(config.PAIRING_GROUP)
    dabe = Dabe(group)
    gp = dabe.setup()

    repo.save_global_params(group, gp, config.PAIRING_GROUP)
    print("Global params saved.")

    for name, cfg in config.HOSPITAL_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "hospital", cfg["prefix"], sk, pk)
        print(f"  Authority saved: {name} ({len(cfg['attributes'])} attributes)")

    for name, cfg in config.INSURANCE_CONFIGS.items():
        # Insurance configs don't currently carry a prefix in config.py — derive it.
        prefix = _prefix_from_insurance_attrs(cfg["attributes"])
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "insurance", prefix, sk, pk)
        print(f"  Authority saved: {name} ({len(cfg['attributes'])} attributes)")


def _prefix_from_insurance_attrs(attrs):
    """All insurance attrs share a prefix like 'insCoA.xxx'. Extract it."""
    return attrs[0].split(".", 1)[0]


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `prefix` field to `INSURANCE_CONFIGS` in `src/config.py`** (cleaner than deriving it)

Edit `src/config.py:63-80`. Add `"prefix": "insCoA"` / `"prefix": "insCoB"`:

```python
INSURANCE_CONFIGS = {
    "InsCoA": {
        "prefix": "insCoA",
        "attributes": [
            "insCoA.underwriter", "insCoA.claims_adjuster", "insCoA.customer_service",
            "insCoA.policy_admin", "insCoA.claims_processing",
            "insCoA.policy_expertise_health", "insCoA.policy_expertise_life",
            "insCoA.clearance1", "insCoA.clearance2", "insCoA.clearance3",
        ],
    },
    "InsCoB": {
        "prefix": "insCoB",
        "attributes": [
            "insCoB.underwriter", "insCoB.claims_adjuster", "insCoB.customer_service",
            "insCoB.policy_admin", "insCoB.claims_processing",
            "insCoB.policy_expertise_auto", "insCoB.policy_expertise_property",
            "insCoB.clearance1", "insCoB.clearance2", "insCoB.clearance3",
        ],
    },
}
```

Then simplify the script — drop `_prefix_from_insurance_attrs` and use `cfg["prefix"]`:

```python
    for name, cfg in config.INSURANCE_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "insurance", cfg["prefix"], sk, pk)
        print(f"  Authority saved: {name} ({len(cfg['attributes'])} attributes)")
```

- [ ] **Step 5: Run the script**

```bash
python scripts/generate_authority_keys.py
```

Expected: prints 4 "Authority saved" lines (Hospital1, Hospital2, InsCoA, InsCoB).

- [ ] **Step 6: Verify via sqlite3**

```bash
sqlite3 data/abe.db "SELECT name, type, prefix, length(sk_bytes), length(pk_bytes) FROM authorities;"
```

Expected: 4 rows, all with non-zero `length(sk_bytes)` and `length(pk_bytes)`.

- [ ] **Step 7: Round-trip check via REPL**

```bash
python -c "
from charm.toolbox.pairinggroup import PairingGroup
from src import config
from src.db import repo
g = PairingGroup(config.PAIRING_GROUP)
sk = repo.load_authority_sk(g, 'Hospital1')
print('SK keys:', list(sk)[:3])
pk = repo.load_all_pks(g)
print('Total PK attrs:', len(pk))
"
```

Expected: prints three attribute keys and a total >= 42 (hospitals 11+11 + insurance 10+10).

- [ ] **Step 8: Delete the obsolete JSON key files**

```bash
git rm keys/public_keys_and_GP.json data/Hospital1/keys/hospital_A_SK.json data/Hospital2/keys/hospital_B_SK.json
```

- [ ] **Step 9: Commit**

```bash
git add src/config.py src/abe/authority.py scripts/generate_authority_keys.py
git commit -m "Persist authorities and global params to SQLite"
```

---

## Task 5: `resolve_authority` helper + pipeline keygen fix

**Files:**
- Modify: `src/hospital/pipeline.py`
- Create: `tests/test_resolve_authority.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolve_authority.py`:

```python
import pytest

from src.hospital.pipeline import resolve_authority


def test_resolves_hospital_a():
    assert resolve_authority("hospitalA.doctor") == "Hospital1"


def test_resolves_hospital_b():
    assert resolve_authority("hospitalB.nurse") == "Hospital2"


def test_resolves_insurance_a():
    assert resolve_authority("insCoA.underwriter") == "InsCoA"


def test_resolves_insurance_b():
    assert resolve_authority("insCoB.claims_adjuster") == "InsCoB"


def test_unknown_prefix_raises():
    with pytest.raises(KeyError):
        resolve_authority("unknownAuthority.doctor")


def test_malformed_attribute_raises():
    with pytest.raises(ValueError):
        resolve_authority("no_dot_here")
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_resolve_authority.py -v
```

Expected: ImportError or AttributeError (function doesn't exist yet).

- [ ] **Step 3: Add `resolve_authority` to `src/hospital/pipeline.py`**

Add near the top of the file, after the imports:

```python
def _build_prefix_map():
    """Map attribute prefix -> authority name. Derived once from config."""
    m = {}
    for name, cfg in config.HOSPITAL_CONFIGS.items():
        m[cfg["prefix"]] = name
    for name, cfg in config.INSURANCE_CONFIGS.items():
        m[cfg["prefix"]] = name
    return m


_PREFIX_MAP = _build_prefix_map()


def resolve_authority(attribute):
    """
    Map a fully-qualified attribute (e.g. "hospitalA.doctor") to the name of
    the authority that owns it. Raises ValueError on a malformed attribute
    and KeyError on an unknown prefix.
    """
    if "." not in attribute:
        raise ValueError(f"Malformed attribute (no prefix): {attribute!r}")
    prefix = attribute.split(".", 1)[0]
    if prefix not in _PREFIX_MAP:
        raise KeyError(f"Unknown attribute prefix: {prefix!r}")
    return _PREFIX_MAP[prefix]
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/test_resolve_authority.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Rewrite `_build_abe_system` and `run_pipeline` keygen block**

Replace the existing `_build_abe_system` in `src/hospital/pipeline.py` with:

```python
def _build_abe_system():
    """
    Construct the Hybrid ABE wrapper, load persisted global params and
    public keys from the DB. Returns (hyb, group, gp, all_pk).
    """
    from src.db import repo

    group = PairingGroup(config.PAIRING_GROUP)
    hyb = HybridABEncMA(Dabe(group), group)
    gp = repo.load_global_params(group)
    all_pk = repo.load_all_pks(group)
    return hyb, group, gp, all_pk
```

Then update the imports at the top of the file — add:

```python
from src.db import repo
```

(adjacent to the other imports).

Then update `run_pipeline`'s Stage 4 section to use the new tuple shape and per-authority keygen. Locate the block in `src/hospital/pipeline.py` that starts at the `# Stage 4` heading (originally around line 138-144) and replace it with:

```python
    # -----------------------------------------------------------------------
    # Stage 4 — ABE Encryption + Decryption
    # -----------------------------------------------------------------------
    print("\n[4/4] Running ABE encryption...")
    hyb, group, gp, all_pk = _build_abe_system()

    # Issue keys per owning authority (the multi-authority fix)
    user_keys = {}
    for attr in user_attributes:
        authority_name = resolve_authority(attr)
        authority_sk = repo.load_authority_sk(group, authority_name)
        hyb.keygen(gp, authority_sk, attr, user_gid, user_keys)
```

- [ ] **Step 6: Re-run the resolve_authority tests**

```bash
pytest tests/test_resolve_authority.py -v
```

Expected: still 6 PASS (no regression).

- [ ] **Step 7: Commit**

```bash
git add src/hospital/pipeline.py tests/test_resolve_authority.py
git commit -m "Route ABE keygen to the owning authority per attribute prefix"
```

---

## Task 6: Patient repo + migrate `register_patient`

**Files:**
- Modify: `src/db/repo.py` (extend), `src/hospital/patient.py`, `scripts/register_patient.py`

- [ ] **Step 1: Add patient functions to `src/db/repo.py`**

Append:

```python
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
```

- [ ] **Step 2: Rewrite `src/hospital/patient.py`**

Replace the whole file with:

```python
"""
Patient directory management + DB-backed registration.

Filesystem layout is unchanged: each patient still gets a
Patient_{id}/{Plaindata,Classifieddata,DataAttribute,Accesspolicy}/ tree.
The CSV mapping is retired — GID ↔ Patient_ID now lives in SQLite.
"""
from pathlib import Path

from src.db import repo as _repo

PATIENT_SUBDIRS = ("Plaindata", "Classifieddata", "DataAttribute", "Accesspolicy")


def get_patient_dir(patients_dir, patient_id):
    return Path(patients_dir) / f"Patient_{patient_id}"


def create_patient_directory(patients_dir, patient_id):
    patient_dir = get_patient_dir(patients_dir, patient_id)
    for sub in PATIENT_SUBDIRS:
        (patient_dir / sub).mkdir(parents=True, exist_ok=True)
    return patient_dir


def register_patient(hospital_name, patients_dir, gid, patient_id):
    """Create filesystem layout AND insert the patients row."""
    patient_dir = create_patient_directory(patients_dir, patient_id)
    _repo.register_patient(hospital_name, patient_id, gid)
    print(f"Registered patient {patient_id} (GID={gid}) under {hospital_name}")
    return patient_dir


def list_patients(hospital_name):
    """Return patient IDs registered for *hospital_name* (DB-backed)."""
    return _repo.list_patients(hospital_name)
```

- [ ] **Step 3: Update `scripts/register_patient.py`**

Replace the body of `main()` with the new signature:

```python
def main():
    hospitals = list(config.HOSPITAL_CONFIGS.keys())
    print("Available hospitals: " + ", ".join(hospitals))
    hospital_name = input("Select hospital: ").strip()
    if hospital_name not in config.HOSPITAL_CONFIGS:
        print(f"Unknown hospital '{hospital_name}'.")
        sys.exit(1)

    gid = input("Enter patient GID: ").strip()
    patient_id = input("Enter Patient ID (e.g. PT12345): ").strip()
    if not gid or not patient_id:
        print("GID and Patient ID cannot be empty.")
        sys.exit(1)

    hospital_cfg = config.HOSPITAL_CONFIGS[hospital_name]
    register_patient(
        hospital_name=hospital_name,
        patients_dir=hospital_cfg["patients_dir"],
        gid=gid,
        patient_id=patient_id,
    )
```

- [ ] **Step 4: Update `scripts/run_pipeline.py` `_prompt_patient`**

In `scripts/run_pipeline.py`, replace the `_prompt_patient` body to pass `hospital_name` to `list_patients`:

```python
def _prompt_patient(hospital_name):
    available = list_patients(hospital_name)
    if available:
        print("Registered patients: " + ", ".join(available))
    patient_id = input("Enter Patient ID (e.g. PT12345): ").strip()
    xml_filename = input(
        f"Enter XML filename (e.g. Patient_{patient_id}_1.xml): "
    ).strip()
    return patient_id, xml_filename
```

- [ ] **Step 5: Smoke-check by re-registering an existing fixture patient**

```bash
echo "Hospital1
GID67890
PT12345" | python scripts/register_patient.py
```

Expected: prints "Registered patient PT12345 (GID=GID67890) under Hospital1".

Verify:

```bash
sqlite3 data/abe.db "SELECT * FROM patients;"
```

Expected: at least one row for PT12345 in Hospital1.

- [ ] **Step 6: Delete the obsolete CSV mapping files**

```bash
git rm -f data/Hospital1/patient_mapping.csv data/Hospital2/patient_mapping.csv
```

(If they don't exist in git, skip silently.)

- [ ] **Step 7: Commit**

```bash
git add src/db/repo.py src/hospital/patient.py scripts/register_patient.py scripts/run_pipeline.py
git commit -m "Migrate patient registration from CSV to SQLite"
```

---

## Task 7: Persist patient_records + ciphertexts during pipeline run

**Files:**
- Modify: `src/db/repo.py` (extend), `src/hospital/pipeline.py`

- [ ] **Step 1: Add record + ciphertext functions to `src/db/repo.py`**

Append:

```python
# ---------------------------------------------------------------------------
# Patient records (per pipeline run)
# ---------------------------------------------------------------------------

def save_patient_record(
    hospital_name,
    patient_id,
    xml_filename,
    document_label,
    section_labels,
    policy_text,
    abe_policy_string,
):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patient_records "
            "(hospital_name, patient_id, xml_filename, document_label, "
            " section_labels_json, policy_text, abe_policy_string) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                hospital_name,
                patient_id,
                xml_filename,
                document_label,
                json.dumps(section_labels),
                policy_text,
                abe_policy_string,
            ),
        )
        conn.commit()
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Ciphertexts
# ---------------------------------------------------------------------------

def save_ciphertext(group, record_id, ciphertext):
    """Serialise *ciphertext* (dict from HybridABEncMA.encrypt) and store it."""
    blob = serialize_obj(group, ciphertext)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO ciphertexts (record_id, ciphertext_bytes) VALUES (?, ?)",
            (record_id, blob),
        )
        conn.commit()
        return cur.lastrowid


def load_ciphertext(group, ciphertext_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ciphertext_bytes FROM ciphertexts WHERE id = ?",
            (ciphertext_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Ciphertext not found: {ciphertext_id}")
    return deserialize_obj(group, row["ciphertext_bytes"])
```

- [ ] **Step 2: Wire `pipeline.run_pipeline` to persist a record + ciphertext**

In `src/hospital/pipeline.py`, locate the entire Stage 4 section and the existing `try`/`except` decrypt block immediately after it. Replace **everything from `# Stage 4` through the end of `run_pipeline` (but stop before the final decrypt try/except — that block is replaced separately in Task 8)** with the following. The decrypt block is intentionally left in place untouched here; Task 8 will swap it.

```python
    # -----------------------------------------------------------------------
    # Stage 4 — ABE Encryption + Decryption
    # -----------------------------------------------------------------------
    print("\n[4/4] Running ABE encryption...")
    hyb, group, gp, all_pk = _build_abe_system()

    # Derive the ABE policy string before anything that needs it
    policy_str = parse_policy_to_abe_format(policy_text, prefix)
    if not policy_str:
        print("  WARNING: Could not derive an ABE policy string from the generated text.")
        return False

    # Persist the pipeline run as a patient_records row
    record_id = repo.save_patient_record(
        hospital_name=hospital_name,
        patient_id=patient_id,
        xml_filename=xml_filename,
        document_label=document_label,
        section_labels=section_labels,
        policy_text=policy_text,
        abe_policy_string=policy_str,
    )

    # Issue keys per owning authority (the multi-authority fix)
    user_keys = {}
    for attr in user_attributes:
        authority_name = resolve_authority(attr)
        authority_sk = repo.load_authority_sk(group, authority_name)
        hyb.keygen(gp, authority_sk, attr, user_gid, user_keys)

    # Encrypt the plain XML bytes
    xml_bytes = plain_xml.read_bytes()
    print(f"  ABE policy : {policy_str}")
    ct = hyb.encrypt(gp, all_pk, xml_bytes, policy_str)
    repo.save_ciphertext(group, record_id, ct)

    # Verify round-trip (this try/except is replaced in Task 8 with audit logging)
    try:
        decrypted = hyb.decrypt(gp, user_keys, ct)
        assert decrypted == xml_bytes, "Decrypted content does not match original!"
        print("  Decryption successful. Round-trip verified.")
        return True
    except Exception as exc:
        print(f"  Decryption failed: {exc}")
        return False
```

End state of Stage 4 after this step: build → derive policy_str → save record → per-authority keygen → encrypt → save ciphertext → decrypt round-trip. Task 8 will swap the final try/except for the audit-logging version.

- [ ] **Step 3: Manual end-to-end check (depends on trained models being present)**

This step is skippable if models aren't trained; Task 10's smoke test covers the same path with stubs. If you have trained models:

```bash
python scripts/run_pipeline.py
# Hospital1, PT12345, Patient_PT12345_1.xml, alice, hospitalA.doctor,hospitalA.cardiology
sqlite3 data/abe.db "SELECT id, hospital_name, patient_id, document_label, abe_policy_string FROM patient_records;"
sqlite3 data/abe.db "SELECT id, record_id, length(ciphertext_bytes) FROM ciphertexts;"
```

Expected: one row in each table, non-zero ciphertext size.

- [ ] **Step 4: Commit**

```bash
git add src/db/repo.py src/hospital/pipeline.py
git commit -m "Persist patient_records and ciphertexts each pipeline run"
```

---

## Task 8: Persist audit runs

**Files:**
- Modify: `src/db/repo.py` (extend), `src/hospital/pipeline.py`

- [ ] **Step 1: Add audit function to `src/db/repo.py`**

Append:

```python
# ---------------------------------------------------------------------------
# Audit runs
# ---------------------------------------------------------------------------

def save_audit_run(
    record_id,
    user_gid,
    user_attributes,
    outcome,
    duration_ms,
    error_message=None,
):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_runs "
            "(record_id, user_gid, user_attributes_json, outcome, duration_ms, error_message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                user_gid,
                json.dumps(user_attributes),
                outcome,
                duration_ms,
                error_message,
            ),
        )
        conn.commit()
```

- [ ] **Step 2: Wrap the decrypt block in `pipeline.run_pipeline`**

Replace the existing decrypt try/except at the bottom of `run_pipeline` with:

```python
    import time

    t0 = time.time()
    try:
        decrypted = hyb.decrypt(gp, user_keys, ct)
        duration_ms = int((time.time() - t0) * 1000)
        if decrypted != xml_bytes:
            repo.save_audit_run(
                record_id, user_gid, user_attributes,
                outcome="error", duration_ms=duration_ms,
                error_message="Round-trip mismatch",
            )
            print("  Decryption mismatch.")
            return False
        repo.save_audit_run(
            record_id, user_gid, user_attributes,
            outcome="success", duration_ms=duration_ms,
        )
        print(f"  Decryption successful ({duration_ms} ms).")
        return True
    except Exception as exc:
        duration_ms = int((time.time() - t0) * 1000)
        outcome = "policy_not_satisfied" if "satisfy" in str(exc).lower() else "error"
        repo.save_audit_run(
            record_id, user_gid, user_attributes,
            outcome=outcome, duration_ms=duration_ms,
            error_message=str(exc),
        )
        print(f"  Decryption failed ({outcome}): {exc}")
        return False
```

- [ ] **Step 3: Commit**

```bash
git add src/db/repo.py src/hospital/pipeline.py
git commit -m "Record audit_runs for each decrypt attempt"
```

---

## Task 9: Fix the empty-Content crash + remove dead hybrid check

**Files:**
- Modify: `src/classification/classifier.py`, `src/abe/hybrid.py`
- Create: `tests/test_classifier_guard.py`

- [ ] **Step 1: Write the failing classifier test**

Create `tests/test_classifier_guard.py`:

```python
from src.classification.classifier import SecurityClassifier


def test_classify_document_empty_returns_public():
    """If no sections were classified (e.g. XML had no <Content>),
    classify_document must NOT divide by zero. It returns 'Public'."""
    # We bypass __init__ to avoid loading the BERT model in unit tests.
    clf = SecurityClassifier.__new__(SecurityClassifier)
    assert clf.classify_document({}) == "Public"
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest tests/test_classifier_guard.py -v
```

Expected: FAIL with `ZeroDivisionError`.

- [ ] **Step 3: Patch `src/classification/classifier.py`**

Replace the body of `classify_document` (around lines 62-87) with:

```python
    def classify_document(self, section_labels, thresholds=None):
        """
        Aggregate per-section labels into a single document-level label.

        If *section_labels* is empty (e.g. the XML had no <Content> element),
        returns "Public" rather than dividing by zero.
        """
        if not section_labels:
            return "Public"
        thresholds = thresholds or CLASSIFICATION_THRESHOLDS
        counts = Counter(section_labels.values())
        total = len(section_labels)
        for label, threshold in thresholds.items():
            if counts[label] / total >= threshold:
                return label
        return "Public"
```

- [ ] **Step 4: Re-run, confirm pass**

```bash
pytest tests/test_classifier_guard.py -v
```

Expected: PASS.

- [ ] **Step 5: Remove dead `is False` block from `src/abe/hybrid.py`**

Replace `decrypt` (lines 53-57) with:

```python
    def decrypt(self, gp, sk, ct):
        key = self._scheme.decrypt(gp, sk, ct["c1"])
        return AuthenticatedCryptoAbstraction(sha2(key)).decrypt(ct["c2"])
```

(`Dabe.decrypt` raises on failure; the wrapper lets that propagate.)

- [ ] **Step 6: Commit**

```bash
git add src/classification/classifier.py src/abe/hybrid.py tests/test_classifier_guard.py
git commit -m "Guard empty section_labels; drop dead is-False check in hybrid decrypt"
```

---

## Task 10: Injectable ML stages + end-to-end smoke test

**Files:**
- Modify: `src/hospital/pipeline.py` (add `classifier=` / `extractor=` kwargs)
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Allow injection in `run_pipeline`**

Change the `run_pipeline` signature (currently `def run_pipeline(hospital_name, patient_id, xml_filename, user_gid, user_attributes):`) to:

```python
def run_pipeline(
    hospital_name,
    patient_id,
    xml_filename,
    user_gid,
    user_attributes,
    classifier=None,
    extractor=None,
):
```

Then change the lines that instantiate them. Replace:

```python
    classifier = SecurityClassifier()
```

with:

```python
    if classifier is None:
        classifier = SecurityClassifier()
```

And replace:

```python
    extractor = PolicyExtractor()
```

with:

```python
    if extractor is None:
        extractor = PolicyExtractor()
```

- [ ] **Step 2: Create `scripts/smoke_test.py`**

```python
#!/usr/bin/env python3
"""
End-to-end smoke test. Exercises the full persistence + ABE workflow with
deterministic ML stubs, so it does not require trained models.

Scenarios:
  1. User with hospitalA.doctor + hospitalA.oncology decrypts successfully.
  2. User with only hospitalA.nurse is denied (policy not satisfied).
  3. Cross-authority user (hospitalA.doctor + insCoA.underwriter) succeeds —
     regression test for the multi-authority keygen bug.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _patch_config(tmp_data_dir, tmp_db_path):
    """Point config at a temp data dir + temp DB before importing pipeline modules."""
    os.environ["DATA_DIR"] = str(tmp_data_dir)
    os.environ["DB_PATH"] = str(tmp_db_path)


def _stub_classifier():
    class _Stub:
        def classify_sections(self, sections):
            # Force at least 30% Highly Confidential so doc label = "Highly Confidential"
            return {tag: "Highly Confidential" for tag in sections}

        def classify_document(self, section_labels, thresholds=None):
            return "Highly Confidential" if section_labels else "Public"

    return _Stub()


def _stub_extractor():
    class _Stub:
        def extract(self, prompt_text):
            # Deterministic, parser-friendly output
            return "Data accessible by Doctor in Oncology with clearance level 2."
    return _Stub()


def _copy_fixture_patient(src_root, dst_root, hospital, patient_id):
    src = src_root / hospital / "patients" / f"Patient_{patient_id}"
    dst = dst_root / hospital / "patients" / f"Patient_{patient_id}"
    for sub in ("Plaindata", "Classifieddata", "DataAttribute", "Accesspolicy"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "Plaindata" / f"Patient_{patient_id}_1.xml",
                dst / "Plaindata" / f"Patient_{patient_id}_1.xml")


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="abe_smoke_"))
    tmp_data = tmp_root / "data"
    tmp_db = tmp_data / "abe.db"
    print(f"Using temp data dir: {tmp_data}")

    _patch_config(tmp_data, tmp_db)

    # Re-import config so the env vars are picked up.
    import importlib
    from src import config as _cfg
    importlib.reload(_cfg)

    from src.db.connection import init_db
    from src.db import repo
    from src.hospital.patient import register_patient
    from src.hospital.pipeline import run_pipeline
    from charm.toolbox.pairinggroup import PairingGroup
    from src.abe.dabe import Dabe

    # 1. Init DB + authority keys
    init_db()
    group = PairingGroup(_cfg.PAIRING_GROUP)
    dabe = Dabe(group)
    gp = dabe.setup()
    repo.save_global_params(group, gp, _cfg.PAIRING_GROUP)
    for name, cfg in _cfg.HOSPITAL_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "hospital", cfg["prefix"], sk, pk)
    for name, cfg in _cfg.INSURANCE_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "insurance", cfg["prefix"], sk, pk)

    # 2. Copy a fixture patient under the temp data root
    fixture_root = ROOT / "data"
    _copy_fixture_patient(fixture_root, tmp_data, "Hospital1", "PT12345")
    register_patient(
        hospital_name="Hospital1",
        patients_dir=_cfg.HOSPITAL_CONFIGS["Hospital1"]["patients_dir"],
        gid="GID-smoke",
        patient_id="PT12345",
    )

    # 3. Scenario 1: doctor in oncology — should decrypt
    ok = run_pipeline(
        "Hospital1", "PT12345", "Patient_PT12345_1.xml",
        user_gid="alice@hospitalA",
        user_attributes=["hospitalA.doctor", "hospitalA.oncology", "hospitalA.clearance2"],
        classifier=_stub_classifier(),
        extractor=_stub_extractor(),
    )
    assert ok, "Scenario 1 (authorised doctor) should succeed"
    print("Scenario 1 PASS")

    # 4. Scenario 2: nurse only — should be denied
    ok = run_pipeline(
        "Hospital1", "PT12345", "Patient_PT12345_1.xml",
        user_gid="bob@hospitalA",
        user_attributes=["hospitalA.nurse"],
        classifier=_stub_classifier(),
        extractor=_stub_extractor(),
    )
    assert not ok, "Scenario 2 (unauthorised nurse) should be denied"
    print("Scenario 2 PASS")

    # 5. Scenario 3: cross-authority user — should still decrypt
    ok = run_pipeline(
        "Hospital1", "PT12345", "Patient_PT12345_1.xml",
        user_gid="carol@cross",
        user_attributes=[
            "hospitalA.doctor", "hospitalA.oncology", "hospitalA.clearance2",
            "insCoA.underwriter",  # extra cross-authority attr
        ],
        classifier=_stub_classifier(),
        extractor=_stub_extractor(),
    )
    assert ok, "Scenario 3 (cross-authority user) should succeed — regression for multi-authority keygen bug"
    print("Scenario 3 PASS")

    # 6. Audit log sanity
    import sqlite3
    rows = sqlite3.connect(str(tmp_db)).execute(
        "SELECT outcome, COUNT(*) FROM audit_runs GROUP BY outcome"
    ).fetchall()
    print(f"Audit summary: {rows}")
    outcomes = dict(rows)
    assert outcomes.get("success", 0) >= 2
    assert outcomes.get("policy_not_satisfied", 0) >= 1

    print("\nALL SMOKE SCENARIOS PASSED")
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test**

```bash
python scripts/smoke_test.py
```

Expected output:
```
Scenario 1 PASS
Scenario 2 PASS
Scenario 3 PASS
Audit summary: [('policy_not_satisfied', 1), ('success', 2)]
ALL SMOKE SCENARIOS PASSED
```

If Scenario 3 fails, the multi-authority keygen routing in Task 5 has regressed — re-check `resolve_authority` and the keygen loop in `pipeline.py`.

- [ ] **Step 4: Commit**

```bash
git add src/hospital/pipeline.py scripts/smoke_test.py
git commit -m "Add end-to-end smoke test with injectable ML stages"
```

---

## Task 11: README + cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

```bash
head -80 README.md
```

- [ ] **Step 2: Update the "Running the Pipeline" section**

Find the section in `README.md` that documents the run/setup flow. Replace it with:

```markdown
## Setup (one-time, per fresh checkout)

```bash
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirement.txt

python scripts/init_db.py                  # create data/abe.db
python scripts/generate_authority_keys.py  # populate authorities + global params
```

## Running the pipeline

```bash
python scripts/register_patient.py    # register a new patient (DB + filesystem)
python scripts/run_pipeline.py        # full pipeline: classify → policy → encrypt/decrypt
python scripts/smoke_test.py          # end-to-end check, uses ML stubs (no trained models needed)
```

State now lives in `data/abe.db` (SQLite). The previous JSON key files and CSV patient mappings have been retired.
```

(Adapt to match the existing README's heading style.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Update README for the SQLite-backed workflow"
```

---

## Done criteria

After all tasks:

```bash
pytest tests/ -v          # all green
python scripts/smoke_test.py   # all three scenarios PASS
git log --oneline -15     # one commit per task, ~11 commits ahead
```

The persistence layer is in place and the multi-authority keygen bug is fixed. We are now unblocked on the real work: classification model, policy extraction model, ABE scheme contribution.
