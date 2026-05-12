# Persistence Migration & Workflow Validation — Design

**Date:** 2026-05-12
**Status:** Approved
**Branch:** `reorganise` (or follow-up branch)

## Context

Automated-ABE is the implementation backing a research paper on a zero-knowledge, three-pillar architecture (Data / Governance / Attribute) for multi-party healthcare data sharing using Lewko-Waters multi-authority ABE. Multi-parties = hospitals + insurance companies, each acting as an independent ABE authority over a disjoint attribute namespace.

The paper's contribution areas are:

1. Demonstrating that **security classification → policy generation** is a solid approach (Data Pillar → Governance Pillar).
2. An **ABE scheme contribution** — either a new scheme or an extension of Lewko-Waters (Attribute Pillar).

This design covers the **supporting infrastructure** only. It is not a contribution; it is plumbing that unblocks both contribution tracks by making the workflow solid, persistent, and end-to-end reproducible.

## Goals

- Migrate from filesystem-scattered state (CSV mappings, broken JSON key dumps, in-memory regeneration of authority keys on every run) to a single SQLite database with proper Charm-Crypto serialisation.
- Fix critical correctness bugs that contradict the paper's architectural claims, in particular the single-authority keygen bug in the pipeline.
- Keep XML payloads on the filesystem — they are large, human-readable, and not what the DB is for.
- Validate the workflow end-to-end with a smoke test, including the cross-authority case.

## Non-Goals

- User / role management (no users table, no per-user attribute issuance flow).
- Request / access decision split (one-shot CLI pipeline remains).
- Experiment / evaluation harness (deferred — re-brainstorm as Approach B in a follow-up session).
- Postgres or any non-SQLite database.
- ORM (direct `sqlite3` with parameterised queries).

## Architecture

```
src/
  config.py                    ← unchanged: paths, hospital/insurance configs
  db/                          ← NEW
    __init__.py
    schema.sql                 ← table definitions, single source of truth
    connection.py              ← get_conn() context manager, applies schema on first run
    repo.py                    ← thin domain functions; no ORM
  abe/
    dabe.py, hybrid.py         ← crypto math unchanged
    authority.py               ← Charm-native serialisation replaces JSON-string dumps
    serialization.py           ← NEW: serialize_sk / deserialize_sk, ciphertext helpers
    policy_parser.py           ← unchanged
  classification/, policy/     ← unchanged
  hospital/
    patient.py                 ← register_patient writes to DB instead of CSV
    pipeline.py                ← loads persisted authority keys; routes keygen per-authority
  utils/xml_utils.py           ← unchanged

scripts/
  init_db.py                   ← NEW: create SQLite file + apply schema
  generate_authority_keys.py   ← rewritten to persist into DB via repo
  register_patient.py          ← unchanged surface; writes to DB
  run_pipeline.py              ← loads keys from DB instead of regenerating
  smoke_test.py                ← NEW: end-to-end workflow check

data/
  abe.db                       ← NEW: single SQLite file, gitignored
  Hospital1/patients/...       ← unchanged: XML payloads stay on disk
  Hospital2/patients/...       ← unchanged
  (CSV mappings and JSON key files retired)
```

### Module boundaries

- **`db/`** is the only module that touches SQL. All other code calls `repo` functions (e.g. `repo.load_authority_sk(name)`, `repo.save_ciphertext(...)`). Swapping SQLite for Postgres later changes only this module.
- **`abe/`** does not know about persistence. It still operates on in-memory Charm objects. The new `abe/serialization.py` exposes serialize/deserialize helpers that `repo` calls when writing/reading BLOB columns.
- **`pipeline.py`** is the only orchestrator. Scripts are argument parsing + a call into pipeline functions.
- **No ORM.** Roughly 8-10 repo functions covering the workflow are simpler and easier to debug than fighting SQLAlchemy models around Charm objects.

## Database Schema

```sql
-- One row per authority (Hospital1, Hospital2, InsCoA, InsCoB)
CREATE TABLE authorities (
    name        TEXT PRIMARY KEY,           -- "Hospital1", "InsCoA"
    type        TEXT NOT NULL CHECK(type IN ('hospital','insurance')),
    prefix      TEXT NOT NULL UNIQUE,       -- "hospitalA", "insCoA"
    sk_bytes    BLOB NOT NULL,              -- Charm-serialised secret-key dict
    pk_bytes    BLOB NOT NULL,              -- Charm-serialised public-key dict
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Singleton row: global pairing parameters
CREATE TABLE global_params (
    id              INTEGER PRIMARY KEY CHECK(id = 1),
    pairing_group   TEXT NOT NULL,          -- "SS512"
    g_bytes         BLOB NOT NULL,          -- group.serialize(gp["g"])
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- H is reconstructed from the group at load time (lambda x: group.hash(x, G1))

CREATE TABLE patients (
    patient_id      TEXT NOT NULL,          -- "PT12345"
    hospital_name   TEXT NOT NULL REFERENCES authorities(name),
    gid             TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hospital_name, patient_id)
);

-- One row per pipeline run on a specific XML file
CREATE TABLE patient_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_name        TEXT NOT NULL,
    patient_id           TEXT NOT NULL,
    xml_filename         TEXT NOT NULL,
    document_label       TEXT,
    section_labels_json  TEXT,              -- {"Diagnosis":"Confidential",...}
    policy_text          TEXT,              -- raw GPT-2 output
    abe_policy_string    TEXT,              -- "(hospitalA.doctor OR ...) AND ..."
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hospital_name, patient_id) REFERENCES patients(hospital_name, patient_id)
);

CREATE TABLE ciphertexts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id           INTEGER NOT NULL REFERENCES patient_records(id),
    ciphertext_bytes    BLOB NOT NULL,      -- serialised hybrid ct (c1 ABE + c2 AES)
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id            INTEGER REFERENCES patient_records(id),
    user_gid             TEXT,
    user_attributes_json TEXT,
    outcome              TEXT NOT NULL,     -- "success" | "policy_not_satisfied" | "error"
    duration_ms          INTEGER,
    error_message        TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**What's NOT in the DB:** plain XML files, classified XML files, the `DataAttribute/` prompt files, the `Accesspolicy/` text files. The filesystem remains the source of truth for human-readable artefacts. Paths are derived from `config.py` + patient_id + filename.

**Serialisation:** Charm group elements use `objectToBytes` / `bytesToObject` (or `group.serialize` / `group.deserialize` for individual elements) — they round-trip cleanly, unlike the current `str(...)` JSON dumps.

## Workflow

### One-time setup (per fresh checkout)

```
python scripts/init_db.py                   # create data/abe.db, apply schema
python scripts/generate_authority_keys.py   # run Dabe.setup + authsetup per authority,
                                            # persist global params + each authority's SK+PK
```

After this, the DB holds global params and all four authorities (Hospital1, Hospital2, InsCoA, InsCoB) with their SK and PK. The current broken JSON key files are retired.

### Patient registration

```
python scripts/register_patient.py
  → prompts for hospital, GID, patient ID
  → INSERT INTO patients (hospital, patient_id, gid, ...)
  → creates Plaindata/Classifieddata/DataAttribute/Accesspolicy dirs on disk
  → user drops Patient_{id}_1.xml into Plaindata/
```

### Pipeline run (per record)

```
python scripts/run_pipeline.py
  → prompts for hospital, patient_id, xml filename, user_gid, user_attributes
  → load global_params from DB; reconstruct H = lambda x: group.hash(x, G1)
  → load all authorities' PKs from DB (no key regeneration)
  → Stage 1 — classify   → write Classifieddata/XYZ.xml + INSERT patient_records row
  → Stage 2 — extract    → write DataAttribute/XYZ.txt
  → Stage 3 — GPT-2      → write Accesspolicy/XYZ.txt; store policy_text in row
                         → policy_parser → store abe_policy_string in row
  → Stage 4a — encrypt   → INSERT ciphertexts row
  → Stage 4b — per-authority keygen (THE FIX):
        for attr in user_attributes:
            authority_name = resolve_authority(attr)    # by prefix
            sk = repo.load_authority_sk(authority_name)
            hyb.keygen(gp, sk, attr, user_gid, user_keys)
  → decrypt → INSERT audit_runs row (outcome + duration_ms)
```

`resolve_authority(attr)` maps `"hospitalA.doctor"` → `"Hospital1"`, `"insCoA.underwriter"` → `"InsCoA"`, etc. The prefix-to-authority map is derived from `HOSPITAL_CONFIGS` and `INSURANCE_CONFIGS` at module load. A cross-authority user (e.g. a hospital doctor who is also an insurance underwriter) now receives keys from both authorities — which is the whole point of multi-authority LW.

## Bug Fixes Folded Into Migration

1. **Multi-authority keygen** (`src/hospital/pipeline.py:141-144`) — handled by `resolve_authority` above.
2. **ZeroDivisionError on empty `<Content>`** (`src/classification/classifier.py:85`) — guard for empty `section_labels`, return `"Public"`.
3. **Dead `if key is False` check** (`src/abe/hybrid.py:54-56`) — remove; `Dabe.decrypt` raises on failure.
4. **Lambda repr in audit JSON** (`src/abe/authority.py:72`) — irrelevant; that JSON path is retired.

## Error Handling

- **DB errors:** bubble up. Each pipeline stage commits its writes in its own transaction (no whole-run transaction), so a failure mid-pipeline leaves earlier stages' rows on disk for debugging, but no row is ever half-written.
- **Decryption failure** (policy not satisfied): caught, recorded as `audit_runs.outcome = "policy_not_satisfied"`, pipeline exits non-zero.
- **Charm deserialisation failure:** fatal. Clear message: `"authority keys corrupt or scheme mismatch — re-run generate_authority_keys.py"`.
- **Schema drift:** `connection.py` applies `schema.sql` only when the database is created. Schema changes after that require manual migration (acceptable for a research PoC).

## Testing

One end-to-end smoke test: `scripts/smoke_test.py`.

1. Use a temp SQLite file (not in-memory — the workflow expects a path), run `init_db` + `generate_authority_keys`.
2. Register a fixture patient; drop a fixture XML into `Plaindata/`.
3. Run the full pipeline with a known-good attribute set → assert decryption succeeds.
4. Run again with an attribute set that does **not** satisfy the policy → assert outcome is `"policy_not_satisfied"`.
5. Run once with a **cross-authority** user (hospital attrs + insurance attrs) → assert decryption succeeds. This is the regression test for the multi-authority keygen bug.

No unit suite at this stage. Focused unit tests will arrive when we start touching the classification, policy-extraction, and ABE modules for the paper contributions.

## Out of Scope (explicit)

- Replacing GPT-2 with a better policy extractor.
- LSSS matrix construction owned by our code (instead of delegated to Charm).
- Any change to the ABE scheme itself.
- User authentication, sessions, web UI.
- Multi-process / multi-host authority deployment.

These are deliberately left for the contribution tracks listed in `CLAUDE.md` and the paper outline.
