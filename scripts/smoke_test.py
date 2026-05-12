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
