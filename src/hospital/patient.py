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
