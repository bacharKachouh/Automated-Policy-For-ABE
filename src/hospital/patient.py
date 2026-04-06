"""
Patient directory management and GID → PatientID mapping.
"""
import csv
from pathlib import Path

PATIENT_SUBDIRS = ("Plaindata", "Classifieddata", "DataAttribute", "Accesspolicy")


def get_patient_dir(patients_dir, patient_id):
    """Return the Path for ``Patient_{patient_id}`` inside *patients_dir*."""
    return Path(patients_dir) / f"Patient_{patient_id}"


def create_patient_directory(patients_dir, patient_id):
    """
    Create the standard subdirectory tree for a new patient.

    Creates ``Patient_{patient_id}/{Plaindata,Classifieddata,DataAttribute,Accesspolicy}/``
    under *patients_dir*.  Safe to call on an existing patient (idempotent).

    Returns
    -------
    Path
        Path to the patient root directory.
    """
    patient_dir = get_patient_dir(patients_dir, patient_id)
    for sub in PATIENT_SUBDIRS:
        (patient_dir / sub).mkdir(parents=True, exist_ok=True)
    return patient_dir


def register_patient(patients_dir, gid, patient_id, mapping_file):
    """
    Create the patient directory and record the GID → PatientID mapping.

    Parameters
    ----------
    patients_dir : str or Path
        Base patients directory for the hospital.
    gid : str
        Global identity of the patient.
    patient_id : str
        Hospital-local patient ID (e.g. ``"PT12345"``).
    mapping_file : str or Path
        CSV file storing the GID ↔ PatientID table.

    Returns
    -------
    Path
        Path to the created patient directory.
    """
    patient_dir = create_patient_directory(patients_dir, patient_id)
    mapping_file = Path(mapping_file)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not mapping_file.exists()
    with open(mapping_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["GID", "Patient_ID"])
        writer.writerow([gid, patient_id])
    print(f"Registered patient {patient_id} (GID={gid})")
    return patient_dir


def list_patients(patients_dir):
    """
    Return a sorted list of patient IDs found in *patients_dir*.

    Returns
    -------
    list[str]
        IDs extracted from directory names matching ``Patient_*``.
    """
    patients_dir = Path(patients_dir)
    if not patients_dir.exists():
        return []
    return sorted(
        p.name[len("Patient_"):]
        for p in patients_dir.iterdir()
        if p.is_dir() and p.name.startswith("Patient_")
    )
