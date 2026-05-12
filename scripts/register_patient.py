#!/usr/bin/env python3
"""
Register a new patient: create the standard directory structure and
update the GID → PatientID mapping CSV for the chosen hospital.

Usage
-----
    python scripts/register_patient.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.hospital.patient import register_patient


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


if __name__ == "__main__":
    main()
