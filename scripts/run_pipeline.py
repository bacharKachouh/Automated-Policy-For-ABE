#!/usr/bin/env python3
"""
Interactive entry point for the Automated-ABE pipeline.

Usage
-----
    python scripts/run_pipeline.py

The script prompts for a hospital, patient ID, XML filename, requesting user
GID, and user attributes, then runs all four pipeline stages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.hospital.patient import list_patients
from src.hospital.pipeline import run_pipeline


def _prompt_hospital():
    hospitals = list(config.HOSPITAL_CONFIGS.keys())
    print("Available hospitals: " + ", ".join(hospitals))
    choice = input("Select hospital: ").strip()
    if choice not in config.HOSPITAL_CONFIGS:
        print(f"Unknown hospital '{choice}'.")
        sys.exit(1)
    return choice


def _prompt_patient(hospital_name):
    patients_dir = config.HOSPITAL_CONFIGS[hospital_name]["patients_dir"]
    available = list_patients(patients_dir)
    if available:
        print("Registered patients: " + ", ".join(available))
    patient_id = input("Enter Patient ID (e.g. PT12345): ").strip()
    xml_filename = input(
        f"Enter XML filename (e.g. Patient_{patient_id}_1.xml): "
    ).strip()
    return patient_id, xml_filename


def _prompt_user(hospital_name):
    prefix = config.HOSPITAL_CONFIGS[hospital_name]["prefix"]
    user_gid = input("Enter requesting user GID: ").strip()
    print(f"\nAttribute prefix for {hospital_name}: {prefix}")
    print("Examples: doctor, nurse, oncology, clearance2")
    raw = input("Enter user attributes (comma-separated short names): ").strip()
    user_attributes = [f"{prefix}.{a.strip()}" for a in raw.split(",") if a.strip()]
    return user_gid, user_attributes


def main():
    print("=== Automated ABE Healthcare Data Pipeline ===\n")

    hospital_name = _prompt_hospital()
    patient_id, xml_filename = _prompt_patient(hospital_name)
    user_gid, user_attributes = _prompt_user(hospital_name)

    print()
    success = run_pipeline(hospital_name, patient_id, xml_filename, user_gid, user_attributes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
