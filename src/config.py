"""
Centralised configuration for the Automated-ABE system.

All paths and tuneable parameters live here.  Override any value via
environment variables before starting the application.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout — override via env vars for non-default deployments
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
KEYS_DIR = Path(os.environ.get("KEYS_DIR", str(BASE_DIR / "keys")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(BASE_DIR / "models")))
TRAINING_DATA_DIR = BASE_DIR / "training" / "data"

# ---------------------------------------------------------------------------
# Trained model locations
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL_PATH = Path(
    os.environ.get("CLASSIFIER_MODEL_PATH", str(MODELS_DIR / "security_classifier"))
)
POLICY_MODEL_PATH = Path(
    os.environ.get("POLICY_MODEL_PATH", str(MODELS_DIR / "policy_extractor"))
)

# ---------------------------------------------------------------------------
# ABE pairing group
# ---------------------------------------------------------------------------
PAIRING_GROUP = os.environ.get("PAIRING_GROUP", "SS512")

# ---------------------------------------------------------------------------
# Hospital configurations
# ---------------------------------------------------------------------------
HOSPITAL_CONFIGS = {
    "Hospital1": {
        "prefix": "hospitalA",
        "sk_file": DATA_DIR / "Hospital1" / "keys" / "hospital_A_SK.json",
        "patients_dir": DATA_DIR / "Hospital1" / "patients",
        "mapping_file": DATA_DIR / "Hospital1" / "patient_mapping.csv",
        "attributes": [
            "hospitalA.nurse", "hospitalA.doctor", "hospitalA.admin", "hospitalA.researcher",
            "hospitalA.cardiology", "hospitalA.oncology", "hospitalA.pharmacy", "hospitalA.emergency",
            "hospitalA.clearance1", "hospitalA.clearance2", "hospitalA.clearance3",
        ],
    },
    "Hospital2": {
        "prefix": "hospitalB",
        "sk_file": DATA_DIR / "Hospital2" / "keys" / "hospital_B_SK.json",
        "patients_dir": DATA_DIR / "Hospital2" / "patients",
        "mapping_file": DATA_DIR / "Hospital2" / "patient_mapping.csv",
        "attributes": [
            "hospitalB.nurse", "hospitalB.doctor", "hospitalB.admin", "hospitalB.researcher",
            "hospitalB.cardiology", "hospitalB.oncology", "hospitalB.pharmacy", "hospitalB.emergency",
            "hospitalB.clearance1", "hospitalB.clearance2", "hospitalB.clearance3",
        ],
    },
}

INSURANCE_CONFIGS = {
    "InsCoA": {
        "attributes": [
            "insCoA.underwriter", "insCoA.claims_adjuster", "insCoA.customer_service",
            "insCoA.policy_admin", "insCoA.claims_processing",
            "insCoA.policy_expertise_health", "insCoA.policy_expertise_life",
            "insCoA.clearance1", "insCoA.clearance2", "insCoA.clearance3",
        ],
    },
    "InsCoB": {
        "attributes": [
            "insCoB.underwriter", "insCoB.claims_adjuster", "insCoB.customer_service",
            "insCoB.policy_admin", "insCoB.claims_processing",
            "insCoB.policy_expertise_auto", "insCoB.policy_expertise_property",
            "insCoB.clearance1", "insCoB.clearance2", "insCoB.clearance3",
        ],
    },
}

PUBLIC_KEYS_FILE = KEYS_DIR / "public_keys_and_GP.json"

# ---------------------------------------------------------------------------
# Security classification
# ---------------------------------------------------------------------------
# Thresholds are checked in priority order (highest sensitivity first).
# If ≥ threshold fraction of sections carry a label, that label wins.
CLASSIFICATION_THRESHOLDS = {
    "Highly Confidential": 0.3,
    "Confidential": 0.35,
    "Restricted": 0.4,
    "Public": 0.0,
}

SECURITY_LABEL_MAP = {
    0: "Highly Confidential",
    1: "Confidential",
    2: "Restricted",
    3: "Public",
}
