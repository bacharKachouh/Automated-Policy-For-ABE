# Automated Policy for ABE

A **Data-Centric Security Model** for healthcare data privacy.  The system automatically classifies patient records, generates access control policies, and encrypts data using Attribute-Based Encryption (ABE).

## Pipeline Overview

```
Patient XML  →  Security Classification  →  Attribute Extraction  →  Policy Generation  →  ABE Encryption
              (BioClinicalBERT)            (XML metadata)           (GPT-2 fine-tuned)    (Lewko-Waters)
```

1. **Security Classification** — BioClinicalBERT classifies each section of a patient XML record into one of four sensitivity levels: *Highly Confidential*, *Confidential*, *Restricted*, *Public*.
2. **Attribute Extraction** — Metadata fields (data type, department, purpose, emergency flag) are extracted from the classified record.
3. **Policy Generation** — A fine-tuned GPT-2 model generates a natural-language access policy from the extracted attributes.
4. **Hybrid ABE Encryption** — The policy is converted into a Charm-Crypto policy string and the record is encrypted using the Lewko-Waters decentralised ABE scheme (ABE key encapsulation + AES).

## Requirements

- **Python 3.7** is required — Charm-Crypto 0.50 only supports Python 3.7.
- A virtual environment is mandatory to avoid package conflicts.
- A GPU is strongly recommended for model training.

## Setup

```bash
python3.7 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirement.txt
```

## First-Time Setup

### 1 — Train the models

```bash
# Optional: regenerate the synthetic training data
python training/generate_data.py

# Fine-tune BioClinicalBERT (security classifier)
python scripts/train_classifier.py

# Fine-tune GPT-2 (policy extractor)
python scripts/train_policy_extractor.py
```

Trained models are saved to `models/security_classifier/` and `models/policy_extractor/`.

### 2 — Generate authority keys

```bash
python scripts/generate_authority_keys.py
```

### 3 — Register a patient

```bash
python scripts/register_patient.py
```

Place the patient XML file inside the created `Plaindata/` folder.

## Running the Pipeline

```bash
python scripts/run_pipeline.py
```

You will be prompted for the hospital, patient ID, XML filename, requesting user GID, and user attributes.

> **Before each run**, make sure `Classifieddata/`, `DataAttribute/`, and `Accesspolicy/` are empty for the target patient — only `Plaindata/` should contain the input XML.

## Project Structure

```
src/                    ← core Python package
  config.py             ← all paths and settings (override via env vars)
  abe/                  ← Lewko-Waters DABE + hybrid encryption + policy parser
  classification/       ← BioClinicalBERT security classifier
  policy/               ← GPT-2 policy extractor
  hospital/             ← patient management + pipeline orchestration
  utils/                ← XML parsing utilities

scripts/                ← executable entry points
  run_pipeline.py
  register_patient.py
  generate_authority_keys.py
  train_classifier.py
  train_policy_extractor.py

training/
  data/                 ← CSV training datasets
  generate_data.py      ← synthetic data generator

data/
  Hospital1/
    keys/               ← authority secret key
    patients/           ← patient records (Plaindata, Classifieddata, …)
  Hospital2/

keys/                   ← shared public keys and global parameters
models/                 ← trained model artifacts (gitignored, built by training scripts)
```

## Configuration

Copy `.env.example` to `.env` and set any paths that differ from the defaults:

```bash
cp .env.example .env
```

All configuration lives in `src/config.py` and can be overridden via environment variables — no hardcoded paths anywhere in the codebase.

## Contact

Developed by Bachar KACHOUH — [bachar.kachouh@hotmail.com](mailto:bachar.kachouh@hotmail.com)
