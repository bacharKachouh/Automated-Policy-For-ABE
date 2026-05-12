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
