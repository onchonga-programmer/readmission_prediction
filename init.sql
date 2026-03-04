
-- patients_raw: raw ingestion table for the Diabetes 130-US Hospitals dataset
-- All columns stored as TEXT to preserve raw values exactly as they appear in the CSV.
-- Downstream cleaning / typing happens in later pipeline stages.

CREATE TABLE IF NOT EXISTS patients_raw (
    encounter_id                BIGINT,
    patient_nbr                 BIGINT,
    race                        TEXT,
    gender                      TEXT,
    age                         TEXT,
    weight                      TEXT,
    admission_type_id           INTEGER,
    discharge_disposition_id    INTEGER,
    admission_source_id         INTEGER,
    time_in_hospital            INTEGER,
    payer_code                  TEXT,
    medical_specialty           TEXT,
    num_lab_procedures          INTEGER,
    num_procedures              INTEGER,
    num_medications             INTEGER,
    number_outpatient           INTEGER,
    number_emergency            INTEGER,
    number_inpatient            INTEGER,
    diag_1                      TEXT,
    diag_2                      TEXT,
    diag_3                      TEXT,
    number_diagnoses            INTEGER,
    max_glu_serum               TEXT,
    a1cresult                   TEXT,
    metformin                   TEXT,
    repaglinide                 TEXT,
    nateglinide                 TEXT,
    chlorpropamide              TEXT,
    glimepiride                 TEXT,
    acetohexamide               TEXT,
    glipizide                   TEXT,
    glyburide                   TEXT,
    tolbutamide                 TEXT,
    pioglitazone                TEXT,
    rosiglitazone               TEXT,
    acarbose                    TEXT,
    miglitol                    TEXT,
    troglitazone                TEXT,
    tolazamide                  TEXT,
    examide                     TEXT,
    citoglipton                 TEXT,
    insulin                     TEXT,
    glyburide_metformin         TEXT,
    glipizide_metformin         TEXT,
    glimepiride_pioglitazone    TEXT,
    metformin_rosiglitazone     TEXT,
    metformin_pioglitazone      TEXT,
    change_col                  TEXT,
    diabetesmed                 TEXT,
    readmitted                  TEXT,
    ingested_at                 TIMESTAMP DEFAULT NOW()
);

-- Index for the most common lookup patterns
CREATE INDEX IF NOT EXISTS idx_patients_raw_patient_nbr    ON patients_raw (patient_nbr);
CREATE INDEX IF NOT EXISTS idx_patients_raw_encounter_id   ON patients_raw (encounter_id);
CREATE INDEX IF NOT EXISTS idx_patients_raw_readmitted     ON patients_raw (readmitted);