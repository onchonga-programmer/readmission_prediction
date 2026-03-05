import pandas as pd
import psycopg2
import os
import time

# Wait for Postgres to be fully ready
print("Waiting for Postgres...")
time.sleep(10)

# Read the CSV file
print("Reading CSV...")
df = pd.read_csv("/data/diabetic_data.csv")
print(f"Loaded {len(df)} rows from CSV")

# Connect to Postgres
# NOTE: host is 'postgres' (the Docker service name), not localhost
conn = psycopg2.connect(
    host="postgres",
    database=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    port=5432
)
cur = conn.cursor()

# Insert rows one by one
print("Inserting rows... this will take a few minutes.")
inserted = 0
for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO patients_raw (
            encounter_id, patient_nbr, race, gender, age, weight,
            admission_type_id, discharge_disposition_id, admission_source_id,
            time_in_hospital, payer_code, medical_specialty,
            num_lab_procedures, num_procedures, num_medications,
            number_outpatient, number_emergency, number_inpatient,
            diag_1, diag_2, diag_3, number_diagnoses,
            max_glu_serum, a1cresult, metformin, repaglinide,
            nateglinide, chlorpropamide, glimepiride, acetohexamide,
            glipizide, glyburide, tolbutamide, pioglitazone,
            rosiglitazone, acarbose, miglitol, troglitazone,
            tolazamide, examide, citoglipton, insulin,
            glyburide_metformin, glipizide_metformin,
            glimepiride_pioglitazone, metformin_rosiglitazone,
            metformin_pioglitazone, change_col, diabetesmed, readmitted
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT DO NOTHING
    """, tuple(row))
    inserted += 1
    if inserted % 10000 == 0:
        conn.commit()
        print(f"  {inserted} rows inserted so far...")

conn.commit()
cur.close()
conn.close()

print(f"Done! {inserted} rows loaded into Postgres.")