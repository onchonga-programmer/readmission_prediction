import pandas as pd
import psycopg2
import os
import time

# ── 1. Wait for Postgres to be ready ──────────────────────────────────────
# Containers start fast but Postgres needs a few seconds to be truly ready.
# We retry the connection up to 10 times before giving up.
def get_connection():
    for attempt in range(10):
        try:
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                dbname=os.environ["POSTGRES_DB"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                port=5432
            )
            print("Connected to Postgres!")
            return conn
        except psycopg2.OperationalError as e:
            print(f"Attempt {attempt + 1}/10 failed. Retrying in 3 seconds...")
            time.sleep(3)
    raise Exception("Could not connect to Postgres after 10 attempts.")


# ── 2. Load the CSV ────────────────────────────────────────────────────────
def load_csv():
    path = "/data/diabetic_data.csv"
    print(f"Reading CSV from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ── 3. Insert rows into Postgres ───────────────────────────────────────────
def insert_data(conn, df):
    cursor = conn.cursor()

    # Clear existing data so re-running doesn't duplicate rows
    cursor.execute("TRUNCATE TABLE patients_raw;")

    # Only insert columns that exist in our table
    columns = [
        "encounter_id", "patient_nbr", "race", "gender", "age",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "insulin", "readmitted"
    ]

    # Replace NaN with None so Postgres stores NULL correctly
    df = df[columns].where(pd.notnull(df[columns]), None)

    inserted = 0
    for row in df.itertuples(index=False):
        cursor.execute(
            f"""
            INSERT INTO patients_raw ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            """,
            list(row)
        )
        inserted += 1
        if inserted % 10000 == 0:
            print(f"  Inserted {inserted} rows...")
            conn.commit()  # commit in batches to avoid huge transactions

    conn.commit()
    print(f"Done! Inserted {inserted} rows into patients_raw.")
    cursor.close()


# ── 4. Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    conn = get_connection()
    df = load_csv()
    insert_data(conn, df)
    conn.close()