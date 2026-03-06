import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import time

time.sleep(5)

DB_USER = os.environ["POSTGRES_USER"]
DB_PASS = os.environ["POSTGRES_PASSWORD"]
DB_NAME = os.environ["POSTGRES_DB"]
DB_HOST = "postgres"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}")
print("✅ Connected to Postgres")

print("📖 Reading raw data...")
df = pd.read_sql("SELECT * FROM patients_raw", engine)
print(f"   Loaded {len(df)} rows")

df.replace("?", np.nan, inplace=True)
df.drop(columns=["encounter_id", "patient_nbr", "payer_code",
                  "medical_specialty"], inplace=True)
df.dropna(subset=["gender"], inplace=True)
print("✅ Missing values handled")

age_map = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}
df["age_numeric"] = df["age"].map(age_map)
df["gender_numeric"] = df["gender"].map({"Male": 0, "Female": 1})
df["race"].fillna("Caucasian", inplace=True)
race_map = {"Caucasian": 0, "AfricanAmerican": 1,
            "Hispanic": 2, "Asian": 3, "Other": 4}
df["race_encoded"] = df["race"].map(race_map).fillna(4)
med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
df["insulin_encoded"]   = df["insulin"].map(med_map).fillna(0)
df["metformin_encoded"] = df["metformin"].map(med_map).fillna(0)
df["change_encoded"]       = df["change_col"].map({"No": 0, "Ch": 1}).fillna(0)
df["diabetes_med_encoded"] = df["diabetesmed"].map({"No": 0, "Yes": 1}).fillna(0)
print("✅ Categorical columns encoded")

df["total_meds_changed"] = (
    (df["insulin_encoded"] > 1).astype(int) +
    (df["metformin_encoded"] > 1).astype(int)
)
df["total_visits"] = (
    df["number_outpatient"] +
    df["number_emergency"] +
    df["number_inpatient"]
)
df["readmitted_30days"] = df["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})

positive = df["readmitted_30days"].sum()
total = len(df)
print(f"📊 Class imbalance: {positive} readmitted ({positive/total*100:.1f}%) "
      f"vs {total-positive} not ({(total-positive)/total*100:.1f}%)")

feature_columns = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_numeric",
    "gender_numeric", "race_encoded", "insulin_encoded",
    "metformin_encoded", "change_encoded", "diabetes_med_encoded",
    "total_meds_changed", "total_visits", "readmitted_30days"
]

features_df = df[feature_columns].dropna()
print(f"💾 Writing {len(features_df)} rows to patient_features table...")
features_df.to_sql("patient_features", engine, if_exists="replace", index=False)
print("✅ Feature engineering complete!")