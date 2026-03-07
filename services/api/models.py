from pydantic import BaseModel


class PatientFeatures(BaseModel):
    time_in_hospital: float
    num_lab_procedures: float
    num_procedures: float
    num_medications: float
    number_outpatient: float
    number_emergency: float
    number_inpatient: float
    number_diagnoses: float
    age_numeric: float
    gender_numeric: float
    race_encoded: float
    insulin_encoded: float
    metformin_encoded: float
    change_encoded: float
    diabetes_med_encoded: float
    total_meds_changed: float
    total_visits: float


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool