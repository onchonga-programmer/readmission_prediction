"""
main.py — The FastAPI Prediction API

How this file is structured:
1. Imports and configuration
2. App startup: load the model from MLflow ONCE (not on every request)
3. Routes (endpoints):
   - GET  /health      → Is the server alive?
   - GET  /model-info  → What model version is loaded?
   - POST /predict     → Accept patient data, return risk score

Why load the model on startup?
   Loading a model takes time (reading files, deserialising weights).
   If you loaded it on every prediction request your API would be very slow.
   FastAPI's lifespan pattern lets you run setup code once when the server starts.
"""

import os
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from models import PatientFeatures, PredictionResponse, HealthResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app_state = {
    "model": None,
    "model_version": "unknown",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up — loading model from MLflow...")

  
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name   = os.getenv("MODEL_NAME", "readmission_model")

    # Tell the MLflow client where your MLflow server lives
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Load the model that was promoted to "Production" in Stage 4.
        # "models:/<name>/Production" is MLflow's URI syntax for the registry.
        model_uri = f"models:/{model_name}/Production"
        logger.info(f"Loading model from: {model_uri}")

        app_state["model"] = mlflow.sklearn.load_model(model_uri)

        # Fetch the version number for the /model-info endpoint
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            app_state["model_version"] = versions[0].version
            logger.info(f"Model version {app_state['model_version']} loaded successfully.")
        else:
            logger.warning("No Production model found — predictions will fail.")

    except Exception as e:
        # We log the error but don't crash. The /health endpoint will report
        # model_loaded: false so callers know something is wrong.
        logger.error(f"Failed to load model: {e}")

    yield  # <── The app runs while execution is paused here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("Server shutting down — cleaning up.")
    app_state["model"] = None


# Create the FastAPI application
app = FastAPI(
    title="Patient Readmission Risk API",
    description="Predicts 30-day hospital readmission risk using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Returns server status. Docker health checks call this endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=app_state["model"] is not None,
    )


@app.get("/model-info")
def model_info():
    """Returns the currently loaded model version."""
    return {
        "model_name": os.getenv("MODEL_NAME", "readmission_model"),
        "model_version": app_state["model_version"],
        "model_loaded": app_state["model"] is not None,
    }




@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """
    Accepts patient feature data and returns a readmission risk score.

    The `patient` argument is automatically populated from the JSON request body
    by FastAPI. We don't need to manually call json.loads() or validate fields.
    """
    # Guard: if the model failed to load on startup, refuse requests cleanly
    if app_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for status.",
        )


    input_df = pd.DataFrame([patient.dict()])

    try:
        # predict_proba returns probabilities for each class: [[prob_class_0, prob_class_1]]
        # We want the probability of class 1 (readmitted), so we take index [0][1]
        prob = float(app_state["model"].predict_proba(input_df)[0][1])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Convert the raw probability to a human-readable risk label
    if prob >= 0.6:
        risk_level = "High"
    elif prob >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return PredictionResponse(
        readmission_probability=round(prob, 4),
        risk_level=risk_level,
        model_version=app_state["model_version"],
    )