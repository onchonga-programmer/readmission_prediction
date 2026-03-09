# 🏥 Patient Readmission Risk Predictor

A production-grade, end-to-end machine learning system that predicts whether a hospital patient will be readmitted within 30 days of discharge. Built as a fully containerised pipeline using Docker Compose — spin up the entire system with a single command.

```bash
docker compose up
```

---

## 🎯 The Business Problem

Hospital readmissions within 30 days are one of the most costly and preventable problems in healthcare:

- In the United States alone, hospitals face **over $26 billion** in costs annually from unplanned readmissions
- Medicare penalises hospitals with high readmission rates, directly impacting revenue
- Early identification of at-risk patients allows care teams to intervene **before** discharge — scheduling follow-ups, adjusting medication, or arranging home care

This system gives hospitals a tool to flag high-risk patients at the point of discharge, enabling proactive intervention rather than reactive treatment.

---

## 📊 The Dataset

This project uses the **UCI Diabetes 130-US Hospitals dataset** — a real-world clinical dataset covering 10 years of diabetes patient records across 130 US hospitals.

| Property | Detail |
|---|---|
| Rows | ~101,766 patient encounters |
| Features | 50 clinical variables |
| Target | Readmitted within 30 days (Yes / No) |
| Class imbalance | ~11% positive (readmitted within 30 days) |

Key features include: time in hospital, number of medications, number of prior inpatient visits, age, diagnosis codes, and medication changes.

---

## 🏗️ System Architecture

The system is built as a pipeline of 8 independent Docker services, each with a single responsibility. Data flows through the pipeline from raw CSV to a live prediction API.

```
Raw CSV File
     ↓
MinIO (object storage — local S3)
     ↓
Ingestion Service (Python — reads CSV, writes to Postgres)
     ↓
Postgres (cleaned + structured data)
     ↓
Feature Engineering Service (Python — builds ML features)
     ↓
Training Service (scikit-learn + logs to MLflow)
     ↓
MLflow Server (experiment tracking + model registry)
     ↓
FastAPI Prediction API (serves live predictions)
     ↓
Streamlit Dashboard (visual interface for clinicians)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Containerisation | Docker + Docker Compose | Runs every service in isolation |
| Database | PostgreSQL | Stores raw patient data and engineered features |
| Object Storage | MinIO (S3-compatible) | Stores trained model artifacts |
| ML Training | scikit-learn, XGBoost | Logistic Regression, Random Forest, XGBoost |
| Experiment Tracking | MLflow | Logs every training run, metrics, and models |
| Prediction API | FastAPI | Serves real-time predictions over REST |
| Dashboard | Streamlit | Visual interface for clinicians |

---

## 📁 Project Structure

```
readmission-predictor/
│
├── docker-compose.yml          ← Wires all 8 services together
├── .env                        ← Secrets (never committed to Git)
├── .env.example                ← Template for other developers
├── .gitignore                  ← Excludes .env, data, and model files
├── init.sql                    ← Creates Postgres tables on first startup
├── README.md                   ← This file
│
├── data/
│   └── diabetic_data.csv       ← Raw UCI dataset
│
└── services/
    ├── ingestion/              ← Loads CSV into Postgres
    ├── feature_engineering/    ← Builds ML-ready features
    ├── training/               ← Trains models, logs to MLflow
    ├── api/                    ← FastAPI prediction endpoint
    └── dashboard/              ← Streamlit web UI
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least 4GB of free RAM allocated to Docker
- The dataset downloaded to `data/diabetic_data.csv` (available from [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008))

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/onchonga-programmer/readmission-predictor.git
cd readmission-predictor
```

**2. Create your environment file**
```bash
cp .env.example .env
```
Then open `.env` and fill in your credentials (see Environment Variables section below).

**3. Start everything**
```bash
docker compose up
```

Docker will build all images and start all services in the correct order. The first run takes 5–10 minutes. Subsequent runs are much faster.

**4. Open the interfaces**

| Interface | URL | Purpose |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | Main user interface |
| FastAPI Docs | http://localhost:8000/docs | Interactive API testing |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| MinIO Console | http://localhost:9001 | Browse stored model files |

---

## 🔬 Machine Learning Details

### Models Trained

Three models are trained and compared in every run:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline — interpretable and fast |
| Random Forest | Good balance of performance and explainability |
| XGBoost | Typically highest AUC — the production model |

### Handling Class Imbalance

Only ~11% of patients are readmitted within 30 days. Training naively on this data produces a model that almost always predicts "not readmitted" and still achieves 89% accuracy — which is useless clinically.

This project addresses imbalance using `class_weight='balanced'` during training, and evaluates models using **AUC-ROC** and **F1 score** rather than accuracy.

### Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| AUC-ROC | Measures model discrimination across all thresholds |
| F1 Score | Balances precision and recall for the minority class |
| Recall | Critical in healthcare — missing a high-risk patient (false negative) is more costly than a false alarm |

---

## 🔌 API Reference

### POST /predict

Accepts patient features and returns a readmission risk score.

**Request body example:**
```json
{
  "time_in_hospital": 5,
  "num_medications": 12,
  "number_inpatient": 2,
  "age_numeric": 55,
  "insulin_encoded": 1
}
```

**Response:**
```json
{
  "readmission_probability": 0.74,
  "risk_level": "High",
  "model_version": "3"
}
```

### GET /health
Returns service status. Used by Docker health checks.

### GET /model-info
Returns the currently loaded model version and its training metrics.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in these values. **Never commit `.env` to Git.**

| Variable | Description |
|---|---|
| `POSTGRES_USER` | Your chosen Postgres username |
| `POSTGRES_PASSWORD` | Your chosen Postgres password |
| `POSTGRES_DB` | Database name — use `readmission` |
| `MINIO_USER` | MinIO root username |
| `MINIO_PASSWORD` | MinIO root password (min 8 characters) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` (internal Docker hostname) |
| `MODEL_NAME` | Name used to register the model in MLflow |

---

## 🔧 Useful Commands

```bash
# Start all services
docker compose up

# Start in background (detached mode)
docker compose up -d

# View logs for a specific service
docker compose logs -f api

# Fully reset everything (wipes all data and volumes)
docker compose down -v

# Rebuild a specific service after code changes
docker compose build training
docker compose up training

# Connect directly to Postgres
docker exec -it readmission_db psql -U admin -d readmission

# Verify data was loaded
docker exec -it readmission_db psql -U admin -d readmission -c "SELECT COUNT(*) FROM patients_raw;"
```

---

## 📈 Service Startup Order

Docker Compose starts services in this order, using health checks to ensure each layer is ready before the next begins:

```
postgres + minio → mlflow → ingestion → feature_engineering → training → api → dashboard
```

One-shot services (ingestion, feature_engineering, training) run once and exit. The API and dashboard run continuously.

---

## 💡 What I Learned Building This

**Docker & Infrastructure**
- Building multi-container systems with Docker Compose
- Inter-service communication over Docker networks (services reference each other by name)
- Named volumes, health checks, and service dependency ordering
- Managing secrets with environment variables and `.env` files

**Data Engineering**
- Designing a layered data pipeline: raw → cleaned → features → model
- Working with S3-compatible object storage (MinIO)
- Writing modular services where each component has a single responsibility

**Machine Learning Engineering**
- End-to-end ML pipeline from messy CSV to deployed REST API
- Handling class imbalance in medical datasets
- Experiment tracking and model versioning with MLflow
- Serving ML models with FastAPI and validating inputs with Pydantic

---

## 🗺️ Potential Improvements

- [ ] Add Apache Airflow for proper pipeline orchestration and scheduling
- [ ] Add model monitoring to detect prediction drift over time
- [ ] Add SHAP values to the API response to explain individual predictions
- [ ] Write unit tests for the feature engineering and API services
- [ ] Set up a CI/CD pipeline with GitHub Actions to rebuild and test on every push
- [ ] Deploy to a cloud provider (AWS EC2 or DigitalOcean) for public access

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

