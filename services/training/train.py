import os
import time
import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. WAIT FOR POSTGRES TO BE READY
# ─────────────────────────────────────────────
# Services start at different speeds. We loop until Postgres is up.
def wait_for_postgres():
    print("Waiting for Postgres...")
    while True:
        try:
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                dbname=os.environ["POSTGRES_DB"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
            )
            conn.close()
            print("Postgres is ready!")
            return
        except Exception:
            print("  Postgres not ready yet, retrying in 3s...")
            time.sleep(3)

# ─────────────────────────────────────────────
# 2. LOAD FEATURES FROM POSTGRES
# ─────────────────────────────────────────────
def load_features():
    print("Loading features from Postgres...")
    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    df = pd.read_sql("SELECT * FROM patient_features", conn)
    conn.close()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ─────────────────────────────────────────────
# 3. TRAIN AND EVALUATE ONE MODEL
# ─────────────────────────────────────────────
def train_and_log(model, model_name, X_train, X_test, y_train, y_test, params):
    """Train a model, evaluate it, and log everything to MLflow."""
    
    with mlflow.start_run(run_name=model_name):
        # Log the hyperparameters we chose
        mlflow.log_params(params)

        # Train the model
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)

        # Get probability predictions (not just 0/1 labels)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Calculate metrics
        auc   = roc_auc_score(y_test, y_prob)
        f1    = f1_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred)
        rec   = recall_score(y_test, y_pred)

        # Log metrics to MLflow — these appear in the UI
        mlflow.log_metrics({"auc": auc, "f1": f1, "precision": prec, "recall": rec})

        # Save the model artifact to MinIO via MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"    AUC={auc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
        return auc, mlflow.active_run().info.run_id

# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    wait_for_postgres()

    # Give MLflow server a moment to be ready
    time.sleep(5)

    # Tell MLflow where its server is
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("readmission-prediction")

    # Load data
    df = load_features()

    # Separate features (X) from the target label (y)
    target_col = "readmitted_30days"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"Imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")

    # Stratified split — keeps the same class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

    # ── Model 1: Logistic Regression ──
    # Good baseline. class_weight='balanced' compensates for imbalance.
    lr_params = {"model_type": "logistic_regression", "C": 1.0, "class_weight": "balanced"}
    lr_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    lr_auc, lr_run_id = train_and_log(lr_model, "LogisticRegression", X_train, X_test, y_train, y_test, lr_params)

    # ── Model 2: Random Forest ──
    # Ensemble of decision trees. More powerful than LR.
    rf_params = {"model_type": "random_forest", "n_estimators": 100, "class_weight": "balanced"}
    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_auc, rf_run_id = train_and_log(rf_model, "RandomForest", X_train, X_test, y_train, y_test, rf_params)

    # ── Model 3: XGBoost ──
    # Gradient boosting — often the best performer on tabular data.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos  # Tells XGBoost how imbalanced the data is
    xgb_params = {"model_type": "xgboost", "n_estimators": 100, "scale_pos_weight": float(scale)}
    xgb_model = XGBClassifier(n_estimators=100, scale_pos_weight=scale, random_state=42,
                               eval_metric="auc", verbosity=0)
    xgb_auc, xgb_run_id = train_and_log(xgb_model, "XGBoost", X_train, X_test, y_train, y_test, xgb_params)

    # ── Register the best model ──
    best = max([(lr_auc, lr_run_id, "LogisticRegression"),
                (rf_auc, rf_run_id, "RandomForest"),
                (xgb_auc, xgb_run_id, "XGBoost")], key=lambda x: x[0])

    best_auc, best_run_id, best_name = best
    print(f"\n🏆 Best model: {best_name} with AUC={best_auc:.4f}")

    # Register the best model so the API can find it by name
    model_uri = f"runs:/{best_run_id}/model"
    model_name = os.environ.get("MODEL_NAME", "readmission-model")
    
    client = MlflowClient()
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Promote it to "Production" stage
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"✅ Model '{model_name}' v{registered.version} promoted to Production!")

if __name__ == "__main__":
    main()