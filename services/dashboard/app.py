"""
Streamlit dashboard for patient readmission risk prediction."""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def predict_single(patient_data: dict):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the prediction API. Is it running?")
    return None

def predict_batch(df: pd.DataFrame):
    results = []
    progress_bar = st.progress(0)
    for i, (_, row) in enumerate(df.iterrows()):
        result = predict_single(row.to_dict())
        results.append(result)
        progress_bar.progress((i + 1) / len(df))
    return results

def display_risk_gauge(probability: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Readmission Risk %", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30],  "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f39c12"},
                {"range": [60, 100],"color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Performance"],
)

st.sidebar.divider()
st.sidebar.subheader("API Status")
if check_api_health():
    st.sidebar.success("✅ API is online")
else:
    st.sidebar.error("❌ API is offline")



if page == "🔍 Single Prediction":
    st.title("🔍 Single Patient Readmission Prediction")
    st.write("Fill in the patient details below and click **Predict** to get a readmission risk score.")

    with st.form("patient_form"):

        st.subheader("🏥 Hospital Stay")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=30, value=5)
        with col2:
            num_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=132, value=44)
        with col3:
            num_procedures = st.number_input("Procedures During Stay", min_value=0, max_value=6, value=1)

        st.subheader("💊 Medications")
        col4, col5 = st.columns(2)
        with col4:
            num_medications = st.number_input("Number of Medications", min_value=1, max_value=81, value=12)
            metformin_encoded = st.selectbox(
                "Metformin",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["No", "Steady", "Up", "Down"][x]
            )
            change_encoded = st.selectbox(
                "Medication Change Made?",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x]
            )
        with col5:
            insulin_encoded = st.selectbox(
                "Insulin",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["No", "Steady", "Up", "Down"][x]
            )
            diabetes_med_encoded = st.selectbox(
                "Diabetes Medication Prescribed?",
                options=[0, 1],
                format_func=lambda x: ["No", "Yes"][x]
            )
            total_meds_changed = st.number_input("Total Medications Changed", min_value=0, max_value=10, value=1)

        st.subheader("📋 Visit History")
        col6, col7, col8 = st.columns(3)
        with col6:
            number_outpatient = st.number_input("Prior Outpatient Visits", min_value=0, max_value=42, value=0)
        with col7:
            number_emergency = st.number_input("Prior Emergency Visits", min_value=0, max_value=76, value=0)
        with col8:
            number_inpatient = st.number_input("Prior Inpatient Visits", min_value=0, max_value=21, value=1)

        total_visits = number_outpatient + number_emergency + number_inpatient
        st.info(f"Total prior visits (auto-calculated): **{total_visits}**")

        st.subheader("🧍 Patient Demographics")
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=8)
        with col10:
            age_numeric = st.number_input("Age", min_value=0, max_value=100, value=55)
        with col11:
            gender_numeric = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: ["Female", "Male"][x]
            )
        with col12:
            race_encoded = st.selectbox(
                "Race",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"][x]
            )

        submitted = st.form_submit_button("🔮 Predict Readmission Risk", use_container_width=True)

    if submitted:
        # All 17 fields — must match PatientFeatures in models.py exactly
        patient_data = {
            "time_in_hospital":     float(time_in_hospital),
            "num_lab_procedures":   float(num_lab_procedures),
            "num_procedures":       float(num_procedures),
            "num_medications":      float(num_medications),
            "number_outpatient":    float(number_outpatient),
            "number_emergency":     float(number_emergency),
            "number_inpatient":     float(number_inpatient),
            "number_diagnoses":     float(number_diagnoses),
            "age_numeric":          float(age_numeric),
            "gender_numeric":       float(gender_numeric),
            "race_encoded":         float(race_encoded),
            "insulin_encoded":      float(insulin_encoded),
            "metformin_encoded":    float(metformin_encoded),
            "change_encoded":       float(change_encoded),
            "diabetes_med_encoded": float(diabetes_med_encoded),
            "total_meds_changed":   float(total_meds_changed),
            "total_visits":         float(total_visits),
        }

        with st.spinner("Getting prediction from model..."):
            result = predict_single(patient_data)

        if result:
            st.divider()
            st.subheader("Prediction Result")

            prob          = result.get("readmission_probability", 0)
            risk_level    = result.get("risk_level", "Unknown")
            model_version = result.get("model_version", "?")

            display_risk_gauge(prob)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Readmission Probability", f"{prob:.1%}")
            with col_b:
                colour = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_level, "gray")
                st.markdown(f"**Risk Level:** :{colour}[{risk_level}]")
            with col_c:
                st.metric("Model Version", f"v{model_version}")

            if risk_level == "High":
                st.error("⚠️ **High Risk**: This patient has a high probability of readmission within 30 days. Consider a follow-up care plan.")
            elif risk_level == "Medium":
                st.warning("🟡 **Medium Risk**: Monitor this patient closely after discharge.")
            else:
                st.success("✅ **Low Risk**: This patient is unlikely to be readmitted within 30 days.")


elif page == "📊 Batch Prediction":
    st.title("📊 Batch Patient Prediction")
    st.write("Upload a CSV file with patient records to get predictions for all of them at once.")
    st.info("Your CSV must have these 17 columns: time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses, age_numeric, gender_numeric, race_encoded, insulin_encoded, metformin_encoded, change_encoded, diabetes_med_encoded, total_meds_changed, total_visits")

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"📋 Loaded **{len(df)} patients** from file.")
        st.dataframe(df.head())

        if st.button("🚀 Run Batch Predictions"):
            with st.spinner(f"Running predictions for {len(df)} patients..."):
                results = predict_batch(df)

            df["readmission_probability"] = [
                r.get("readmission_probability", None) if r else None for r in results
            ]
            df["risk_level"] = [
                r.get("risk_level", None) if r else None for r in results
            ]

            st.success("✅ Predictions complete!")
            st.dataframe(df)

            if "risk_level" in df.columns:
                st.subheader("Risk Level Distribution")
                risk_counts = df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                colour_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                fig = px.bar(
                    risk_counts, x="Risk Level", y="Count",
                    color="Risk Level", color_discrete_map=colour_map,
                    title="Patient Risk Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

            csv_output = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_output,
                file_name="readmission_predictions.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("Live metrics from the currently deployed model, fetched from MLflow via the API.")

    info = get_model_info()

    if info:
        st.subheader(f"Model: {info.get('model_name', 'Unknown')} — Version {info.get('model_version', '?')}")

        metrics = info.get("metrics", {})

        if metrics:
            cols = st.columns(len(metrics))
            for col, (metric_name, metric_value) in zip(cols, metrics.items()):
                with col:
                    st.metric(
                        label=metric_name.upper().replace("_", " "),
                        value=f"{metric_value:.3f}" if isinstance(metric_value, float) else metric_value,
                    )

            st.subheader("Metrics Overview")
            metric_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            fig = px.bar(
                metric_df, x="Metric", y="Value",
                title="Model Evaluation Metrics",
                color="Value", color_continuous_scale="blues",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics found for this model. Check MLflow to see if metrics were logged during training.")
    else:
        st.error("Could not fetch model info from the API. Make sure the API service is running.")
        st.code(f"Expected endpoint: GET {API_URL}/model-info")