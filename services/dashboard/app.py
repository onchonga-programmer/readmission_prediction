"""
Stage 7 - Streamlit Dashboard
Connects to your FastAPI (Stage 5) and shows predictions visually.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# CONFIGURATION
# The API URL uses the Docker service name "api" as the hostname.
# Inside Docker's network, containers find each other by service name.
# When running locally (outside Docker), you'd use localhost:8000 instead.
# ─────────────────────────────────────────────
API_URL = "http://api:8000"

# ─────────────────────────────────────────────
# PAGE SETUP
# st.set_page_config must be the FIRST Streamlit call in your script.
# It controls the browser tab title and page layout.
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",   # "wide" uses the full browser width
)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# These are plain Python functions — nothing Streamlit-specific.
# They call your FastAPI using the requests library.
# ─────────────────────────────────────────────

def check_api_health():
    """
    Calls GET /health on your FastAPI.
    Returns True if the API is up, False if it's unreachable.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_model_info():
    """
    Calls GET /model-info to get the current model version and metrics.
    Returns a dict, or None if the call fails.
    """
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def predict_single(patient_data: dict):
    """
    Calls POST /predict with a single patient record.
    patient_data is a Python dict matching your Pydantic schema from schema.py.
    Returns the API response as a dict, or None on failure.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,   # requests automatically sets Content-Type: application/json
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
    """
    Sends each row of a DataFrame to the API one at a time.
    Returns a list of prediction results.
    
    A real production system would use a POST /predict/batch endpoint,
    but this approach keeps things simple and easy to understand.
    """
    results = []
    # st.progress shows a progress bar in the UI — Streamlit updates it in real time
    progress_bar = st.progress(0)
    for i, (_, row) in enumerate(df.iterrows()):
        result = predict_single(row.to_dict())
        results.append(result)
        progress_bar.progress((i + 1) / len(df))
    return results

# ─────────────────────────────────────────────
# RISK DISPLAY HELPER
# Converts a 0–1 probability into a colour-coded visual.
# ─────────────────────────────────────────────

def display_risk_gauge(probability: float):
    """
    Draws a speedometer-style gauge using Plotly.
    This is a great example of using Plotly inside Streamlit — 
    you build a Plotly figure, then pass it to st.plotly_chart().
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,           # convert 0–1 to 0–100 for display
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Readmission Risk %", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},   # green = low risk
                {"range": [30, 60], "color": "#f39c12"},  # orange = medium risk
                {"range": [60, 100], "color": "#e74c3c"}, # red = high risk
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
# st.sidebar.* puts widgets in the left panel.
# This is where users navigate between pages.
# ─────────────────────────────────────────────

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Performance"],
)

# API health check shown in the sidebar so it's always visible
st.sidebar.divider()
st.sidebar.subheader("API Status")
if check_api_health():
    st.sidebar.success("✅ API is online")
else:
    st.sidebar.error("❌ API is offline")

# ─────────────────────────────────────────────
# PAGE 1: SINGLE PATIENT PREDICTION
# Users fill in a form and get a prediction for one patient.
# ─────────────────────────────────────────────

if page == "🔍 Single Prediction":
    st.title("🔍 Single Patient Readmission Prediction")
    st.write("Fill in the patient details below and click **Predict** to get a readmission risk score.")

    # st.form groups inputs together so the API call only happens when
    # the user clicks Submit — not on every keystroke.
    with st.form("patient_form"):
        st.subheader("Patient Details")

        # st.columns splits the page into side-by-side columns
        col1, col2, col3 = st.columns(3)

        with col1:
            time_in_hospital = st.number_input(
                "Days in Hospital", min_value=1, max_value=30, value=5,
                help="Number of days the patient stayed in hospital"
            )
            num_medications = st.number_input(
                "Number of Medications", min_value=1, max_value=81, value=12
            )
            number_inpatient = st.number_input(
                "Prior Inpatient Visits", min_value=0, max_value=21, value=1
            )

        with col2:
            age_numeric = st.number_input(
                "Age", min_value=0, max_value=100, value=55
            )
            num_lab_procedures = st.number_input(
                "Lab Procedures", min_value=0, max_value=132, value=44
            )
            number_outpatient = st.number_input(
                "Prior Outpatient Visits", min_value=0, max_value=42, value=0
            )

        with col3:
            insulin_encoded = st.selectbox(
                "Insulin", options=[0, 1, 2, 3],
                format_func=lambda x: ["No", "Steady", "Up", "Down"][x],
                help="0=No insulin, 1=Steady dose, 2=Increased, 3=Decreased"
            )
            num_procedures = st.number_input(
                "Procedures During Stay", min_value=0, max_value=6, value=1
            )
            number_diagnoses = st.number_input(
                "Number of Diagnoses", min_value=1, max_value=16, value=8
            )

        # st.form_submit_button only works inside st.form
        # It returns True when the user clicks it
        submitted = st.form_submit_button("🔮 Predict Readmission Risk", use_container_width=True)

    # This block runs only when the form is submitted
    if submitted:
        # Build the dict that matches your FastAPI Pydantic schema
        patient_data = {
            "time_in_hospital": time_in_hospital,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "age_numeric": age_numeric,
            "num_lab_procedures": num_lab_procedures,
            "number_outpatient": number_outpatient,
            "insulin_encoded": insulin_encoded,
            "num_procedures": num_procedures,
            "number_diagnoses": number_diagnoses,
        }

        # st.spinner shows a loading animation while the API call runs
        with st.spinner("Getting prediction from model..."):
            result = predict_single(patient_data)

        if result:
            st.divider()
            st.subheader("Prediction Result")

            prob = result.get("readmission_probability", 0)
            risk_level = result.get("risk_level", "Unknown")
            model_version = result.get("model_version", "?")

            # Show the gauge chart
            display_risk_gauge(prob)

            # Show risk level with colour coding using st.metric
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Readmission Probability", f"{prob:.1%}")
            with col_b:
                # Colour the risk level text based on severity
                colour = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_level, "gray")
                st.markdown(f"**Risk Level:** :{colour}[{risk_level}]")
            with col_c:
                st.metric("Model Version", f"v{model_version}")

            # Show a clinical interpretation message
            if risk_level == "High":
                st.error("⚠️ **High Risk**: This patient has a high probability of readmission within 30 days. Consider a follow-up care plan.")
            elif risk_level == "Medium":
                st.warning("🟡 **Medium Risk**: Monitor this patient closely after discharge.")
            else:
                st.success("✅ **Low Risk**: This patient is unlikely to be readmitted within 30 days.")

# ─────────────────────────────────────────────
# PAGE 2: BATCH PREDICTION
# Users upload a CSV and get predictions for every row.
# ─────────────────────────────────────────────

elif page == "📊 Batch Prediction":
    st.title("📊 Batch Patient Prediction")
    st.write("Upload a CSV file with patient records to get predictions for all of them at once.")

    # st.file_uploader lets users drag-and-drop or browse for a file
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded_file is not None:
        # pandas reads the uploaded file directly — st.file_uploader returns a file-like object
        df = pd.read_csv(uploaded_file)
        st.write(f"📋 Loaded **{len(df)} patients** from file.")
        st.dataframe(df.head())  # Show first 5 rows as a preview

        if st.button("🚀 Run Batch Predictions"):
            with st.spinner(f"Running predictions for {len(df)} patients..."):
                results = predict_batch(df)

            # Add predictions as a new column on the original DataFrame
            df["readmission_probability"] = [
                r.get("readmission_probability", None) if r else None for r in results
            ]
            df["risk_level"] = [
                r.get("risk_level", None) if r else None for r in results
            ]

            st.success("✅ Predictions complete!")
            st.dataframe(df)

            # Risk distribution chart — how many High / Medium / Low patients?
            if "risk_level" in df.columns:
                st.subheader("Risk Level Distribution")
                risk_counts = df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                colour_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    color="Risk Level",
                    color_discrete_map=colour_map,
                    title="Patient Risk Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

            # st.download_button lets users save the results as a CSV
            csv_output = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_output,
                file_name="readmission_predictions.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# Fetches model metrics from your FastAPI and displays them.
# ─────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("Live metrics from the currently deployed model, fetched from MLflow via the API.")

    info = get_model_info()

    if info:
        st.subheader(f"Model: {info.get('model_name', 'Unknown')} — Version {info.get('model_version', '?')}")

        metrics = info.get("metrics", {})

        if metrics:
            # Display key metrics as a row of metric boxes
            cols = st.columns(len(metrics))
            for col, (metric_name, metric_value) in zip(cols, metrics.items()):
                with col:
                    st.metric(
                        label=metric_name.upper().replace("_", " "),
                        value=f"{metric_value:.3f}" if isinstance(metric_value, float) else metric_value,
                    )

            # AUC bar chart — compare metric values visually
            st.subheader("Metrics Overview")
            metric_df = pd.DataFrame(
                list(metrics.items()), columns=["Metric", "Value"]
            )
            fig = px.bar(
                metric_df,
                x="Metric",
                y="Value",
                title="Model Evaluation Metrics",
                color="Value",
                color_continuous_scale="blues",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics found for this model. Check MLflow to see if metrics were logged during training.")
    else:
        st.error("Could not fetch model info from the API. Make sure the API service is running.")
        st.code(f"Expected endpoint: GET {API_URL}/model-info")