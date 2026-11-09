import streamlit as st
import pandas as pd
import joblib

# Load trained models
@st.cache_resource
def load_models():
    clf = joblib.load("best_aqi_classifier.pkl")
    reg = joblib.load("best_aqi_regressor.pkl")
    return clf, reg

try:
    clf, reg = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Delhi AQI Prediction",
    page_icon="🌍",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
div.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌍 Delhi Air Quality Prediction Dashboard")
st.subheader("Machine Learning Based Next-Day AQI Forecast")
st.markdown(
    "Enter pollutant concentrations to predict the next-day "
    "Air Quality Index and its category."
)

st.divider()

# Input fields
st.header("🧪 Enter Pollutant Concentrations")

c1, c2 = st.columns(2)

with c1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=60.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=100.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=30.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0)

with c2:
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, value=20.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=30.0)

# Exact feature names and order used during training
input_data = pd.DataFrame([{
    'PM2.5 (µg/m³)': pm25,
    'PM10 (µg/m³)': pm10,
    'NO2 (µg/m³)': no2,
    'SO2 (µg/m³)': so2,
    'NH3 (µg/m³)': nh3,
    'CO (mg/m³)': co,
    'Ozone (µg/m³)': ozone
}])

st.divider()

# Prediction
if st.button("🔍 Predict Next-Day AQI"):

    try:
        predicted_aqi = float(reg.predict(input_data)[0])
        predicted_category = clf.predict(input_data)[0]

        predicted_aqi = max(0, predicted_aqi)

        st.header("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Next-Day AQI",
                f"{predicted_aqi:.2f}"
            )

        with col2:
            st.metric(
                "Predicted AQI Category",
                str(predicted_category)
            )

        st.success("Prediction generated successfully!")

        st.caption(
            "Predictions are based on the trained Delhi AQI "
            "machine learning models. They are estimates, not "
            "official CPCB forecasts."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.markdown(
    "### About This Project\n"
    "Delhi AQI ML Bake-off | Dataset: 2023–2024\n\n"
    "Best Regression Model: Linear Regression (R² = 0.539)\n\n"
    "Best Classification Model: Logistic Regression "
    "(Macro F1 = 0.449)"
)
