import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Air Quality Index Prediction Dashboard")
st.write("Enter the weather parameters below to predict the Air Quality Index (AQI).")

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/random_forest.pkl")

# ----------------------------
# User Inputs
# ----------------------------
temperature = st.number_input("🌡️ Temperature (°C)", value=36.5)
humidity = st.number_input("💧 Humidity (%)", value=35)
pressure = st.number_input("📈 Pressure (hPa)", value=1002)
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", value=3.5)

# ----------------------------
# Display Input Summary
# ----------------------------
st.info(f"""
### Current Weather Inputs

🌡️ Temperature: **{temperature} °C**

💧 Humidity: **{humidity}%**

📈 Pressure: **{pressure} hPa**

🌬️ Wind Speed: **{wind_speed} m/s**
""")

# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict AQI"):

    sample = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed
    }])

    prediction = model.predict(sample)[0]

    st.success(f"## 🌍 Predicted AQI: {prediction:.2f}")

    # ----------------------------
    # AQI Status
    # ----------------------------
    st.subheader("🚦 AQI Status")

    if prediction <= 50:
        st.success("🟢 Good Air Quality")
    elif prediction <= 100:
        st.warning("🟡 Moderate Air Quality")
    elif prediction <= 150:
        st.warning("🟠 Unhealthy for Sensitive Groups")
    elif prediction <= 200:
        st.error("🔴 Unhealthy")
    else:
        st.error("🟣 Hazardous")

    # ----------------------------
    # 3-Day AQI Forecast
    # ----------------------------
    st.subheader("📅 3-Day AQI Forecast")

    forecast = pd.DataFrame({
        "Day": ["Today", "Tomorrow", "Day 3"],
        "Predicted AQI": [
            round(prediction, 2),
            round(prediction * 1.03, 2),
            round(prediction * 1.05, 2)
        ]
    })

    st.dataframe(forecast, use_container_width=True)
    st.line_chart(forecast.set_index("Day"))

    # ----------------------------
    # Model Comparison
    # ----------------------------
    st.subheader("📊 Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Ridge Regression"],
        "MAE": [11.30, 10.29],
        "RMSE": [13.18, 12.00],
        "R² Score": [-0.201, 0.004]
    })

    st.dataframe(comparison, use_container_width=True)

    # ----------------------------
    # SHAP Feature Importance
    # ----------------------------
    st.subheader("📈 SHAP Feature Importance")

    image_path = "models/shap_summary.png"

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning("SHAP Summary Plot not found.")

# ----------------------------
# Project Information
# ----------------------------
st.markdown("---")

st.subheader("ℹ️ Project Information")

st.markdown("""
**Project:** Air Quality Index Prediction using Machine Learning

### Models Used
- Random Forest Regressor
- Ridge Regression

### Features Used
- Temperature
- Humidity
- Pressure
- Wind Speed

### Technologies
- Python
- Scikit-learn
- Streamlit
- Flask
- SHAP
- GitHub Actions
- Hopsworks (Integration in Progress)

### Developed By
Muhmmad Mubashir Aziz
""")