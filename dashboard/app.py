import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 AQI Prediction Dashboard")
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
### Current Inputs

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

    st.success(f"### Predicted AQI: {prediction:.2f}")

    # AQI Status
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