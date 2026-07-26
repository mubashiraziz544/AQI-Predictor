import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="AQI Predictor", page_icon="🌍")

st.title("🌍 AQI Prediction Dashboard")

model = joblib.load("models/random_forest.pkl")

temperature = st.number_input("Temperature (°C)", value=36.5)
humidity = st.number_input("Humidity (%)", value=35)
pressure = st.number_input("Pressure (hPa)", value=1002)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.5)

if st.button("Predict AQI"):

    sample = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed
    }])

    prediction = model.predict(sample)[0]

    st.success(f"Predicted AQI: {prediction:.2f}")

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