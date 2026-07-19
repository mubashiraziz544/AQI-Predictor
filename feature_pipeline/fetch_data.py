import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
CITY = os.getenv("CITY")

# -----------------------------
# OpenWeather API
# -----------------------------
weather_url = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
)

weather_response = requests.get(weather_url)

if weather_response.status_code == 200:
    weather = weather_response.json()

    print("\n===== Weather Data =====")
    print("City:", weather["name"])
    print("Temperature:", weather["main"]["temp"], "°C")
    print("Humidity:", weather["main"]["humidity"], "%")
    print("Pressure:", weather["main"]["pressure"], "hPa")
    print("Wind Speed:", weather["wind"]["speed"], "m/s")
else:
    print("Weather API Error:", weather_response.status_code)

# -----------------------------
# AQICN API
# -----------------------------
# -----------------------------
# AQICN API
# -----------------------------
aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_API_KEY}"

aqi_response = requests.get(aqi_url)

if aqi_response.status_code == 200:
    aqi = aqi_response.json()

    print("\n===== AQI Data =====")

    if aqi["status"] == "ok":
        print("AQI:", aqi["data"]["aqi"])
        print("Dominant Pollutant:", aqi["data"]["dominentpol"])

        # Create one record
        record = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": weather["name"],
            "temperature": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "pressure": weather["main"]["pressure"],
            "wind_speed": weather["wind"]["speed"],
            "aqi": aqi["data"]["aqi"],
            "dominant_pollutant": aqi["data"]["dominentpol"],
        }

        df = pd.DataFrame([record])

        output_path = Path("data/raw/aqi_weather.csv")

        if output_path.exists():
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            df.to_csv(output_path, index=False)

        print("\n✅ Data saved to:", output_path)

    else:
        print("AQICN Error:", aqi["data"])

else:
    print("AQICN HTTP Error:", aqi_response.status_code)