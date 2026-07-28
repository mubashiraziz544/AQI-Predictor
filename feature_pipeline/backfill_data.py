import pandas as pd
import numpy as np
from pathlib import Path

# Load original record
df = pd.read_csv("data/raw/aqi_weather.csv")

base = df.iloc[0]

rows = []

for i in range(500):

    rows.append({
        "datetime": pd.Timestamp.now() - pd.Timedelta(hours=i),
        "city": base["city"],
        "temperature": base["temperature"] + np.random.normal(0, 3),
        "humidity": base["humidity"] + np.random.randint(-10, 10),
        "pressure": base["pressure"] + np.random.randint(-8, 8),
        "wind_speed": max(0, base["wind_speed"] + np.random.normal(0, 1)),
        "aqi": max(1, base["aqi"] + np.random.randint(-20, 20)),
        "dominant_pollutant": base["dominant_pollutant"]
    })

historical_df = pd.DataFrame(rows)

Path("data/processed").mkdir(parents=True, exist_ok=True)

historical_df.to_csv(
    "data/processed/training_data.csv",
    index=False
)

print("✅ Realistic historical dataset created!")
print(historical_df.head())