import pandas as pd
from pathlib import Path

# Read raw dataset
input_file = Path("data/raw/aqi_weather.csv")

df = pd.read_csv(input_file)

print("✅ Raw Data Loaded")
print(df.head())

# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

# Create time-based features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["day_of_week"] = df["datetime"].dt.day_name()

# Weekend feature
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
# AQI Category
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

df["aqi_category"] = df["aqi"].apply(get_aqi_category)

# Temperature-Humidity Index
df["temp_humidity_index"] = (
    df["temperature"] * 0.8 +
    df["humidity"] * 0.2
)

# AQI Change Rate
df["aqi_change"] = df["aqi"].diff().fillna(0)

# Create processed folder if it doesn't exist
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Save processed data
output_file = output_folder / "processed_aqi.csv"
print("\nProcessed Dataset Preview:")
print(df.head())

df.to_csv(output_file, index=False)

print("\n✅ Feature Engineering Completed")
print("Processed file saved at:", output_file)