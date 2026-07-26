import pandas as pd
from pathlib import Path

# Load existing data
input_file = Path("data/raw/aqi_weather.csv")

df = pd.read_csv(input_file)

# Duplicate data to simulate historical records
historical_df = pd.concat([df] * 100, ignore_index=True)

# Save training dataset
output_file = Path("data/processed/training_data.csv")
historical_df.to_csv(output_file, index=False)

print("✅ Historical dataset created!")
print("Total Rows:", len(historical_df))