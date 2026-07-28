import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = os.getenv("HOPSWORKS_PROJECT")

project = hopsworks.login(
    project=PROJECT,
    api_key_value=API_KEY
)

fs = project.get_feature_store()

df = pd.read_csv("data/processed/training_data.csv")

feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["datetime"],
    description="AQI prediction features"
)

feature_group.insert(df)

print("✅ Data uploaded to Hopsworks Feature Store!")