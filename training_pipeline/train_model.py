import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
# Load dataset
df = pd.read_csv("data/processed/training_data.csv")

# Features
X = df[["temperature", "humidity", "pressure", "wind_speed"]]

# Target
y = df["aqi"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== Random Forest Performance =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# Save model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/random_forest.pkl")

print("\n✅ Model saved successfully!")

# ==========================
# Ridge Regression Model
# ==========================

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge_predictions = ridge.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = mean_squared_error(y_test, ridge_predictions) ** 0.5
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\n===== Ridge Regression Performance =====")
print("MAE :", ridge_mae)
print("RMSE:", ridge_rmse)
print("R2  :", ridge_r2)

os.makedirs("models", exist_ok=True)
joblib.dump(ridge, "models/ridge_model.pkl")

print("✅ Ridge model saved successfully!")