import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/training_data.csv")

# Features
X = df[["temperature", "humidity", "pressure", "wind_speed"]]

# Load trained model
model = joblib.load("models/random_forest.pkl")

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(shap_values, X, show=False)

plt.savefig("models/shap_summary.png", dpi=300, bbox_inches="tight")

print("✅ SHAP summary plot saved!")python training_pipeline/explain_model.py