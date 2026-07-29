# Air Quality Index (AQI) Prediction System

An end-to-end Machine Learning project that predicts the **Air Quality Index (AQI)** using weather parameters. The project includes data collection, feature engineering, model training, explainability with SHAP, Flask API, Streamlit dashboard, and automated CI/CD using GitHub Actions.

---

# Project Features

- 🌦️ Fetch weather and AQI data
- ⚙️ Feature Engineering Pipeline
- 📊 Historical Data Backfill
- 🤖 Random Forest Regression Model
- 📈 Ridge Regression Model
- 🔍 SHAP Explainability
- 🌐 Flask Prediction API
- 📱 Streamlit Dashboard
- 🔄 Automated GitHub Actions Pipeline
- 📅 3-Day AQI Forecast
- 📊 Model Performance Comparison

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Flask
- Streamlit
- GitHub Actions
- Joblib
- Matplotlib

---

# Project Structure

```text
AQI-Predictor
│
├── api/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── feature_pipeline/
├── models/
├── training_pipeline/
├── utils/
├── .github/
│   └── workflows/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Models

| Model | Status |
|--------|--------|
| Random Forest | ✅ |
| Ridge Regression | ✅ |

---

# Model Performance

| Model | MAE | RMSE | R² Score |
|--------|------:|------:|------:|
| Random Forest | 11.30 | 13.18 | -0.201 |
| Ridge Regression | 10.29 | 12.00 | 0.004 |

---

# Dashboard Features

- AQI Prediction
- AQI Status
- Current Weather Inputs
- 3-Day AQI Forecast
- Model Comparison
- SHAP Feature Importance

---

# CI/CD Pipeline

GitHub Actions automatically:

- Installs dependencies
- Runs feature pipeline
- Generates historical data
- Trains machine learning model
- Generates SHAP explanation

The workflow also supports scheduled execution.

---

# Run the Project

Clone the repository:

```bash
git clone https://github.com/mubashiraziz544/AQI-Predictor.git
```

Go to the project folder:

```bash
cd AQI-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask API:

```bash
python api/app.py
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

---

# Future Improvements

- Hopsworks Feature Store Integration
- TensorFlow/PyTorch Model
- Model Registry
- Cloud Deployment
- Live AQI API Integration

---

# Author

**Mubashir Aziz**

Bachelor of Software Engineering

CGPA: 3.63

GitHub:

https://github.com/mubashiraziz544

---

# ⭐ Project Highlights

✅ End-to-End Machine Learning Pipeline

✅ Feature Engineering

✅ Model Explainability (SHAP)

✅ Streamlit Dashboard

✅ Flask REST API

✅ Automated GitHub Actions CI/CD

✅ Multiple Machine Learning Models

---

If you find this project useful, don't forget to ⭐ the repository.