from flask import Flask, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest.pkl")

@app.route("/")
def home():
    return "AQI Predictor API is Running!"

@app.route("/predict")
def predict():

    sample = pd.DataFrame([{
        "temperature": 36.5,
        "humidity": 35,
        "pressure": 1002,
        "wind_speed": 3.5
    }])

    prediction = model.predict(sample)[0]

    return jsonify({
        "Predicted AQI": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)