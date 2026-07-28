import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

# Load dataset
df = pd.read_csv("data/processed/training_data.csv")

X = df[["temperature", "humidity", "pressure", "wind_speed"]]
y = df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=1
)

predictions = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== TensorFlow Model =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

model.save("models/tensorflow_model.keras")

print("\n✅ TensorFlow model saved successfully!")