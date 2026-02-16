from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

model = Sequential()
model.add(Dense(units=10, input_dim=5, activation="relu"))
model.compile(loss="mse", optimizer="adam")

os.makedirs('test_models', exist_ok=True)
model_path = 'test_models/test_save'
try:
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    if os.path.isdir(model_path):
        print(f"{model_path} is a directory.")
    elif os.path.isfile(model_path):
        print(f"{model_path} is a file.")
except Exception as e:
    print(f"Error saving model: {e}")
