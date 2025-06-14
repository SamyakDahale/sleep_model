# app.py
import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']  # assuming input is a list of features
    input_array = np.array(data).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_label = label_encoder.inverse_transform(prediction)
    return jsonify({'result': predicted_label[0]})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)