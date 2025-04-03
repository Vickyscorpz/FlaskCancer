import os
import pickle
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "cancer_model.pkl")  # Default if not set
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")  # Default if not set

# Initialize Flask app
app = Flask(__name__)

# Load model & scaler from environment variable paths
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    print(f"Error loading model/scaler: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        transformed_data = scaler.transform([data])
        prediction = model.predict(transformed_data)
        return render_template("index.html", prediction_text=f'Prediction: {int(prediction[0])}')
    except Exception as e:
        return render_template("index.html", prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
