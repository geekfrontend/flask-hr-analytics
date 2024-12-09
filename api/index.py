from flask import Flask, request, jsonify # type: ignore
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")


MODEL_PATH = './models/logistic_regression_model.pkl'
SCALER_PATH = './models/scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/", methods=["GET"])
def home():
    """Root endpoint with API information."""
    return jsonify({
        "description": "Welcome to the Next.js Flask API!",
        "message": "Please use the form on the Next.js frontend (http://localhost:3000)."
    })


@app.route("/api/predict", methods=["POST", "GET"])
def predict():
    
    if request.method == "GET":
        return jsonify({"error": "GET requests are not allowed."}), 405
    
    """Prediction endpoint."""
    try:
        # Extract features from the request
        data = request.get_json()
        features = np.array(data.get('features', [])).reshape(1, -1)

        if features.size == 0:
            raise ValueError("Features cannot be empty.")

        # Scale features and make predictions
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[:, 1]

        # Return prediction and probability
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
