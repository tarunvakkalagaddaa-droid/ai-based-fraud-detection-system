"""
app.py
-------
Simple Flask API that serves the trained fraud-detection model.

Run:
    python app.py

Then, in another terminal:
    curl -X POST http://127.0.0.1:5000/predict \\
        -H "Content-Type: application/json" \\
        -d '{
              "amount": 2350.00,
              "hour_of_day": 3,
              "transaction_type": "online",
              "is_foreign_country": 1,
              "account_age_days": 12,
              "num_transactions_today": 8,
              "distance_from_home_km": 640.0
            }'
"""

from flask import Flask, request, jsonify
from predict import load_model, predict_transaction

app = Flask(__name__)

REQUIRED_FIELDS = [
    "amount",
    "hour_of_day",
    "transaction_type",
    "is_foreign_country",
    "account_age_days",
    "num_transactions_today",
    "distance_from_home_km",
]

# Load the model once at startup rather than per-request.
try:
    model = load_model()
except FileNotFoundError as e:
    model = None
    print(f"[warning] {e}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    transaction = {field: payload[field] for field in REQUIRED_FIELDS}

    try:
        result = predict_transaction(transaction, model=model)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
