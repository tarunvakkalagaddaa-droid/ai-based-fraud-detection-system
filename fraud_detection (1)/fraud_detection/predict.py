"""
predict.py
-----------
Loads the trained model and predicts whether a single transaction is
fraudulent. Can be used as a library (see `predict_transaction`) or
run directly for a quick demo.

Run:
    python predict.py
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "fraud_model.joblib")

FRAUD_THRESHOLD = 0.5  # probability above which a transaction is flagged as fraud


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python train_model.py` first to train and save a model."
        )
    return joblib.load(path)


def predict_transaction(transaction: dict, model=None, threshold: float = FRAUD_THRESHOLD):
    """
    transaction: dict with keys:
        amount, hour_of_day, transaction_type, is_foreign_country,
        account_age_days, num_transactions_today, distance_from_home_km

    Returns: dict with fraud_probability, is_fraud (bool), and the input echoed back.
    """
    if model is None:
        model = load_model()

    df = pd.DataFrame([transaction])
    proba = model.predict_proba(df)[0, 1]

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": bool(proba >= threshold),
        "threshold_used": threshold,
        "input": transaction,
    }


if __name__ == "__main__":
    model = load_model()

    sample_transactions = [
        {
            "amount": 45.00,
            "hour_of_day": 14,
            "transaction_type": "purchase",
            "is_foreign_country": 0,
            "account_age_days": 900,
            "num_transactions_today": 2,
            "distance_from_home_km": 3.2,
        },
        {
            "amount": 2350.00,
            "hour_of_day": 3,
            "transaction_type": "online",
            "is_foreign_country": 1,
            "account_age_days": 12,
            "num_transactions_today": 8,
            "distance_from_home_km": 640.0,
        },
    ]

    for i, txn in enumerate(sample_transactions, 1):
        result = predict_transaction(txn, model=model)
        label = "FRAUD" if result["is_fraud"] else "legit"
        print(f"Transaction {i}: probability={result['fraud_probability']:.2%} -> {label}")
