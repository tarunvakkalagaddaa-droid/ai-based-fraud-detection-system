# AI-Based Fraud Detection System

A simple, end-to-end base project for detecting fraudulent transactions
using a machine learning model (Random Forest), with a REST API for
real-time scoring.

## Project Structure

```
fraud_detection/
├── generate_data.py   # Creates a synthetic transactions dataset (swap for real data)
├── train_model.py      # Preprocesses data, trains model, evaluates, saves it
├── predict.py           # Loads the model and scores a transaction
├── app.py                # Flask REST API exposing /predict and /health
├── requirements.txt
├── data/
│   └── transactions.csv   # generated dataset (created by generate_data.py)
└── models/
    └── fraud_model.joblib  # trained model (created by train_model.py)
```

## How It Works

1. **Data** — Each transaction has features like `amount`, `hour_of_day`,
   `transaction_type`, `is_foreign_country`, `account_age_days`,
   `num_transactions_today`, and `distance_from_home_km`. The label is
   `is_fraud` (0 = legit, 1 = fraud).
2. **Preprocessing** — Numeric features are scaled with `StandardScaler`;
   the categorical `transaction_type` is one-hot encoded. This is wrapped
   in a `ColumnTransformer` inside an sklearn `Pipeline`, so the same
   transformations are automatically applied at prediction time.
3. **Model** — A `RandomForestClassifier` with `class_weight="balanced"`
   to handle the natural class imbalance (fraud is rare).
4. **Evaluation** — Precision, recall, F1, confusion matrix, ROC-AUC, and
   a look at precision/recall trade-offs at different probability
   thresholds (important for fraud, where recall vs. false-positive cost
   is a business decision, not just an accuracy number).
5. **Serving** — `app.py` loads the saved model once and exposes a
   `/predict` endpoint that returns a fraud probability and a flag.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate sample data (skip this if you have real data)

```bash
python generate_data.py
```

This creates `data/transactions.csv`. To use your own dataset instead,
just place a CSV with the same columns at that path (or edit
`DATA_PATH` in `train_model.py`).

### 2. Train the model

```bash
python train_model.py
```

Prints a classification report, confusion matrix, ROC-AUC, and feature
importances, then saves the trained pipeline to `models/fraud_model.joblib`.

### 3. Predict from the command line

```bash
python predict.py
```

Runs two example transactions through the model and prints fraud
probabilities.

### 4. Run the API

```bash
python app.py
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "amount": 2350.00,
        "hour_of_day": 3,
        "transaction_type": "online",
        "is_foreign_country": 1,
        "account_age_days": 12,
        "num_transactions_today": 8,
        "distance_from_home_km": 640.0
      }'
```

Response:

```json
{
  "fraud_probability": 1.0,
  "is_fraud": true,
  "threshold_used": 0.5,
  "input": { ... }
}
```

Check the API is alive with `GET /health`.

## Next Steps (for extending this base project)

- Swap the synthetic data generator for a real dataset (e.g. Kaggle's
  "Credit Card Fraud Detection" dataset) or a live transaction feed.
- Try other models (XGBoost, LightGBM, or a neural network/autoencoder
  for anomaly detection) and compare against this baseline.
- Add cross-validation and hyperparameter tuning (`GridSearchCV` /
  `Optuna`) instead of fixed hyperparameters.
- Log predictions and outcomes to retrain the model periodically
  (concept drift is a real problem in fraud detection).
- Add authentication and rate limiting to the API before any real
  deployment, and swap Flask's dev server for a production WSGI server
  (e.g. gunicorn).
- Add SHAP or similar explainability output so flagged transactions
  can be reviewed by a human analyst.
