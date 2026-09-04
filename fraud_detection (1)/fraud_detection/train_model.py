"""
train_model.py
----------------
Loads data/transactions.csv, preprocesses it, trains a Random Forest
fraud classifier, evaluates it, and saves the trained model + the
preprocessing objects (scaler, encoder) to the models/ folder.

Run:
    python generate_data.py     # only needed once, to create sample data
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)

DATA_PATH = os.path.join("data", "transactions.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")

NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "is_foreign_country",
    "account_age_days",
    "num_transactions_today",
    "distance_from_home_km",
]
CATEGORICAL_FEATURES = ["transaction_type"]
TARGET = "is_fraud"


def build_pipeline():
    """Builds a preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",  # important: fraud is rare, so re-weight classes
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first, "
            f"or replace it with your own transactions dataset."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    print("=" * 60)
    print("CONFUSION MATRIX  (rows=actual, cols=predicted)")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=["actual_legit", "actual_fraud"],
                        columns=["pred_legit", "pred_fraud"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {auc:.4f}")

    # Show what fraud-probability threshold gives good precision/recall trade-off
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print("\nSample precision/recall at a few thresholds:")
    for t in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds - t))
        print(f"  threshold={t:.1f} -> precision={precisions[idx]:.2f}, recall={recalls[idx]:.2f}")

    # ---- Feature importance ----
    ohe_cols = pipeline.named_steps["preprocessor"] \
        .named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERIC_FEATURES + list(ohe_cols)
    importances = pipeline.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(feat_imp.head(10))

    # ---- Save ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
