"""
generate_data.py
-----------------
Creates a synthetic credit-card / online-transaction dataset with a
realistic mix of normal and fraudulent transactions, and saves it to
data/transactions.csv.

In a real project you would replace this with an actual dataset
(e.g. the Kaggle "Credit Card Fraud Detection" dataset). This
synthetic generator exists so the whole pipeline runs end-to-end
out of the box.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 20000
FRAUD_RATIO = 0.03  # ~3% of transactions are fraudulent (realistic imbalance)


def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # ---- Normal transactions ----
    normal = pd.DataFrame({
        "amount": np.round(rng.gamma(shape=2.0, scale=40, size=n_normal), 2),
        "hour_of_day": rng.integers(6, 23, size=n_normal),  # mostly daytime
        "transaction_type": rng.choice(
            ["purchase", "withdrawal", "transfer", "online"],
            size=n_normal, p=[0.5, 0.2, 0.15, 0.15]
        ),
        "is_foreign_country": rng.choice([0, 1], size=n_normal, p=[0.95, 0.05]),
        "account_age_days": rng.integers(30, 3650, size=n_normal),
        "num_transactions_today": rng.poisson(2, size=n_normal),
        "distance_from_home_km": np.round(rng.exponential(scale=15, size=n_normal), 2),
        "is_fraud": 0,
    })

    # ---- Fraudulent transactions (different distribution) ----
    fraud = pd.DataFrame({
        "amount": np.round(rng.gamma(shape=3.0, scale=250, size=n_fraud), 2),  # larger amounts
        "hour_of_day": rng.choice(
            range(24), size=n_fraud,
            p=_night_weighted_probs()
        ),  # skewed toward odd hours
        "transaction_type": rng.choice(
            ["purchase", "withdrawal", "transfer", "online"],
            size=n_fraud, p=[0.2, 0.15, 0.15, 0.5]
        ),
        "is_foreign_country": rng.choice([0, 1], size=n_fraud, p=[0.4, 0.6]),
        "account_age_days": rng.integers(1, 365, size=n_fraud),  # newer accounts
        "num_transactions_today": rng.poisson(6, size=n_fraud),  # burst of activity
        "distance_from_home_km": np.round(rng.exponential(scale=300, size=n_fraud), 2),
        "is_fraud": 1,
    })

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def _night_weighted_probs():
    """Return a probability distribution over 24 hours weighted toward late night."""
    hours = np.arange(24)
    weights = np.where((hours >= 0) & (hours <= 5), 3.0, 1.0)
    return weights / weights.sum()


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions -> {out_path}")
    print(df["is_fraud"].value_counts(normalize=True).rename("proportion"))
