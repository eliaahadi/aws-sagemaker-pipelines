# data/generate_telco.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def generate_telco(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Numeric drivers
    tenure = rng.integers(0, 72, size=n)  # months
    monthly = np.round(rng.normal(70, 20, size=n).clip(15, 200), 2)
    total = np.round(monthly * (tenure + rng.normal(1.0, 0.2, size=n)).clip(0.1, None), 2)

    # Categorical/boolean drivers
    contract = rng.choice(["month-to-month", "one_year", "two_year"], p=[0.62, 0.22, 0.16], size=n)
    internet = rng.choice(["dsl", "fiber", "none"], p=[0.33, 0.49, 0.18], size=n)
    online_sec = rng.choice(["yes", "no"], p=[0.42, 0.58], size=n)
    tech_support = rng.choice(["yes", "no"], p=[0.44, 0.56], size=n)
    senior = rng.choice([0, 1], p=[0.84, 0.16], size=n)
    partner = rng.choice([0, 1], p=[0.52, 0.48], size=n)
    dependents = rng.choice([0, 1], p=[0.7, 0.3], size=n)
    paperless = rng.choice(["yes", "no"], p=[0.72, 0.28], size=n)
    payment = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        p=[0.41, 0.21, 0.19, 0.19],
        size=n,
    )

    # Latent churn logit (higher => more likely to churn)
    logit = (
        -2.0
        + 0.9 * (contract == "month-to-month").astype(float)
        + 0.6 * (paperless == "yes").astype(float)
        + 0.7 * (internet == "fiber").astype(float)
        - 0.03 * tenure
        + 0.008 * (monthly - 70.0)
        + 0.5 * (online_sec == "no").astype(float)
        + 0.4 * (tech_support == "no").astype(float)
        + 0.15 * senior
        - 0.1 * partner
        - 0.12 * dependents
        + rng.normal(0, 0.5, size=n)
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.random(size=n) < p).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract": contract,
        "internet_service": internet,
        "online_security": online_sec,
        "tech_support": tech_support,
        "is_senior": senior,
        "has_partner": partner,
        "has_dependents": dependents,
        "paperless_billing": paperless,
        "payment_method": payment,
        "churn": churn,
    })
    return df

if __name__ == "__main__":
    out_dir = Path("data"); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "telco_churn.csv"
    generate_telco().to_csv(path, index=False)
    print(f"Wrote {path.resolve()}")