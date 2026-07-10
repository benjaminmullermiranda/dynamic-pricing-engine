"""Model evaluation: metrics per product + recovered vs. true elasticity."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.dataset.generate_dataset import generate_dataset, PRODUCTS
from src.model.train_model import train, FEATURES, TARGET
from src.optimization.price_optimizer import estimate_elasticity


def evaluate():
    df = generate_dataset()
    pipe, metrics = train(df)
    print("Global test metrics:", metrics)

    rows = []
    for product, p in PRODUCTS.items():
        sub = df[df["product"] == product]
        pred = pipe.predict(sub[FEATURES])
        context = {
            "competitor_price": p["base_price"], "promotion": 0,
            "day_of_week": 2, "month": 6, "is_weekend": 0,
            "unit_cost": p["cost"],
        }
        rows.append({
            "product": product,
            "MAE": round(mean_absolute_error(sub[TARGET], pred), 2),
            "R2": round(r2_score(sub[TARGET], pred), 3),
            "true_elasticity": p["elasticity"],
            "model_elasticity": estimate_elasticity(pipe, product, p["base_price"], context),
        })
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    evaluate()
