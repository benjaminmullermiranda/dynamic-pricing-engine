"""Train the demand forecasting model.

A gradient boosting regressor learns units_sold as a function of price,
competitor price, promotions and calendar features. This learned demand
curve is what the price optimizer searches over.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

FEATURES = ["product", "price", "competitor_price", "promotion", "day_of_week", "month", "is_weekend"]
TARGET = "units_sold"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("product", OneHotEncoder(handle_unknown="ignore"), ["product"])],
        remainder="passthrough",
    )
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.9, random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])


def train(df: pd.DataFrame):
    """Train and return (pipeline, metrics dict)."""
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    metrics = {
        "MAE": round(mean_absolute_error(y_te, pred), 2),
        "R2": round(r2_score(y_te, pred), 3),
        "MAPE_%": round(float(np.mean(np.abs((y_te - pred) / np.maximum(y_te, 1)))) * 100, 1),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    return pipe, metrics


if __name__ == "__main__":
    from src.dataset.generate_dataset import generate_dataset

    df = generate_dataset()
    pipe, metrics = train(df)
    joblib.dump(pipe, "src/weights/demand_model.joblib")
    print("Saved model. Metrics:", metrics)
