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
    """Train and return (pipeline, metrics dict).

    Uses a TEMPORAL split (last 20% of days as hold-out): random splits on
    daily time series leak autocorrelated information between train and test
    and inflate metrics.
    """
    df = df.sort_values("date")
    cutoff = df["date"].quantile(0.8)
    train_df, test_df = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_te, y_te = test_df[FEATURES], test_df[TARGET]

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


ARTIFACT_PATH = "src/weights/demand_model.joblib"


if __name__ == "__main__":
    import os

    from src.dataset.generate_dataset import generate_dataset

    df = generate_dataset()
    pipe, metrics = train(df)
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipe, "metrics": metrics}, ARTIFACT_PATH)
    print(f"Saved artifact to {ARTIFACT_PATH}. Metrics:", metrics)
