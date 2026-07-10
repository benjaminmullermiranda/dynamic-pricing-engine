"""Price optimization on top of the learned demand model.

Sweeps a grid of candidate prices, predicts demand at each price with the
trained model, and returns the price that maximizes expected revenue or
profit for a given market context.
"""

import numpy as np
import pandas as pd


def demand_curve(pipe, product: str, prices: np.ndarray, context: dict) -> pd.DataFrame:
    """Predict demand, revenue and profit across a grid of prices.

    context keys: competitor_price, promotion, day_of_week, month,
    is_weekend, unit_cost.
    """
    X = pd.DataFrame({
        "product": product,
        "price": prices,
        "competitor_price": context["competitor_price"],
        "promotion": context["promotion"],
        "day_of_week": context["day_of_week"],
        "month": context["month"],
        "is_weekend": context["is_weekend"],
    })
    demand = np.maximum(pipe.predict(X), 0)
    return pd.DataFrame({
        "price": prices,
        "predicted_units": demand,
        "expected_revenue": prices * demand,
        "expected_profit": (prices - context["unit_cost"]) * demand,
    })


def optimize_price(pipe, product: str, price_range: tuple, context: dict,
                   objective: str = "expected_profit", n_grid: int = 200) -> dict:
    """Return the optimal price and its expected outcomes."""
    prices = np.linspace(price_range[0], price_range[1], n_grid)
    curve = demand_curve(pipe, product, prices, context)
    best = curve.loc[curve[objective].idxmax()]
    return {
        "optimal_price": round(float(best["price"]), 2),
        "expected_units": round(float(best["predicted_units"]), 1),
        "expected_revenue": round(float(best["expected_revenue"]), 2),
        "expected_profit": round(float(best["expected_profit"]), 2),
        "curve": curve,
    }


def estimate_elasticity(pipe, product: str, base_price: float, context: dict) -> float:
    """Price elasticity of demand via log-log regression over the demand curve.

    A two-point finite difference is noisy on tree models (step functions);
    fitting log(demand) ~ log(price) across the +/-20% price band averages
    over many tree splits and matches the simulator's constant-elasticity
    ground truth directly (the slope IS the elasticity).
    """
    prices = np.linspace(base_price * 0.80, base_price * 1.20, 41)
    d = demand_curve(pipe, product, prices, context)["predicted_units"].values
    mask = d > 0
    if mask.sum() < 3:
        return 0.0
    slope = np.polyfit(np.log(prices[mask]), np.log(d[mask]), 1)[0]
    return round(float(slope), 2)
