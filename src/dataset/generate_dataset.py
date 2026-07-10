"""Synthetic retail pricing dataset generator.

Simulates 2 years of daily sales for a catalog of products with realistic
price elasticity, seasonality, promotions and competitor effects.
"""

import numpy as np
import pandas as pd

PRODUCTS = {
    "Wireless Earbuds": {"base_price": 79.0, "cost": 32.0, "elasticity": -2.1, "base_demand": 120},
    "Smart Watch": {"base_price": 199.0, "cost": 95.0, "elasticity": -1.6, "base_demand": 60},
    "Coffee Maker": {"base_price": 89.0, "cost": 41.0, "elasticity": -1.3, "base_demand": 85},
    "Yoga Mat": {"base_price": 35.0, "cost": 11.0, "elasticity": -0.9, "base_demand": 150},
    "Gaming Mouse": {"base_price": 59.0, "cost": 22.0, "elasticity": -1.8, "base_demand": 100},
    "Protein Powder": {"base_price": 45.0, "cost": 18.0, "elasticity": -0.7, "base_demand": 200},
}


def generate_dataset(start="2024-01-01", days=730, seed=42) -> pd.DataFrame:
    """Return a daily sales DataFrame for all products."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []

    for product, p in PRODUCTS.items():
        # our price varies around base price (historical pricing experiments)
        price = p["base_price"] * rng.uniform(0.75, 1.25, days)

        # competitor tracks the market price with noise
        comp_price = p["base_price"] * rng.normal(1.0, 0.08, days)

        promo = rng.random(days) < 0.12  # ~12% of days on promotion
        dow = dates.dayofweek
        weekend = (dow >= 5).astype(float)
        # yearly seasonality + holiday-season bump
        season = 1 + 0.20 * np.sin(2 * np.pi * (dates.dayofyear - 320) / 365)
        holiday = np.where(dates.month == 12, 1.25, 1.0)

        # ground-truth demand: constant elasticity + business effects
        demand = (
            p["base_demand"]
            * (price / p["base_price"]) ** p["elasticity"]
            * (comp_price / p["base_price"]) ** 0.6  # cross-elasticity
            * season * holiday
            * (1 + 0.35 * promo)
            * (1 + 0.15 * weekend)
        )
        units = rng.poisson(np.maximum(demand, 0.1))

        rows.append(pd.DataFrame({
            "date": dates,
            "product": product,
            "price": price.round(2),
            "competitor_price": comp_price.round(2),
            "promotion": promo.astype(int),
            "day_of_week": dow,
            "month": dates.month,
            "is_weekend": weekend.astype(int),
            "units_sold": units,
            "unit_cost": p["cost"],
        }))

    df = pd.concat(rows, ignore_index=True)
    df["revenue"] = (df["price"] * df["units_sold"]).round(2)
    df["profit"] = ((df["price"] - df["unit_cost"]) * df["units_sold"]).round(2)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/raw/retail_pricing.csv", index=False)
    print(f"Generated {len(df):,} rows for {df['product'].nunique()} products")
