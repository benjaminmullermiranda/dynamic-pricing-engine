"""Trend radar: products trending in Asia and their expected diffusion
to other continents.

Uses a logistic (S-curve) diffusion model — the standard way product
adoption spreads across markets — to forecast profit in lagging
continents and score the probability that the trend crosses over.
"""

import numpy as np
import pandas as pd

# Products already viral in Asia, not yet elsewhere.
# asia_start: months since trend started in Asia
# lag: typical months of delay Asia -> West for the category
# peak_profit: monthly profit at saturation in Asia ($/month, one seller mid-size)
TRENDING = {
    "Mini Projector 4K": {"asia_start": 14, "lag": 8, "peak_profit": 42000, "category": "Electrónica"},
    "Air Fryer Compacto": {"asia_start": 20, "lag": 6, "peak_profit": 35000, "category": "Hogar"},
    "Skincare LED Mask": {"asia_start": 10, "lag": 9, "peak_profit": 28000, "category": "Belleza"},
    "E-Scooter Plegable": {"asia_start": 16, "lag": 10, "peak_profit": 55000, "category": "Movilidad"},
    "Bubble Tea Kit": {"asia_start": 24, "lag": 5, "peak_profit": 18000, "category": "Alimentación"},
}

# how receptive each continent is to Asian trends (0-1) and extra lag in months
RECEPTIVITY = {
    "Europe": {"factor": 0.75, "extra_lag": 0},
    "North America": {"factor": 0.85, "extra_lag": -1},
    "South America": {"factor": 0.55, "extra_lag": 4},
    "Africa": {"factor": 0.35, "extra_lag": 8},
    "Oceania": {"factor": 0.65, "extra_lag": 2},
}


def _logistic(t, peak, midpoint, rate=0.55):
    return peak / (1 + np.exp(-rate * (t - midpoint)))


def trend_timeline(product: str, months_history: int = 24, months_forecast: int = 18,
                   seed: int = 7) -> pd.DataFrame:
    """Monthly profit: Asia (observed + projected) and Europe (observed + forecast)."""
    p = TRENDING[product]
    rng = np.random.default_rng(seed + len(product))
    t = np.arange(-months_history, months_forecast + 1)  # 0 = today

    # Asia: S-curve that started asia_start months ago
    asia = _logistic(t + months_history, p["peak_profit"],
                     midpoint=months_history - p["asia_start"] + 6)
    asia_obs = asia * rng.normal(1, 0.06, len(t))

    # Europe: same curve delayed by lag, scaled by receptivity
    r = RECEPTIVITY["Europe"]
    eu_peak = p["peak_profit"] * r["factor"]
    europe = _logistic(t + months_history, eu_peak,
                       midpoint=months_history - p["asia_start"] + 6 + p["lag"] + r["extra_lag"])
    europe_obs = europe * rng.normal(1, 0.10, len(t))

    df = pd.DataFrame({"month": t, "asia": asia_obs.round(0), "europe": europe_obs.round(0)})
    df["period"] = np.where(df["month"] <= 0, "history", "forecast")
    return df


def crossover_probability(product: str) -> pd.DataFrame:
    """Probability (%) that the Asian trend becomes a trend in each continent,
    and estimated months until it takes off there."""
    p = TRENDING[product]
    rows = []
    for cont, r in RECEPTIVITY.items():
        # more months trending in Asia + higher receptivity => higher probability
        maturity = min(p["asia_start"] / 24, 1.0)          # how proven the trend is
        prob = min(0.15 + 0.75 * maturity * r["factor"] + 0.05, 0.97)
        months_to_takeoff = max(p["lag"] + r["extra_lag"] - p["asia_start"] + 12, 1)
        rows.append({
            "continent": cont,
            "crossover_score": round(prob * 100, 0),
            "est_months_to_takeoff": int(months_to_takeoff),
            "potential_monthly_profit": round(p["peak_profit"] * r["factor"], -2),
        })
    return pd.DataFrame(rows).sort_values("crossover_score", ascending=False)
