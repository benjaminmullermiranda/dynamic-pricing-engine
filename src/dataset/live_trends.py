"""Live product-trend data from Google Trends (pytrends).

Real search-interest data, refreshed daily (cached by the app). If Google
rate-limits or the request fails, callers should fall back to the
synthetic simulator in src/market/trend_radar.py.
"""

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

# representative country per continent (keeps requests low to avoid rate limits)
CONTINENT_GEO = {
    "Asia": "JP",
    "Europe": "ES",
    "North America": "US",
    "South America": "BR",
    "Africa": "ZA",
    "Oceania": "AU",
}

# search keywords for the trending products
TREND_KEYWORDS = {
    "Mini Projector 4K": "mini projector",
    "Air Fryer Compacto": "air fryer",
    "Skincare LED Mask": "led face mask",
    "E-Scooter Plegable": "electric scooter",
    "Bubble Tea Kit": "bubble tea kit",
}


def _client() -> TrendReq:
    return TrendReq(hl="en-US", tz=0, timeout=(5, 15))


def interest_timeline(product: str, continents=("Asia", "Europe"),
                      timeframe: str = "today 5-y") -> pd.DataFrame:
    """Monthly real search interest (0-100) for a product in the given continents."""
    kw = TREND_KEYWORDS[product]
    py = _client()
    frames = {}
    for cont in continents:
        py.build_payload([kw], geo=CONTINENT_GEO[cont], timeframe=timeframe)
        df = py.interest_over_time()
        if df.empty:
            raise RuntimeError(f"No Google Trends data for {kw} in {cont}")
        frames[cont] = df[kw].resample("MS").mean()
    out = pd.DataFrame(frames).dropna()
    out.index.name = "date"
    return out.reset_index()


def current_interest_by_continent(product: str) -> pd.DataFrame:
    """Current search interest snapshot per continent (last 3 months)."""
    kw = TREND_KEYWORDS[product]
    py = _client()
    rows = []
    for cont, geo in CONTINENT_GEO.items():
        py.build_payload([kw], geo=geo, timeframe="today 3-m")
        df = py.interest_over_time()
        rows.append({"continente": cont,
                     "interes_actual": round(float(df[kw].mean()), 1) if not df.empty else 0.0})
    return pd.DataFrame(rows)


def forecast_interest(series: pd.Series, months: int = 12) -> np.ndarray:
    """Simple momentum forecast: fit a line to the last 12 points and
    project it forward, capped to the 0-100 Trends scale."""
    y = series.values[-12:]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    future = intercept + slope * np.arange(len(y), len(y) + months)
    return np.clip(future, 0, 100)


def crossover_score(asia_now: float, target_now: float, target_series: pd.Series) -> dict:
    """Probability that an Asia trend crosses over to a target continent,
    based on the real gap and the target's recent momentum."""
    gap = max(asia_now - target_now, 0) / 100          # how far behind
    y = target_series.values[-6:]
    momentum = np.clip(np.polyfit(np.arange(len(y)), y, 1)[0] / 5, -1, 1)
    prob = np.clip(0.35 + 0.4 * momentum + 0.2 * (1 - gap), 0.05, 0.97)
    return {"probabilidad_%": round(float(prob) * 100),
            "gap_interes": round(gap * 100),
            "momentum": round(float(momentum), 2)}
