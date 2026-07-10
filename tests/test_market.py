"""Market logic tests: saturation, competitiveness and trend heuristics."""

import pandas as pd

from src.market.market_data import CONTINENTS, market_summary, competitiveness
from src.market.trend_radar import trend_timeline, crossover_probability
from src.dataset.live_trends import forecast_interest, crossover_score


def test_market_summary_all_continents():
    for cont in CONTINENTS:
        m = market_summary("Wireless Earbuds", 79.0, cont)
        assert m["avg_market_price"] > 0
        assert 0 <= m["saturation_score"] <= 1


def test_competitiveness_below_market_is_competitive():
    c = competitiveness(your_price=70, avg_market_price=80, saturation_score=0.8)
    assert c["diff_pct"] < 0
    assert c["verdict"] == "Muy competitivo"


def test_competitiveness_far_above_market_is_not():
    c = competitiveness(your_price=150, avg_market_price=80, saturation_score=0.8)
    assert c["verdict"] == "No competitivo"


def test_trend_timeline_shapes():
    df = trend_timeline("Mini Projector 4K")
    assert {"month", "asia", "europe", "period"}.issubset(df.columns)
    assert (df["asia"] >= 0).all()
    assert set(df["period"]) == {"history", "forecast"}


def test_crossover_probability_sorted():
    df = crossover_probability("Air Fryer Compacto")
    assert df["crossover_score"].is_monotonic_decreasing
    assert df["crossover_score"].between(0, 100).all()


def test_forecast_interest_bounded():
    s = pd.Series(range(30, 90, 5))
    fc = forecast_interest(s, months=12)
    assert len(fc) == 12
    assert (fc >= 0).all() and (fc <= 100).all()


def test_crossover_score_gap_lowers_score():
    s = pd.Series([50, 52, 54, 56, 58, 60])
    near = crossover_score(60, 55, s)["score"]
    far = crossover_score(95, 5, s)["score"]
    assert near >= far
