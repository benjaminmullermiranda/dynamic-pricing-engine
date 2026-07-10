"""End-to-end pipeline tests: data generation, model quality and
elasticity recovery against the simulator's ground truth."""

import pytest

from src.dataset.generate_dataset import generate_dataset, PRODUCTS
from src.model.train_model import train, FEATURES
from src.optimization.price_optimizer import optimize_price, estimate_elasticity


@pytest.fixture(scope="module")
def data():
    return generate_dataset()


@pytest.fixture(scope="module")
def model(data):
    pipe, metrics = train(data)
    return pipe, metrics


def test_dataset_shape_and_sanity(data):
    assert len(data) == 730 * len(PRODUCTS)
    assert data["units_sold"].min() >= 0
    assert data["price"].gt(0).all()
    assert set(FEATURES).issubset(data.columns)


def test_model_quality(model):
    _, metrics = model
    assert metrics["R2"] > 0.85, f"Model R2 too low: {metrics['R2']}"
    assert metrics["MAPE_%"] < 25, f"MAPE too high: {metrics['MAPE_%']}"


@pytest.mark.parametrize("product", list(PRODUCTS.keys()))
def test_elasticity_recovery(model, product):
    """The model must recover the simulator's true elasticity within tolerance."""
    pipe, _ = model
    p = PRODUCTS[product]
    context = {"competitor_price": p["base_price"], "promotion": 0,
               "day_of_week": 2, "month": 6, "is_weekend": 0, "unit_cost": p["cost"]}
    est = estimate_elasticity(pipe, product, p["base_price"], context)
    assert abs(est - p["elasticity"]) < 0.8, (
        f"{product}: recovered {est}, true {p['elasticity']}")


def test_optimal_price_within_search_range(model):
    pipe, _ = model
    p = PRODUCTS["Wireless Earbuds"]
    context = {"competitor_price": p["base_price"], "promotion": 0,
               "day_of_week": 2, "month": 6, "is_weekend": 0, "unit_cost": p["cost"]}
    lo, hi = p["base_price"] * 0.75, p["base_price"] * 1.25
    res = optimize_price(pipe, "Wireless Earbuds", (lo, hi), context)
    assert lo <= res["optimal_price"] <= hi
    assert res["expected_profit"] > 0


def test_elastic_product_prices_lower_than_inelastic(model):
    """Economic sanity: highly elastic products should be priced closer to
    cost than inelastic ones (relative to base price)."""
    pipe, _ = model
    rel_prices = {}
    for name in ["Wireless Earbuds", "Protein Powder"]:  # elastic vs inelastic
        p = PRODUCTS[name]
        context = {"competitor_price": p["base_price"], "promotion": 0,
                   "day_of_week": 2, "month": 6, "is_weekend": 0, "unit_cost": p["cost"]}
        res = optimize_price(pipe, name, (p["base_price"] * 0.75, p["base_price"] * 1.25),
                             context)
        rel_prices[name] = res["optimal_price"] / p["base_price"]
    assert rel_prices["Wireless Earbuds"] < rel_prices["Protein Powder"]
