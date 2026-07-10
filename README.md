# Dynamic Pricing Engine

An end-to-end machine learning pricing system built as a **validated simulation lab**: a market simulator with known ground-truth price elasticities generates realistic sales data, an ML pipeline learns the demand curves, and an optimizer recommends the profit-maximizing price. Because the true elasticities are known, the whole pipeline can be tested end-to-end — the model must recover them.

**Live demo:** _add your Streamlit Cloud URL here_

![CI](https://github.com/benjaminmullermiranda/dynamic-pricing-engine/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-GBM-orange)

## What it does

1. **Market simulator** — 2 years of daily sales for 6 products with constant-elasticity demand, seasonality, promotions and competitor cross-effects. Prices are randomized in the data (as in a pricing experiment), which makes the demand estimate causally valid.
2. **Demand model** — Gradient Boosting pipeline (`ColumnTransformer` + `GradientBoostingRegressor`) predicting daily units sold.
3. **Price optimizer** — grid search over the learned demand curve, restricted to the support of the training data (tree models do not extrapolate).
4. **Validation** — pytest suite asserts the model recovers each product's true elasticity and that optimal prices follow economic logic (elastic products priced lower). Runs in CI on every push.
5. **Trend radar** — real Google Trends data comparing Asia vs other continents, with a momentum projection and an honest, clearly-labeled heuristic crossover score.

## Data

Sales, prices and market stats are **simulated** (labeled in the UI). This is deliberate: it provides ground truth to validate the ML pipeline against, which real sales data never offers. Trend data is **real** (Google Trends, refreshed daily, with graceful fallback when rate-limited).

## Project structure

```
├── app.py                          <- Streamlit app (entry point)
├── .github/workflows/ci.yml       <- Lint + tests on every push
├── tests/                          <- Pipeline & market logic tests
├── evaluation/evaluate_model.py    <- Per-product metrics + elasticity recovery
└── src/
    ├── dataset/
    │   ├── generate_dataset.py     <- Market simulator (ground truth)
    │   └── live_trends.py          <- Google Trends client + forecasts
    ├── model/train_model.py        <- Demand model pipeline
    ├── optimization/price_optimizer.py
    └── market/                     <- Market stats, trend diffusion model
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py     # app
pytest tests -v          # tests
python -m evaluation.evaluate_model   # metrics report
```

## Known limitations / roadmap

- Trend crossover score is a heuristic, not a calibrated probability.
- Point estimates only — next step: quantile regression for price-recommendation intervals.
- Google Trends samples one representative country per continent (rate limits); a scheduled GitHub Action committing daily snapshots would make it robust.

## Author

**Benjamin Muller** — Business graduate specializing in Data Science / ML Engineering.

MIT License. Structure adapted from [ghimiresunil/Machine-Learning-Project-Structure](https://github.com/ghimiresunil/Machine-Learning-Project-Structure).
