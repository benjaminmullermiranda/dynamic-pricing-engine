# 💰 Dynamic Pricing Engine

An end-to-end machine learning system that learns **demand curves** from historical retail sales and recommends the **revenue- or profit-maximizing price** for each product under live market conditions (competitor price, promotions, seasonality).

**🔗 Live demo:** _add your Streamlit Cloud URL here_

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-GBM-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

## Why this project

Most pricing decisions are made by intuition. This engine replaces that with a two-stage ML approach used in real e-commerce and retail:

1. **Demand modeling** — a Gradient Boosting regressor learns `units_sold = f(price, competitor_price, promotion, seasonality)` from 2 years of daily sales across 6 products.
2. **Price optimization** — a grid-search optimizer sweeps candidate prices through the learned demand model and picks the price that maximizes expected revenue or profit.

It also recovers each product's **price elasticity** numerically from the model, which can be validated against the known ground-truth elasticities of the simulated market.

## Project structure

```
dynamic-pricing-engine/
├── app.py                          <- Streamlit app (entry point)
├── config.ini                      <- Project configuration
├── requirements.txt
├── data/                           <- Generated data (git-ignored)
├── evaluation/
│   └── evaluate_model.py           <- Per-product metrics + elasticity recovery
├── notebooks/                      <- EDA / experiments
└── src/
    ├── dataset/
    │   └── generate_dataset.py     <- Synthetic market simulator
    ├── model/
    │   └── train_model.py          <- Demand model training pipeline
    └── optimization/
        └── price_optimizer.py      <- Demand curves, optimizer, elasticity
```

## Results

Run `python -m evaluation.evaluate_model` to get hold-out metrics (MAE, R², MAPE) per product, plus a comparison of **model-recovered elasticities vs. the ground-truth elasticities** of the simulator — evidence the model learned real price-demand structure rather than noise. Paste your actual numbers here after running it.

## Run locally

```bash
git clone https://github.com/<your-user>/dynamic-pricing-engine.git
cd dynamic-pricing-engine
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

Evaluate the model from the CLI:

```bash
python -m evaluation.evaluate_model
```

## Tech stack

Python, pandas, NumPy, scikit-learn (ColumnTransformer + GradientBoostingRegressor pipeline), Plotly, Streamlit. Structure adapted from [ghimiresunil/Machine-Learning-Project-Structure](https://github.com/ghimiresunil/Machine-Learning-Project-Structure).

## Author

**Benjamin Muller** — Business graduate specializing in Data Science / ML Engineering.

## License

MIT
