"""Dynamic Pricing Engine — interactive Streamlit demo.

Trains a demand model on synthetic retail data (cached) and lets the user
explore demand curves, price elasticity and ML-optimized pricing.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataset.generate_dataset import generate_dataset, PRODUCTS
from src.model.train_model import train
from src.optimization.price_optimizer import demand_curve, optimize_price, estimate_elasticity

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="💰", layout="wide")


@st.cache_resource(show_spinner="Training demand model (first load only)...")
def load_model():
    df = generate_dataset()
    pipe, metrics = train(df)
    return df, pipe, metrics


df, pipe, metrics = load_model()

# ---------- Sidebar: market context ----------
st.sidebar.title("⚙️ Market Context")
product = st.sidebar.selectbox("Product", list(PRODUCTS.keys()))
p = PRODUCTS[product]

comp_price = st.sidebar.slider(
    "Competitor price ($)",
    float(round(p["base_price"] * 0.6)), float(round(p["base_price"] * 1.4)),
    float(p["base_price"]),
)
promo = st.sidebar.toggle("Promotion running", value=False)
month = st.sidebar.select_slider("Month", options=list(range(1, 13)), value=6)
weekend = st.sidebar.toggle("Weekend", value=False)
objective = st.sidebar.radio("Optimize for", ["Profit", "Revenue"])

context = {
    "competitor_price": comp_price,
    "promotion": int(promo),
    "day_of_week": 5 if weekend else 2,
    "month": month,
    "is_weekend": int(weekend),
    "unit_cost": p["cost"],
}

# ---------- Header ----------
st.title("💰 Dynamic Pricing Engine")
st.markdown(
    "An ML-driven pricing system: a gradient boosting model learns the **demand curve** "
    "from historical sales, then an optimizer finds the price that maximizes "
    f"**{objective.lower()}** under current market conditions."
)

# ---------- Optimization ----------
obj_col = "expected_profit" if objective == "Profit" else "expected_revenue"
price_range = (p["base_price"] * 0.6, p["base_price"] * 1.4)
result = optimize_price(pipe, product, price_range, context, objective=obj_col)
elasticity = estimate_elasticity(pipe, product, p["base_price"], context)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal price", f"${result['optimal_price']:.2f}",
          f"{(result['optimal_price'] / p['base_price'] - 1) * 100:+.1f}% vs base")
c2.metric("Expected daily units", f"{result['expected_units']:.0f}")
c3.metric(f"Expected daily {objective.lower()}", f"${result[obj_col]:,.0f}")
c4.metric("Price elasticity", f"{elasticity:.2f}",
          "elastic" if abs(elasticity) > 1 else "inelastic", delta_color="off")

# ---------- Curves ----------
curve = result["curve"]
col_a, col_b = st.columns(2)

with col_a:
    fig = go.Figure()
    fig.add_scatter(x=curve["price"], y=curve["predicted_units"], name="Predicted demand",
                    line=dict(color="#4C9BE8", width=3))
    fig.add_vline(x=result["optimal_price"], line_dash="dash", line_color="#E8734C",
                  annotation_text="optimal")
    fig.update_layout(title="Demand curve (model)", xaxis_title="Price ($)",
                      yaxis_title="Units / day", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig2 = go.Figure()
    fig2.add_scatter(x=curve["price"], y=curve["expected_revenue"], name="Revenue",
                     line=dict(color="#57B894", width=3))
    fig2.add_scatter(x=curve["price"], y=curve["expected_profit"], name="Profit",
                     line=dict(color="#B857A8", width=3))
    fig2.add_vline(x=result["optimal_price"], line_dash="dash", line_color="#E8734C")
    fig2.add_vline(x=p["cost"], line_dash="dot", line_color="gray",
                   annotation_text="unit cost")
    fig2.update_layout(title="Revenue & profit vs price", xaxis_title="Price ($)",
                       yaxis_title="$ / day", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- What-if ----------
st.subheader("🎯 What-if: test your own price")
your_price = st.slider("Your price ($)", float(round(price_range[0])),
                       float(round(price_range[1])), float(p["base_price"]))
your = demand_curve(pipe, product, np.array([your_price]), context).iloc[0]
opt_val = result["expected_profit" if objective == "Profit" else "expected_revenue"]
your_val = your["expected_profit" if objective == "Profit" else "expected_revenue"]
gap = opt_val - your_val

w1, w2, w3 = st.columns(3)
w1.metric("Units at your price", f"{your['predicted_units']:.0f}")
w2.metric(f"{objective} at your price", f"${your_val:,.0f}")
w3.metric("Left on the table vs optimal", f"${gap:,.0f}",
          f"{-gap / max(opt_val, 1) * 100:.1f}%", delta_color="inverse")

# ---------- Data & model ----------
with st.expander("📊 Historical data sample & model performance"):
    st.markdown(f"**Model (hold-out test):** MAE = {metrics['MAE']} units · "
                f"R² = {metrics['R2']} · MAPE = {metrics['MAPE_%']}% "
                f"({metrics['n_test']:,} test rows)")
    hist = df[df["product"] == product]
    fig3 = go.Figure()
    fig3.add_scatter(x=hist["price"], y=hist["units_sold"], mode="markers",
                     marker=dict(size=4, opacity=0.35, color="#4C9BE8"),
                     name="Historical sales")
    fig3.update_layout(title=f"Observed price vs units sold — {product}",
                       xaxis_title="Price ($)", yaxis_title="Units sold", height=380)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(hist.tail(200), use_container_width=True, height=250)

st.caption("Built by Benjamin Muller · Demand modeling (Gradient Boosting) + "
           "price optimization · Synthetic data generated with realistic elasticities.")
