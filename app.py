"""Dynamic Pricing Engine - interactive Streamlit demo.

A simulation-driven pricing lab: the market simulator has known ground-truth
elasticities, the ML pipeline must recover them, and the optimizer prices
against the learned demand curve. Trend data is real (Google Trends).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataset.generate_dataset import generate_dataset, PRODUCTS
from src.model.train_model import train
from src.optimization.price_optimizer import demand_curve, optimize_price, estimate_elasticity
from src.market.market_data import CONTINENTS, market_summary, competitiveness
from src.market.trend_radar import TRENDING, trend_timeline, crossover_probability
from src.dataset.live_trends import (
    TREND_KEYWORDS, interest_timeline, current_interest_by_continent,
    forecast_interest, crossover_score,
)

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")

BRAND = {
    "green": "#0acf83", "orange": "#f24e1e", "purple": "#a259ff",
    "red": "#ff7262", "blue": "#1abcfe",
}

# training data uses prices in 0.75-1.25x of base; the optimizer must not
# search outside the support of the training data (GBMs do not extrapolate)
PRICE_SEARCH_RANGE = (0.75, 1.25)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: #0f0f14; }
h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; }
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e5e5ea; border-radius: 12px;
    padding: 20px 24px; transition: box-shadow .2s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 8px 24px rgba(15,15,20,0.08); }
[data-testid="stMetricLabel"] { color: #6e6e78; font-weight: 500; }
[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e5ea; }
.stButton > button { border-radius: 8px; font-weight: 500; padding: 10px 18px; }
[data-testid="stExpander"] { border: 1px solid #e5e5ea; border-radius: 12px; background: #ffffff; }
.badge-sim, .badge-live {
    display: inline-block; border-radius: 999px; padding: 2px 12px;
    font-size: 12px; font-weight: 600; vertical-align: middle;
}
.badge-sim  { background: rgba(242,78,30,.12);  color: #c23e15; }
.badge-live { background: rgba(10,207,131,.15); color: #067a4e; }
.stProgress > div > div > div > div { background-color: #0acf83; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

SIM_BADGE = '<span class="badge-sim">SIMULATED DATA</span>'
LIVE_BADGE = '<span class="badge-live">LIVE DATA</span>'


@st.cache_resource(show_spinner="Loading the demand model...")
def load_model():
    """Load the pre-trained artifact if available; otherwise train from scratch."""
    import os

    import joblib

    from src.model.train_model import ARTIFACT_PATH

    df = generate_dataset()
    if os.path.exists(ARTIFACT_PATH):
        try:
            artifact = joblib.load(ARTIFACT_PATH)
            return df, artifact["pipeline"], artifact["metrics"]
        except Exception:
            pass  # version mismatch etc. -> retrain
    pipe, metrics = train(df)
    return df, pipe, metrics


df, pipe, metrics = load_model()

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
continent = st.sidebar.selectbox("Continent", list(CONTINENTS.keys()), index=2)
product = st.sidebar.selectbox("Product", list(PRODUCTS.keys()))
p = PRODUCTS[product]

comp_price = st.sidebar.slider(
    "Competitor price ($)",
    float(round(p["base_price"] * 0.75)), float(round(p["base_price"] * 1.25)),
    float(p["base_price"]),
)
promo = st.sidebar.toggle("Promotion running", value=False)
month = st.sidebar.select_slider("Month", options=list(range(1, 13)), value=6)
objective = st.sidebar.radio("Optimize for", ["Profit", "Revenue"])

st.sidebar.divider()
st.sidebar.caption(
    "About the data: sales, prices and market stats are **simulated** with known "
    "ground-truth elasticities - the ML pipeline is validated by recovering them. "
    "Trend data is **real** (Google Trends)."
)

context = {
    "competitor_price": comp_price,
    "promotion": int(promo),
    "day_of_week": 2,
    "month": month,
    "is_weekend": 0,
    "unit_cost": p["cost"],
}

# ---------------- Header ----------------
st.title("Dynamic Pricing Engine")
st.markdown(
    "**Selling a product online? This tool answers three questions:** "
    "where to sell it, at what price, and how much you would earn."
)

with st.container(border=True):
    st.markdown(
        "**How to use it (30 seconds):**\n\n"
        "1. In the sidebar, pick a **continent** and a **product**.\n"
        "2. **Tab 1** shows how competitive that market is.\n"
        "3. **Tab 2** gives you the price that earns you the most money.\n"
        "4. **Tab 3** calculates your monthly profit with your own buy/sell prices.\n"
        "5. **Tab 4** shows products trending in Asia before they arrive in your market.\n\n"
        "*Try it: select 'Wireless Earbuds' and 'Europe', then open Tab 2.*"
    )

tab_market, tab_price, tab_profit, tab_trends = st.tabs(
    ["1. Market", "2. Optimal price", "3. Your profit", "4. Trend radar"]
)

# ---------------- Tab 1: Market ----------------
with tab_market:
    st.markdown(f"### Market overview - {continent} &nbsp;{SIM_BADGE}", unsafe_allow_html=True)

    market = market_summary(product, p["base_price"], continent)

    m1, m2 = st.columns(2)
    m1.metric("Average market price", f"${market['avg_market_price']:.2f}")
    m2.metric("Active sellers", f"{market['sellers']:,}")

    sat_labels = {"Saturado": "Saturated - many sellers competing",
                  "Competencia moderada": "Moderate competition",
                  "Oportunidad": "Open opportunity - few sellers"}
    st.markdown(f"**Market saturation:** "
                f"{sat_labels.get(market['saturation_level'], market['saturation_level'])}")
    st.progress(market["saturation_score"])
    st.caption("The fuller the bar, the harder it is to stand out in this market. "
               "Seller counts and market prices are simulated for demonstration.")

    with st.expander("Show market map"):
        fig_map = go.Figure(go.Choropleth(
            locations=CONTINENTS[continent]["countries"],
            z=[1] * len(CONTINENTS[continent]["countries"]),
            colorscale=[[0, BRAND["purple"]], [1, BRAND["purple"]]],
            showscale=False, marker_line_color="white", marker_line_width=0.5,
        ))
        fig_map.update_geos(showcountries=True, countrycolor="#666", showland=True,
                            landcolor="#e8e8e8", fitbounds="locations",
                            projection_type="natural earth")
        fig_map.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# ---------------- Tab 2: Optimal price ----------------
with tab_price:
    st.markdown(f"### Optimal price &nbsp;{SIM_BADGE}", unsafe_allow_html=True)

    obj_col = "expected_profit" if objective == "Profit" else "expected_revenue"
    price_range = (p["base_price"] * PRICE_SEARCH_RANGE[0],
                   p["base_price"] * PRICE_SEARCH_RANGE[1])
    result = optimize_price(pipe, product, price_range, context, objective=obj_col)
    elasticity = estimate_elasticity(pipe, product, p["base_price"], context)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best price", f"${result['optimal_price']:.2f}",
              f"{(result['optimal_price'] / p['base_price'] - 1) * 100:+.1f}% vs base price")
    c2.metric("Expected daily sales", f"{result['expected_units']:.0f} units")
    c3.metric(f"Expected daily {objective.lower()}", f"${result[obj_col]:,.0f}")
    c4.metric("Price sensitivity", f"{abs(elasticity) * 10:.0f}% fewer sales",
              "if you raise the price 10%", delta_color="off",
              help=f"Price elasticity: {elasticity:.2f} (model) vs {p['elasticity']:.1f} "
                   "(simulator ground truth). The model recovering the true value "
                   "validates the pipeline.")

    # actionable conclusion: optimal vs current base price
    base_row = demand_curve(pipe, product, np.array([p["base_price"]]), context).iloc[0]
    base_val = float(base_row[obj_col])
    uplift = (result[obj_col] / base_val - 1) * 100 if base_val > 0 else 0
    st.success(
        f"**Recommendation: sell {product} at ${result['optimal_price']:.2f}.** "
        f"You would earn about ${result[obj_col]:,.0f} per day in {objective.lower()} - "
        f"{uplift:+.0f}% compared with the current base price of ${p['base_price']:.2f}."
    )

    curve = result["curve"]
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_scatter(x=curve["price"], y=curve["predicted_units"], name="Expected sales",
                        line=dict(color=BRAND["blue"], width=3))
        fig.add_vline(x=result["optimal_price"], line_dash="dash", line_color=BRAND["orange"],
                      annotation_text="best price")
        fig.update_layout(title="How sales change with price", xaxis_title="Price ($)",
                          yaxis_title="Units per day", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = go.Figure()
        fig2.add_scatter(x=curve["price"], y=curve["expected_revenue"], name="Revenue",
                         line=dict(color=BRAND["green"], width=3))
        fig2.add_scatter(x=curve["price"], y=curve["expected_profit"], name="Profit",
                         line=dict(color=BRAND["purple"], width=3))
        fig2.add_vline(x=result["optimal_price"], line_dash="dash", line_color=BRAND["orange"])
        fig2.update_layout(title="Revenue and profit at each price", xaxis_title="Price ($)",
                           yaxis_title="$ per day", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"The search range ({PRICE_SEARCH_RANGE[0]}x - {PRICE_SEARCH_RANGE[1]}x of base "
               "price) matches the support of the training data: tree models cannot "
               "extrapolate outside the prices they were trained on.")

    with st.expander("Model performance (hold-out test set)"):
        st.markdown(f"MAE = {metrics['MAE']} units | R2 = {metrics['R2']} | "
                    f"MAPE = {metrics['MAPE_%']}% ({metrics['n_test']:,} test rows)")
        hist = df[df["product"] == product]
        fig3 = go.Figure()
        fig3.add_scatter(x=hist["price"], y=hist["units_sold"], mode="markers",
                         marker=dict(size=4, opacity=0.35, color=BRAND["red"]),
                         name="Simulated sales history")
        fig3.update_layout(title=f"Price vs units sold - {product}",
                           xaxis_title="Price ($)", yaxis_title="Units sold", height=380)
        st.plotly_chart(fig3, use_container_width=True)

# ---------------- Tab 3: Profit calculator ----------------
with tab_profit:
    st.markdown(f"### Profit calculator &nbsp;{SIM_BADGE}", unsafe_allow_html=True)
    st.markdown("Enter your real prices and see your estimated profit. Sales volume is "
                "predicted by the demand model.")

    market = market_summary(product, p["base_price"], continent)

    g1, g2, g3 = st.columns(3)
    buy_price = g1.number_input("Price you BUY at ($)", min_value=0.01,
                                value=float(p["cost"]), step=0.5)
    sell_price = g2.number_input("Price you SELL at ($)", min_value=0.01,
                                 value=float(p["base_price"]), step=0.5)
    competitor_input = g3.number_input("Competitor's price ($)", min_value=0.01,
                                       value=float(comp_price), step=0.5)

    calc_context = dict(context, competitor_price=competitor_input)
    est_units = float(demand_curve(pipe, product, np.array([sell_price]), calc_context)
                      ["predicted_units"].iloc[0])

    margin_unit = sell_price - buy_price
    margin_pct = margin_unit / sell_price * 100
    daily_profit = margin_unit * est_units
    vs_comp = (sell_price - competitor_input) / competitor_input * 100

    comp_verdict = competitiveness(sell_price, market["avg_market_price"],
                                   market["saturation_score"])
    verdict_en = {"Muy competitivo": "Very competitive", "Competitivo": "Competitive",
                  "Poco competitivo": "Weak", "No competitivo": "Not competitive"}

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Profit per unit", f"${margin_unit:,.2f}", f"{margin_pct:.1f}% margin")
    r2.metric("Estimated daily sales", f"{est_units:.0f} units")
    r3.metric("Estimated monthly profit", f"${daily_profit * 30:,.0f}",
              f"${daily_profit:,.0f} per day")
    r4.metric("Your price vs competitor", f"{vs_comp:+.1f}%",
              "cheaper" if vs_comp < 0 else "more expensive", delta_color="off")

    if margin_unit <= 0:
        st.error("You are selling below your purchase cost: you lose money on every sale.")
    elif margin_pct < 15:
        st.warning("Thin margin (below 15%). Remember fees, shipping and ads still "
                   "need to be paid.")
    else:
        st.success(f"Healthy margin. Competitiveness in {continent}: "
                   f"{verdict_en.get(comp_verdict['verdict'], comp_verdict['verdict'])} "
                   f"({comp_verdict['diff_pct']:+.1f}% vs average market price).")

    sell_lo, sell_hi = price_range
    if not (sell_lo <= sell_price <= sell_hi):
        st.info(f"Note: ${sell_price:.2f} is outside the range the model was trained on "
                f"(${sell_lo:.2f} - ${sell_hi:.2f}), so the sales estimate is unreliable there.")

# ---------------- Tab 4: Trend radar ----------------
with tab_trends:
    st.markdown(f"### Trend radar: Asia to the world &nbsp;{LIVE_BADGE}", unsafe_allow_html=True)
    st.markdown("Products already popular in Asia that have not taken off elsewhere yet. "
                "Real Google Trends search interest, refreshed daily, with a 12-month "
                "momentum projection.")

    trend_product = st.selectbox("Trending product", list(TRENDING.keys()))

    @st.cache_data(ttl=86400, show_spinner="Fetching Google Trends data...")
    def load_live_trends(product_name):
        tl = interest_timeline(product_name, ("Asia", "Europe"))
        snapshot = current_interest_by_continent(product_name)
        return tl, snapshot

    live_data_ok = True
    try:
        timeline, snapshot = load_live_trends(trend_product)
    except Exception:
        live_data_ok = False
        st.warning("Google Trends is not available right now (rate limit). "
                   "Showing the diffusion-model simulation instead.")
        st.markdown(SIM_BADGE, unsafe_allow_html=True)

    t1, t2 = st.columns([1.4, 1])
    if live_data_ok:
        fc_eu = forecast_interest(timeline["Europe"])
        fc_asia = forecast_interest(timeline["Asia"])
        future_dates = pd.date_range(timeline["date"].iloc[-1], periods=13, freq="MS")[1:]

        with t1:
            fig_t = go.Figure()
            fig_t.add_scatter(x=timeline["date"], y=timeline["Asia"], name="Asia",
                              line=dict(color=BRAND["red"], width=3))
            fig_t.add_scatter(x=timeline["date"], y=timeline["Europe"], name="Europe",
                              line=dict(color=BRAND["blue"], width=3))
            fig_t.add_scatter(x=future_dates, y=fc_asia, name="Asia (projected)",
                              line=dict(color=BRAND["red"], width=2, dash="dash"))
            fig_t.add_scatter(x=future_dates, y=fc_eu, name="Europe (projected)",
                              line=dict(color=BRAND["blue"], width=2, dash="dash"))
            fig_t.update_layout(title=f"Real search interest - {TREND_KEYWORDS[trend_product]}",
                                yaxis_title="Google Trends score (0-100)", height=420)
            st.plotly_chart(fig_t, use_container_width=True)

        with t2:
            asia_now = float(snapshot.loc[snapshot["continent"] == "Asia",
                                          "current_interest"].iloc[0])
            rows = []
            for _, r in snapshot[snapshot["continent"] != "Asia"].iterrows():
                sc = crossover_score(asia_now, r["current_interest"], timeline["Europe"])
                rows.append({"Continent": r["continent"],
                             "Current interest": r["current_interest"],
                             "Crossover score": sc["score"]})
            prob_df = pd.DataFrame(rows).sort_values("Crossover score", ascending=False)
            st.markdown(f"**Current interest in Asia: {asia_now:.0f}/100**")
            st.dataframe(prob_df, use_container_width=True, hide_index=True, height=260)
            st.caption("Crossover score (0-100) is a heuristic based on the interest gap "
                       "vs Asia and recent momentum - not a calibrated probability. "
                       "One representative country per continent is sampled to respect "
                       "API rate limits.")
    else:
        sim = trend_timeline(trend_product)
        with t1:
            fig_t = go.Figure()
            for region, color in [("asia", BRAND["red"]), ("europe", BRAND["blue"])]:
                h = sim[sim["period"] == "history"]
                f = sim[sim["period"] == "forecast"]
                fig_t.add_scatter(x=h["month"], y=h[region], name=region.title(),
                                  line=dict(color=color, width=3))
                fig_t.add_scatter(x=f["month"], y=f[region], name=f"{region.title()} (projected)",
                                  line=dict(color=color, width=2, dash="dash"))
            fig_t.update_layout(title="Simulated monthly profit (diffusion model)",
                                xaxis_title="Months (0 = today)", yaxis_title="$ per month",
                                height=420)
            st.plotly_chart(fig_t, use_container_width=True)
        with t2:
            st.dataframe(crossover_probability(trend_product), use_container_width=True,
                         hide_index=True, height=260)

st.caption("Built by Benjamin Muller. Demand modeling (Gradient Boosting) + price optimization. "
           "Live trends: Google Trends. Sales and market data: simulated with known "
           "ground-truth elasticities (see README).")
