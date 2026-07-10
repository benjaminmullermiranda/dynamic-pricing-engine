"""Dynamic Pricing Engine - interactive Streamlit demo."""

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
.tint-purple { background: rgba(162,89,255,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-green  { background: rgba(10,207,131,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-blue   { background: rgba(26,188,254,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-orange { background: rgba(242,78,30,.10);  border-radius: 16px; padding: 6px 16px; display: inline-block; }
.stProgress > div > div > div > div { background-color: #0acf83; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the demand model (first load only)...")
def load_model():
    df = generate_dataset()
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

# ---------------- Header ----------------
st.title("Dynamic Pricing Engine")
st.markdown(
    "A machine learning model learns how demand reacts to price, "
    "then finds the price that makes you the most money."
)

# ---------------- Optimal price ----------------
obj_col = "expected_profit" if objective == "Profit" else "expected_revenue"
price_range = (p["base_price"] * 0.6, p["base_price"] * 1.4)
result = optimize_price(pipe, product, price_range, context, objective=obj_col)
elasticity = estimate_elasticity(pipe, product, p["base_price"], context)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best price", f"${result['optimal_price']:.2f}",
          f"{(result['optimal_price'] / p['base_price'] - 1) * 100:+.1f}% vs base price")
c2.metric("Expected daily sales", f"{result['expected_units']:.0f} units")
c3.metric(f"Expected daily {objective.lower()}", f"${result[obj_col]:,.0f}")
c4.metric("Price sensitivity", f"{elasticity:.2f}",
          "high" if abs(elasticity) > 1 else "low", delta_color="off")

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

# ---------------- Market analysis ----------------
st.markdown(f'<h3 class="tint-blue">Market overview - {continent}</h3>', unsafe_allow_html=True)

market = market_summary(product, p["base_price"], continent)

map_col, info_col = st.columns([1.2, 1])
with map_col:
    fig_map = go.Figure(go.Choropleth(
        locations=CONTINENTS[continent]["countries"],
        z=[1] * len(CONTINENTS[continent]["countries"]),
        colorscale=[[0, BRAND["purple"]], [1, BRAND["purple"]]],
        showscale=False, marker_line_color="white", marker_line_width=0.5,
    ))
    fig_map.update_geos(showcountries=True, countrycolor="#666", showland=True,
                        landcolor="#e8e8e8", fitbounds="locations",
                        projection_type="natural earth")
    fig_map.update_layout(title=f"Your market: {continent}", height=380,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with info_col:
    m1, m2 = st.columns(2)
    m1.metric("Average market price", f"${market['avg_market_price']:.2f}")
    m2.metric("Active sellers", f"{market['sellers']:,}")

    sat_labels = {"Saturado": "Saturated - many sellers competing",
                  "Competencia moderada": "Moderate competition",
                  "Oportunidad": "Open opportunity - few sellers"}
    st.markdown(f"**Market saturation:** {sat_labels.get(market['saturation_level'], market['saturation_level'])}")
    st.progress(market["saturation_score"])
    st.caption("The fuller the bar, the harder it is to stand out in this market.")

# ---------------- Profit calculator ----------------
st.markdown('<h3 class="tint-green">Profit calculator</h3>', unsafe_allow_html=True)
st.markdown("Enter your real prices and see your estimated profit.")

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

comp_verdict = competitiveness(sell_price, market["avg_market_price"], market["saturation_score"])
verdict_en = {"Muy competitivo": "Very competitive", "Competitivo": "Competitive",
              "Poco competitivo": "Weak", "No competitivo": "Not competitive"}

r1, r2, r3, r4 = st.columns(4)
r1.metric("Profit per unit", f"${margin_unit:,.2f}", f"{margin_pct:.1f}% margin")
r2.metric("Estimated daily sales", f"{est_units:.0f} units",
          help="Predicted by the ML model using your price and the competitor's price")
r3.metric("Estimated monthly profit", f"${daily_profit * 30:,.0f}",
          f"${daily_profit:,.0f} per day")
r4.metric("Your price vs competitor", f"{vs_comp:+.1f}%",
          "cheaper" if vs_comp < 0 else "more expensive", delta_color="off")

if margin_unit <= 0:
    st.error("You are selling below your purchase cost: you lose money on every sale.")
elif margin_pct < 15:
    st.warning("Thin margin (below 15%). Remember fees, shipping and ads still need to be paid.")
else:
    st.success(f"Healthy margin. Competitiveness in {continent}: "
               f"{verdict_en.get(comp_verdict['verdict'], comp_verdict['verdict'])} "
               f"({comp_verdict['diff_pct']:+.1f}% vs average market price).")

# ---------------- Trend radar ----------------
st.markdown('<h3 class="tint-purple">Trend radar: Asia to the world</h3>', unsafe_allow_html=True)
st.markdown("Products already popular in Asia that have not taken off elsewhere yet. "
            "Real Google Trends data, refreshed daily, with a 12-month projection.")

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
               "Showing the diffusion model simulation instead.")

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
        asia_now = float(snapshot.loc[snapshot["continente"] == "Asia", "interes_actual"].iloc[0])
        rows = []
        for _, r in snapshot[snapshot["continente"] != "Asia"].iterrows():
            sc = crossover_score(asia_now, r["interes_actual"], timeline["Europe"])
            rows.append({"Continent": r["continente"],
                         "Current interest": r["interes_actual"],
                         "Chance of trending (%)": sc["probabilidad_%"]})
        prob_df = pd.DataFrame(rows).sort_values("Chance of trending (%)", ascending=False)
        st.markdown(f"**Current interest in Asia: {asia_now:.0f}/100**")
        st.dataframe(prob_df, use_container_width=True, hide_index=True, height=260)
        st.caption("Estimated from the interest gap vs Asia and each market's recent momentum.")
else:
    sim = trend_timeline(trend_product)
    with t1:
        fig_t = go.Figure()
        for region, color in [("asia", BRAND["red"]), ("europe", BRAND["blue"])]:
            h = sim[sim["period"] == "histórico"]
            f = sim[sim["period"] == "proyección"]
            fig_t.add_scatter(x=h["month"], y=h[region], name=f"{region.title()}",
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

# ---------------- AI business advisor ----------------
st.markdown('<h3 class="tint-orange">AI business advisor</h3>', unsafe_allow_html=True)
st.markdown("Describe your business plan and get the pros, cons and a recommendation, "
            "using the market data on this page.")


def get_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    return key


api_key = get_api_key()
if not api_key:
    api_key = st.text_input("Anthropic API key (not stored)", type="password",
                            help="Get one at console.anthropic.com. On Streamlit Cloud you can "
                                 "set it once in Settings > Secrets as ANTHROPIC_API_KEY.")

plan_text = st.text_area(
    "Your business plan",
    placeholder="Example: I want to import air fryers from China and sell them in Spain "
                "for 65 EUR on Amazon FBA, starting with 5,000 EUR...",
    height=140,
)

if st.button("Analyze my plan", type="primary"):
    if not api_key:
        st.error("Please enter an API key (or set ANTHROPIC_API_KEY in Streamlit Secrets).")
    elif not plan_text.strip():
        st.warning("Please write your plan first.")
    else:
        from src.advisor.business_advisor import analyze_plan

        market_ctx = (
            f"Selected continent: {continent}. Product analyzed: {product}. "
            f"Average market price: ${market['avg_market_price']}. "
            f"User's selling price: ${sell_price}. User's purchase cost: ${buy_price}. "
            f"Market saturation: {market['saturation_level']} ({market['saturation_score']}). "
            f"Trending product viewed: {trend_product}."
        )
        with st.spinner("Analyzing your plan..."):
            try:
                st.markdown(analyze_plan(api_key, plan_text, market_ctx))
            except Exception as e:
                st.error(f"The AI request failed: {e}")

st.caption("Built by Benjamin Muller. Demand modeling (Gradient Boosting) + price optimization. "
           "Live trends: Google Trends. Sales data: simulated.")
