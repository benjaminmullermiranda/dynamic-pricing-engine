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
from src.market.market_data import CONTINENTS, market_summary, competitiveness
from src.market.trend_radar import TRENDING, trend_timeline, crossover_probability
from src.dataset.live_trends import (
    TREND_KEYWORDS, interest_timeline, current_interest_by_continent,
    forecast_interest, crossover_score,
)

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="💰", layout="wide")

# ---------- Playful Color design system ----------
BRAND = {
    "green": "#0acf83", "orange": "#f24e1e", "purple": "#a259ff",
    "red": "#ff7262", "blue": "#1abcfe",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: #0f0f14; }

h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; }

/* metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 12px;
    padding: 20px 24px;
    transition: box-shadow .2s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 8px 24px rgba(15,15,20,0.08); }
[data-testid="stMetricLabel"] { color: #6e6e78; font-weight: 500; }

/* sidebar */
[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e5e5ea;
}

/* buttons & pills */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 500; padding: 10px 18px;
}

/* expander as card */
[data-testid="stExpander"] {
    border: 1px solid #e5e5ea; border-radius: 12px; background: #ffffff;
}

/* section tint bands */
.tint-purple { background: rgba(162,89,255,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-green  { background: rgba(10,207,131,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-blue   { background: rgba(26,188,254,.10); border-radius: 16px; padding: 6px 16px; display: inline-block; }
.tint-orange { background: rgba(242,78,30,.10);  border-radius: 16px; padding: 6px 16px; display: inline-block; }

/* progress bar in brand green */
.stProgress > div > div > div > div { background-color: #0acf83; border-radius: 999px; }
.stProgress > div > div > div { border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training demand model (first load only)...")
def load_model():
    df = generate_dataset()
    pipe, metrics = train(df)
    return df, pipe, metrics


df, pipe, metrics = load_model()

# ---------- Sidebar: market context ----------
st.sidebar.title("⚙️ Market Context")
continent = st.sidebar.selectbox("🌍 Continente", list(CONTINENTS.keys()), index=2)
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
                    line=dict(color=BRAND["blue"], width=3))
    fig.add_vline(x=result["optimal_price"], line_dash="dash", line_color=BRAND["orange"],
                  annotation_text="optimal")
    fig.update_layout(title="Demand curve (model)", xaxis_title="Price ($)",
                      yaxis_title="Units / day", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig2 = go.Figure()
    fig2.add_scatter(x=curve["price"], y=curve["expected_revenue"], name="Revenue",
                     line=dict(color=BRAND["green"], width=3))
    fig2.add_scatter(x=curve["price"], y=curve["expected_profit"], name="Profit",
                     line=dict(color=BRAND["purple"], width=3))
    fig2.add_vline(x=result["optimal_price"], line_dash="dash", line_color=BRAND["orange"])
    fig2.add_vline(x=p["cost"], line_dash="dot", line_color="gray",
                   annotation_text="unit cost")
    fig2.update_layout(title="Revenue & profit vs price", xaxis_title="Price ($)",
                       yaxis_title="$ / day", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- What-if ----------
st.markdown('<h3 class="tint-green">🎯 What-if: prueba tu propio precio</h3>', unsafe_allow_html=True)
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

# ---------- Market analysis by continent ----------
st.markdown(f'<h3 class="tint-blue">🌍 Análisis de mercado — {continent}</h3>', unsafe_allow_html=True)

market = market_summary(product, p["base_price"], continent)
comp = competitiveness(your_price, market["avg_market_price"], market["saturation_score"])

map_col, info_col = st.columns([1.2, 1])

with map_col:
    fig_map = go.Figure(go.Choropleth(
        locations=CONTINENTS[continent]["countries"],
        z=[1] * len(CONTINENTS[continent]["countries"]),
        colorscale=[[0, BRAND["purple"]], [1, BRAND["purple"]]],
        showscale=False,
        marker_line_color="white", marker_line_width=0.5,
    ))
    fig_map.update_geos(
        showcountries=True, countrycolor="#666",
        showland=True, landcolor="#e8e8e8",
        fitbounds="locations", projection_type="natural earth",
    )
    fig_map.update_layout(title=f"Mercado activo: {continent}", height=380,
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with info_col:
    m1, m2 = st.columns(2)
    m1.metric("Precio medio de mercado", f"${market['avg_market_price']:.2f}")
    m2.metric("Tu precio", f"${your_price:.2f}", f"{comp['diff_pct']:+.1f}% vs mercado",
              delta_color="inverse")
    m3, m4 = st.columns(2)
    m3.metric("Vendedores activos", f"{market['sellers']:,}")
    m4.metric("Índice de demanda", f"{market['demand_index']}/100")

    st.markdown(f"**Saturación del mercado:** {market['saturation_emoji']} "
                f"{market['saturation_level']} ({market['saturation_score'] * 100:.0f}%)")
    st.progress(market["saturation_score"])

    st.markdown(f"**¿Eres competitivo?** {comp['emoji']} **{comp['verdict']}**")
    st.caption(comp["note"])

# ---------- Data & model ----------
with st.expander("📊 Historical data sample & model performance"):
    st.markdown(f"**Model (hold-out test):** MAE = {metrics['MAE']} units · "
                f"R² = {metrics['R2']} · MAPE = {metrics['MAPE_%']}% "
                f"({metrics['n_test']:,} test rows)")
    hist = df[df["product"] == product]
    fig3 = go.Figure()
    fig3.add_scatter(x=hist["price"], y=hist["units_sold"], mode="markers",
                     marker=dict(size=4, opacity=0.35, color=BRAND["red"]),
                     name="Historical sales")
    fig3.update_layout(title=f"Observed price vs units sold — {product}",
                       xaxis_title="Price ($)", yaxis_title="Units sold", height=380)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(hist.tail(200), use_container_width=True, height=250)

# ---------- Trend radar: Asia -> world (real Google Trends data) ----------
st.markdown('<h3 class="tint-purple">🚀 Radar de tendencias: Asia → mundo</h3>', unsafe_allow_html=True)
st.markdown("Productos en tendencia en Asia que aún no despegan en otros continentes. "
            "**Datos reales de Google Trends** (actualizados cada 24h) + proyección a 12 meses.")

trend_product = st.selectbox("Producto en tendencia", list(TRENDING.keys()))


@st.cache_data(ttl=86400, show_spinner="Consultando Google Trends...")
def load_live_trends(product_name):
    tl = interest_timeline(product_name, ("Asia", "Europe"))
    snapshot = current_interest_by_continent(product_name)
    return tl, snapshot


live_data_ok = True
try:
    timeline, snapshot = load_live_trends(trend_product)
except Exception:
    live_data_ok = False
    st.warning("⚠️ Google Trends no disponible ahora mismo (límite de peticiones). "
               "Mostrando simulación del modelo de difusión.")

t1, t2 = st.columns([1.4, 1])

if live_data_ok:
    fc_eu = forecast_interest(timeline["Europe"])
    fc_asia = forecast_interest(timeline["Asia"])
    future_dates = pd.date_range(timeline["date"].iloc[-1], periods=13, freq="MS")[1:]

    with t1:
        fig_t = go.Figure()
        fig_t.add_scatter(x=timeline["date"], y=timeline["Asia"], name="Asia (real)",
                          line=dict(color=BRAND["red"], width=3))
        fig_t.add_scatter(x=timeline["date"], y=timeline["Europe"], name="Europa (real)",
                          line=dict(color=BRAND["blue"], width=3))
        fig_t.add_scatter(x=future_dates, y=fc_asia, name="Asia (proyección)",
                          line=dict(color=BRAND["red"], width=2, dash="dash"))
        fig_t.add_scatter(x=future_dates, y=fc_eu, name="Europa (proyección)",
                          line=dict(color=BRAND["blue"], width=2, dash="dash"))
        fig_t.update_layout(title=f"Interés de búsqueda real — {TREND_KEYWORDS[trend_product]}",
                            yaxis_title="Google Trends (0-100)", height=420)
        st.plotly_chart(fig_t, use_container_width=True)

    with t2:
        asia_now = float(snapshot.loc[snapshot["continente"] == "Asia", "interes_actual"].iloc[0])
        rows = []
        for _, r in snapshot[snapshot["continente"] != "Asia"].iterrows():
            # momentum proxy: Europe series (only 2 timelines fetched to respect rate limits)
            sc = crossover_score(asia_now, r["interes_actual"], timeline["Europe"])
            rows.append({"Continente": r["continente"],
                         "Interés actual": r["interes_actual"],
                         "Prob. de tendencia (%)": sc["probabilidad_%"]})
        prob_df = pd.DataFrame(rows).sort_values("Prob. de tendencia (%)", ascending=False)
        st.markdown(f"**Interés actual en Asia: {asia_now:.0f}/100**")
        st.dataframe(prob_df, use_container_width=True, hide_index=True, height=260)
        st.caption("Probabilidad estimada a partir del gap de interés vs Asia "
                   "y el momentum reciente de cada mercado.")
else:
    sim = trend_timeline(trend_product)
    with t1:
        fig_t = go.Figure()
        for region, color in [("asia", BRAND["red"]), ("europe", BRAND["blue"])]:
            h = sim[sim["period"] == "histórico"]
            f = sim[sim["period"] == "proyección"]
            fig_t.add_scatter(x=h["month"], y=h[region], name=f"{region.title()} (histórico)",
                              line=dict(color=color, width=3))
            fig_t.add_scatter(x=f["month"], y=f[region], name=f"{region.title()} (proyección)",
                              line=dict(color=color, width=2, dash="dash"))
        fig_t.update_layout(title="Beneficio mensual simulado (difusión logística)",
                            xaxis_title="Meses (0 = hoy)", yaxis_title="$/mes", height=420)
        st.plotly_chart(fig_t, use_container_width=True)
    with t2:
        st.dataframe(crossover_probability(trend_product), use_container_width=True,
                     hide_index=True, height=260)

# ---------- AI business advisor ----------
st.markdown('<h3 class="tint-orange">🤖 Asesor IA: analiza tu plan de negocio</h3>', unsafe_allow_html=True)
st.markdown("Cuéntale tu plan al asesor y recibirás **pros, contras y una recomendación** "
            "basada en el contexto de mercado actual de esta página.")

plan_text = st.text_area(
    "Tu plan de negocio",
    placeholder="Ej: Quiero importar air fryers desde China y venderlas en España por 65€ "
                "a través de Amazon FBA, con una inversión inicial de 5.000€...",
    height=140,
)

if st.button("Analizar mi plan", type="primary"):
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Falta configurar ANTHROPIC_API_KEY en los Secrets de Streamlit Cloud "
                 "(Settings → Secrets). Ver DEPLOYMENT.md.")
    elif not plan_text.strip():
        st.warning("Escribe tu plan primero.")
    else:
        from src.advisor.business_advisor import analyze_plan

        market_ctx = (
            f"Continente seleccionado: {continent}. Producto analizado: {product}. "
            f"Precio medio de mercado: ${market['avg_market_price']}. "
            f"Precio del usuario: ${your_price}. "
            f"Saturación: {market['saturation_level']} ({market['saturation_score']}). "
            f"Veredicto de competitividad: {comp['verdict']}. "
            f"Producto en tendencia consultado: {trend_product}."
        )
        with st.spinner("Analizando tu plan..."):
            try:
                st.markdown(analyze_plan(api_key, plan_text, market_ctx))
            except Exception as e:
                st.error(f"Error llamando a la API: {e}")

st.caption("Built by Benjamin Muller · Demand modeling (Gradient Boosting) + price optimization · "
           "Tendencias en vivo: Google Trends · Ventas y elasticidades: simuladas.")
