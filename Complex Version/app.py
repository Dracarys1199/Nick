

# app.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from backend import (
    get_model_hub, REGION_COORDS, IMD_COLORS,
    fetch_openweather_all
)

st.set_page_config(page_title="Chennai - IMD Alert System (Frontend)", layout="wide", page_icon="🌦️")

st.title("🌧️ Chennai — IMD Alert System (Frontend)")
st.caption("Cached backend with @st.cache_resource so you **don’t** retrain on every rerun.")

# -----------------------------
# Sidebar controls (frontend only)
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    data_dir = st.text_input("Data folder", value="data")
    flood_threshold = st.slider("Flood threshold (mm)", 100, 500, 200, step=10)

    st.markdown("---")
    st.subheader("Graph + Training")
    use_attention = st.checkbox("Use GAT attention (fallback to GCN)", value=True)
    k_knn = st.slider("K for static KNN graph", 2, 8, 4)
    hidden_dim = st.slider("Hidden dim (GNN)", 16, 256, 64, step=16)
    emb_dim = st.slider("Embedding dim", 16, 256, 64, step=16)

    st.markdown("---")
    st.subheader("Dynamic Edge Weights")
    alpha = st.slider("α (distance inverse)", 0.0, 2.0, 0.6, 0.05)
    beta  = st.slider("β (wind alignment)", 0.0, 2.0, 0.3, 0.05)
    gamma = st.slider("γ (pressure difference)", 0.0, 2.0, 0.1, 0.05)

    st.markdown("---")
    st.subheader("Training (only on demand)")
    epochs = st.slider("Encoder epochs", 20, 300, 80, step=10)
    lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    alpha_smooth = st.slider("Graph smoothness weight", 0.0, 5.0, 1.0, 0.1)
    alpha_flood  = st.slider("Flood loss weight", 0.0, 5.0, 1.0, 0.1)
    alpha_rain   = st.slider("Rain loss weight", 0.0, 5.0, 1.0, 0.1)
    alpha_alert  = st.slider("Alert loss weight", 0.0, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Ablations")
    ablate_dynamic_edges = st.checkbox("Disable dynamic edges", value=False)
    ablate_openweather  = st.checkbox("Disable OpenWeather (zeros)", value=False)

    st.markdown("---")
    retrain_now = st.button("Train / Retrain models now (uses backend cache)")

    if retrain_now:
        with st.spinner("Retraining models..."):
            hub.retrain()
        st.success("Models retrained successfully!")
# -----------------------------
# Cache OpenWeather (frontend)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_forecast_improved(api_key: str):
    """Improved OpenWeather API access with retry logic"""
    if not api_key or api_key.strip() == "":
        st.warning("OpenWeather API key not found. Using demo data.")
        return {r: {'daily':[{'temp':25+np.random.normal(0,2),'humidity':70+np.random.normal(0,5),
                             'wind_speed':5+np.random.normal(0,2),'wind_deg':180+np.random.normal(0,30),
                             'pressure':1013+np.random.normal(0,10),'rain':np.random.exponential(0.5)}]*7,
                   'current':{'wind_speed':5,'wind_deg':180,'pressure':1013,'rain':0}} 
               for r in REGION_COORDS}
    
    try:
        from backend import fetch_openweather_with_retry
        return fetch_openweather_with_retry(api_key, REGION_COORDS)
    except Exception as e:
        st.error(f"OpenWeather API error: {str(e)}")
        return {r: {'daily':[{'temp':25,'humidity':70,'wind_speed':5,'wind_deg':180,'pressure':1013,'rain':0}]*7,
               'current':{'wind_speed':5,'wind_deg':180,'pressure':1013,'rain':0}} 
               for r in REGION_COORDS}

# -----------------------------
# Forecast inputs
# -----------------------------
st.subheader("Forecast Input")
c1, c2, c3 = st.columns(3)
with c1:
    temperature = st.number_input("Temperature (°C)", value=28.0)
    humidity = st.number_input("Humidity (%)", value=75.0)
    date_input = st.date_input("Forecast start date", value=datetime.now().date())
with c2:
    pressure = st.number_input("Pressure (hPa)", value=1008.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=8.0)
    visibility = st.number_input("Visibility (km)", value=10.0)
with c3:
    cloud_cover = st.number_input("Cloud Cover (%)", value=60.0)
    submit = st.button("Fetch 7-day Forecast & Predict")

# -----------------------------
# Predict path
# -----------------------------
# Initialize backend model hub
hub = get_model_hub()
hub.ensure_loaded(
    data_dir=data_dir,
    rainfall_threshold=flood_threshold,
    use_attention=use_attention,
    hidden_dim=hidden_dim,
    emb_dim=emb_dim,
    k_knn=k_knn
)

if submit:
    api_key = os.environ.get('OPENWEATHER_API_KEY', None)
    if api_key is None and not ablate_openweather:
        st.error("Set OPENWEATHER_API_KEY or enable 'Disable OpenWeather'.")
        st.stop()

    with st.spinner("Fetching OpenWeather…"):
        node_weather = get_forecast(api_key) if not ablate_openweather else {
            r:{'daily':[{'temp':0,'humidity':0,'wind_speed':0,'wind_deg':0,'pressure':1013,'rain':0}]*7,'current':{}} 
            for r in REGION_COORDS
        }

    # map UI inputs → feature_cols
    input_d = {}
    for c in hub.sys.feature_cols:
        if c in ['temperature','temp','t']: input_d[c] = temperature
        elif c in ['humidity','rh']:        input_d[c] = humidity
        elif c in ['pressure','p']:         input_d[c] = pressure
        elif c in ['wind_speed','ws']:      input_d[c] = wind_speed
        elif c in ['visibility']:           input_d[c] = visibility
        elif c in ['cloud_cover','clouds']: input_d[c] = cloud_cover
        else:                               input_d[c] = 0.0

    with st.spinner("Predicting (backend cached models)…"):
        preds, influences = hub.predict(
            input_d, node_weather, alpha=alpha, beta=beta, gamma=gamma,
            ablate_dynamic_edges=ablate_dynamic_edges
        )

    days = list(range(1,8))
    df_fore = pd.DataFrame({
        'day': days,
        'date': [date_input + timedelta(days=i) for i in range(1,8)],
        'q10_mm': preds['q10'],
        'median_mm': preds['median'],
        'q90_mm': preds['q90'],
        'flood_prob': preds['flood_prob'],
        'flood_pred': preds['flood_pred'],
        'alert_class': preds['alert_class'],
        'alert_prob': preds['alert_prob']
    })

    # Rainfall chart
    st.subheader("7-day Rainfall Forecast (q10 / q50 / q90)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df_fore['date'], df_fore['median_mm'], marker='o', label='Median (q50)')
    ax.fill_between(df_fore['date'], df_fore['q10_mm'], df_fore['q90_mm'], alpha=0.2, label='80% PI')
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)

    # Flood prob chart
    st.subheader("Flood Probability (next 7 days)")
    fig2, ax2 = plt.subplots(figsize=(8,2.8))
    ax2.plot(df_fore['date'], df_fore['flood_prob'], marker='o')
    ax2.set_ylim(0,1); ax2.set_ylabel("Probability"); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # IMD label chips
    def color_for_alert(c): return IMD_COLORS.get(int(c), ("Unknown","#888"))[1]
    chips = []
    for _,row in df_fore.iterrows():
        lbl, col = IMD_COLORS.get(int(row['alert_class']), ("Unknown","#888"))
        chips.append((row['date'].strftime('%Y-%m-%d'), lbl, f"{row['alert_prob']:.0%}", col))
    st.markdown("**Daily IMD Alerts**")
    st.markdown("".join(
        f"<span style='background:{c};color:#fff;padding:6px 10px;border-radius:8px;margin:4px;display:inline-block'>{d}: {l} ({p})</span>"
        for d,l,p,c in chips
    ), unsafe_allow_html=True)

    # Influence heatmaps (day 1 & 4)
    st.subheader("Graph Influence Weights (dynamic edges)")
    for dsel in [0,3]:
        mat = influences[dsel]
        figm, axm = plt.subplots(figsize=(6,5))
        im = axm.imshow(mat, aspect='auto')
        axm.set_title(f"Day +{dsel+1}")
        axm.set_xticks(range(len(hub.sys.regions))); axm.set_yticks(range(len(hub.sys.regions)))
        axm.set_xticklabels(hub.sys.regions, rotation=90, fontsize=8)
        axm.set_yticklabels(hub.sys.regions, fontsize=8)
        figm.colorbar(im, ax=axm, shrink=0.8)
        st.pyplot(figm)

    # Simple alerting hook
    highrisk = [(r['date'], r['flood_prob']) for _,r in df_fore.iterrows() if r['flood_prob']>=0.6]
    if highrisk:
        st.warning("⚠️ High flood risk days detected (prob ≥ 0.60).")
        st.write(highrisk)
        # TODO: wire webhook/email here if needed.

# -------- Footer --------
st.markdown("---")
st.caption("Models & scalers are cached via @st.cache_resource in backend.py → no repeated retraining unless you click Retrain.")
