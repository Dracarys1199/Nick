# app.py  –  Chennai IMD Alert System  (fixed + redesigned)
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from backend import (
    get_model_hub, REGION_COORDS, IMD_COLORS,
    fetch_openweather_all,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chennai IMD Alert System",
    layout="wide",
    page_icon="🌧️",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --surface2:  #1a2236;
    --accent1:   #00d4ff;   /* electric cyan */
    --accent2:   #ff6b35;   /* storm orange  */
    --accent3:   #7c3aed;   /* deep violet   */
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #22c55e;
    --danger:    #ef4444;
    --warning:   #f59e0b;
    --radius:    12px;
    --glow:      0 0 24px rgba(0,212,255,.20);
}

/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid rgba(0,212,255,.12) !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Main title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent1) 0%, var(--accent3) 60%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 0;
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: var(--surface2);
    border: 1px solid rgba(0,212,255,.10);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--glow);
}

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent1);
    margin-bottom: 0.5rem;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-tile {
    flex: 1; min-width: 130px;
    background: var(--surface);
    border: 1px solid rgba(0,212,255,.12);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-tile .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent1);
    line-height: 1.1;
}
.metric-tile .lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Alert chips ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 0.75rem 0; }
.chip {
    padding: 6px 14px;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    border: 1px solid rgba(255,255,255,.08);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), var(--accent3)) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

.stSlider > div > div > div { background: var(--accent1) !important; }
.stSelectSlider > div { color: var(--accent1) !important; }

label, .stTextInput label, .stNumberInput label,
.stCheckbox label, .stSlider label { color: var(--text) !important; }

/* ── Divider ── */
hr { border-color: rgba(0,212,255,.10) !important; }

/* ── Warning / error banners ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Sidebar subheaders ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent1) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}

/* ── Number inputs ── */
input[type=number], input[type=text] {
    background: var(--surface2) !important;
    border: 1px solid rgba(0,212,255,.2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── pyplot backgrounds ── */
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib dark theme
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#1a2236",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#1e3a5f",
    "text.color":        "#e2e8f0",
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.2,
})

CYAN   = "#00d4ff"
ORANGE = "#ff6b35"
VIOLET = "#7c3aed"

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([0.07, 0.93])
with col_logo:
    st.markdown("<div style='font-size:3rem;line-height:1;padding-top:6px'>🌧️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div class='hero-title'>Chennai — IMD Alert System</div>
    <div class='hero-sub'>Graph-Neural-Network · 7-Day Flood Forecast · OpenWeather Live</div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:0.8rem 0 1.2rem'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Initialize backend hub FIRST (fixes: hub used before definition)
# ──────────────────────────────────────────────────────────────────────────────
hub = get_model_hub()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:1.5rem;text-align:center;padding:0.5rem 0'>⚙️</div>", unsafe_allow_html=True)
    st.header("Settings")

    data_dir        = st.text_input("Data folder", value="data")
    flood_threshold = st.slider("Flood threshold (mm)", 100, 500, 200, step=10)

    st.markdown("---")
    st.subheader("Graph + Training")
    use_attention = st.checkbox("Use GAT attention (fallback → GCN)", value=True)
    k_knn         = st.slider("K — static KNN graph", 2, 8, 4)
    hidden_dim    = st.slider("Hidden dim (GNN)", 16, 256, 64, step=16)
    emb_dim       = st.slider("Embedding dim", 16, 256, 64, step=16)

    st.markdown("---")
    st.subheader("Dynamic Edge Weights")
    alpha = st.slider("α — distance inverse",    0.0, 2.0, 0.6, 0.05)
    beta  = st.slider("β — wind alignment",      0.0, 2.0, 0.3, 0.05)
    gamma = st.slider("γ — pressure difference", 0.0, 2.0, 0.1, 0.05)

    st.markdown("---")
    st.subheader("Training")
    epochs       = st.slider("Encoder epochs",       20, 300,  80, step=10)
    lr           = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    alpha_smooth = st.slider("Graph smoothness weight", 0.0, 5.0, 1.0, 0.1)
    alpha_flood  = st.slider("Flood loss weight",       0.0, 5.0, 1.0, 0.1)
    alpha_rain   = st.slider("Rain loss weight",        0.0, 5.0, 1.0, 0.1)
    alpha_alert  = st.slider("Alert loss weight",       0.0, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Ablations")
    ablate_dynamic_edges = st.checkbox("Disable dynamic edges",        value=False)
    ablate_openweather   = st.checkbox("Disable OpenWeather (zeros)",  value=False)

    st.markdown("---")
    retrain_now = st.button("🔄  Train / Retrain models")

    if retrain_now:
        # hub already initialised above — safe to call
        with st.spinner("Retraining… this may take a minute."):
            hub.retrain()
        st.success("✅ Models retrained successfully!")

# ──────────────────────────────────────────────────────────────────────────────
# Load/ensure backend
# ──────────────────────────────────────────────────────────────────────────────
hub.ensure_loaded(
    data_dir=data_dir,
    rainfall_threshold=flood_threshold,
    use_attention=use_attention,
    hidden_dim=hidden_dim,
    emb_dim=emb_dim,
    k_knn=k_knn,
)

# ──────────────────────────────────────────────────────────────────────────────
# OpenWeather cache (frontend)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_forecast_cached(api_key: str):
    """Fetch 7-day OpenWeather forecast with retry; fall back to synthetic data."""
    if not api_key or not api_key.strip():
        st.warning("⚠️ OPENWEATHER_API_KEY not set — using synthetic demo data.")
        return _synthetic_weather()
    try:
        from backend import fetch_openweather_with_retry
        return fetch_openweather_with_retry(api_key, REGION_COORDS)
    except Exception as e:
        st.error(f"OpenWeather API error: {e}")
        return _synthetic_weather()

def _synthetic_weather():
    rng = np.random.default_rng(42)
    return {
        r: {
            "daily": [
                {
                    "temp":       float(25 + rng.normal(0, 2)),
                    "humidity":   float(70 + rng.normal(0, 5)),
                    "wind_speed": float( 5 + rng.normal(0, 2)),
                    "wind_deg":   float(180 + rng.normal(0, 30)),
                    "pressure":   float(1013 + rng.normal(0, 10)),
                    "rain":       float(rng.exponential(0.5)),
                }
                for _ in range(7)
            ],
            "current": {"wind_speed": 5, "wind_deg": 180, "pressure": 1013, "rain": 0},
        }
        for r in REGION_COORDS
    }

# ──────────────────────────────────────────────────────────────────────────────
# Forecast input panel
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>🌡️ Forecast Inputs</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature (°C)",  value=28.0)
        humidity    = st.number_input("Humidity (%)",      value=75.0)
        date_input  = st.date_input("Forecast start date", value=datetime.now().date())
    with c2:
        pressure   = st.number_input("Pressure (hPa)",    value=1008.0)
        wind_speed = st.number_input("Wind Speed (m/s)",  value=8.0)
        visibility = st.number_input("Visibility (km)",   value=10.0)
    with c3:
        cloud_cover = st.number_input("Cloud Cover (%)", value=60.0)
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        submit = st.button("🔮  Fetch Forecast & Predict")

# ──────────────────────────────────────────────────────────────────────────────
# Prediction path
# ──────────────────────────────────────────────────────────────────────────────
if submit:
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")

    if not api_key and not ablate_openweather:
        st.error("🔑 Set the OPENWEATHER_API_KEY env var, or enable **Disable OpenWeather** in the sidebar.")
        st.stop()

    with st.spinner("Fetching weather data…"):
        if ablate_openweather:
            node_weather = {
                r: {
                    "daily": [{"temp": 0, "humidity": 0, "wind_speed": 0,
                               "wind_deg": 0, "pressure": 1013, "rain": 0}] * 7,
                    "current": {},
                }
                for r in REGION_COORDS
            }
        else:
            # BUG FIX: was calling undefined get_forecast(); now calls get_forecast_cached()
            node_weather = get_forecast_cached(api_key)

    # Map UI inputs → feature_cols
    input_d: dict = {}
    for c in hub.sys.feature_cols:
        if   c in ("temperature", "temp", "t"):   input_d[c] = temperature
        elif c in ("humidity", "rh"):              input_d[c] = humidity
        elif c in ("pressure", "p"):               input_d[c] = pressure
        elif c in ("wind_speed", "ws"):            input_d[c] = wind_speed
        elif c == "visibility":                    input_d[c] = visibility
        elif c in ("cloud_cover", "clouds"):       input_d[c] = cloud_cover
        else:                                      input_d[c] = 0.0

    with st.spinner("Running GNN + ensemble predictions…"):
        preds, influences = hub.predict(
            input_d, node_weather,
            alpha=alpha, beta=beta, gamma=gamma,
            ablate_dynamic_edges=ablate_dynamic_edges,
        )

    days = list(range(1, 8))
    df_fore = pd.DataFrame({
        "day":         days,
        "date":        [date_input + timedelta(days=i) for i in range(1, 8)],
        "q10_mm":      preds["q10"],
        "median_mm":   preds["median"],
        "q90_mm":      preds["q90"],
        "flood_prob":  preds["flood_prob"],
        "flood_pred":  preds["flood_pred"],
        "alert_class": preds["alert_class"],
        "alert_prob":  preds["alert_prob"],
    })

    # ── Summary metric row ────────────────────────────────────────────────────
    max_rain    = df_fore["median_mm"].max()
    max_flood_p = df_fore["flood_prob"].max()
    high_days   = int((df_fore["flood_prob"] >= 0.6).sum())
    peak_day    = df_fore.loc[df_fore["median_mm"].idxmax(), "date"].strftime("%b %d")

    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-tile'>
        <div class='val'>{max_rain:.1f}</div>
        <div class='lbl'>Peak Rain (mm)</div>
      </div>
      <div class='metric-tile'>
        <div class='val' style='color:#ff6b35'>{max_flood_p:.0%}</div>
        <div class='lbl'>Max Flood Prob</div>
      </div>
      <div class='metric-tile'>
        <div class='val' style='color:{"#ef4444" if high_days else "#22c55e"}'>{high_days}</div>
        <div class='lbl'>High-Risk Days</div>
      </div>
      <div class='metric-tile'>
        <div class='val' style='font-size:1.2rem'>{peak_day}</div>
        <div class='lbl'>Peak Date</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── IMD Alert chips ───────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>🚨 Daily IMD Alerts</div>", unsafe_allow_html=True)
    chips_html = "<div class='chip-row'>"
    for _, row in df_fore.iterrows():
        lbl, col = IMD_COLORS.get(int(row["alert_class"]), ("Unknown", "#888"))
        d_str    = row["date"].strftime("%b %d")
        prob_str = f"{row['alert_prob']:.0%}"
        chips_html += (
            f"<span class='chip' style='background:{col}22;border-color:{col}55;color:#fff'>"
            f"<span style='color:{col}'>●</span> {d_str} · {lbl} <span style='opacity:.7'>({prob_str})</span>"
            f"</span>"
        )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # ── Rainfall forecast chart ───────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:1.5rem'>📈 7-Day Rainfall Forecast</div>",
                unsafe_allow_html=True)

    dates = df_fore["date"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(dates, df_fore["q10_mm"], df_fore["q90_mm"],
                    color=CYAN, alpha=0.10, label="80 % prediction interval")
    ax.plot(dates, df_fore["q10_mm"],   color=CYAN,   alpha=0.35, lw=1.2, ls="--")
    ax.plot(dates, df_fore["q90_mm"],   color=CYAN,   alpha=0.35, lw=1.2, ls="--")
    ax.plot(dates, df_fore["median_mm"], color=CYAN,  lw=2.8,  marker="o",
            markersize=7, markerfacecolor="#0a0f1e", markeredgewidth=2, label="Median (q50)")
    ax.set_ylabel("Rainfall (mm)", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#111827", edgecolor="#1e3a5f", labelcolor="#94a3b8")
    fig.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close(fig)

    # ── Flood probability chart ───────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:1rem'>💧 Flood Probability</div>",
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(11, 2.8))
    probs = df_fore["flood_prob"].tolist()
    colors_bar = [ORANGE if p >= 0.6 else CYAN for p in probs]
    bars = ax2.bar(dates, probs, color=colors_bar, alpha=0.80, width=0.55, zorder=3)
    ax2.axhline(0.6, color=ORANGE, lw=1.4, ls="--", alpha=0.7, label="Risk threshold (0.60)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Probability", fontsize=9)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.grid(True, alpha=0.18, axis="y")
    ax2.legend(fontsize=8, facecolor="#111827", edgecolor="#1e3a5f", labelcolor="#94a3b8")
    fig2.tight_layout(pad=0.8)
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Combined table ────────────────────────────────────────────────────────
    with st.expander("📋 Full Forecast Table", expanded=False):
        display_df = df_fore.copy()
        display_df["date"] = display_df["date"].astype(str)
        display_df["flood_prob"] = display_df["flood_prob"].map("{:.1%}".format)
        display_df["alert_prob"] = display_df["alert_prob"].map("{:.1%}".format)
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Influence heatmaps ────────────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:1.5rem'>🕸️ Graph Influence Weights (dynamic edges)</div>",
                unsafe_allow_html=True)
    region_labels = hub.sys.regions
    cols_hm = st.columns(2)
    for idx, dsel in enumerate([0, 3]):
        mat = influences[dsel]
        fig_h, ax_h = plt.subplots(figsize=(5.8, 4.8))
        im = ax_h.imshow(mat, cmap="viridis", aspect="auto")
        ax_h.set_title(f"Day +{dsel + 1}", fontsize=10, color="#e2e8f0", pad=10)
        ax_h.set_xticks(range(len(region_labels)))
        ax_h.set_yticks(range(len(region_labels)))
        ax_h.set_xticklabels(region_labels, rotation=75, fontsize=6.5)
        ax_h.set_yticklabels(region_labels, fontsize=6.5)
        cb = fig_h.colorbar(im, ax=ax_h, shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors="#94a3b8")
        fig_h.tight_layout(pad=0.6)
        with cols_hm[idx]:
            st.pyplot(fig_h)
        plt.close(fig_h)

    # ── High-risk warning banner ──────────────────────────────────────────────
    highrisk = [(r["date"].strftime("%b %d"), f"{r['flood_prob']:.0%}")
                for _, r in df_fore.iterrows() if r["flood_prob"] >= 0.6]
    if highrisk:
        items = " · ".join([f"{d} {p}" for d, p in highrisk])
        st.warning(f"⚠️ High flood-risk days detected (prob ≥ 60 %): **{items}**")
        # TODO: wire webhook / email alert here

else:
    # ── Placeholder state ─────────────────────────────────────────────────────
    st.markdown("""
    <div class='card' style='text-align:center;padding:2.5rem 1rem;border-color:rgba(0,212,255,.18)'>
      <div style='font-size:3rem;margin-bottom:0.6rem'>🌊</div>
      <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:700;
                  color:#00d4ff;margin-bottom:0.4rem'>
        Ready to Forecast
      </div>
      <div style='color:#64748b;font-size:0.85rem;max-width:480px;margin:0 auto'>
        Adjust the input parameters above and click
        <strong style='color:#e2e8f0'>Fetch Forecast & Predict</strong> to run the
        GNN-ensemble pipeline and get your 7-day IMD alert outlook.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin:2rem 0 0.6rem'>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
    "color:#334155;text-align:center;letter-spacing:0.08em'>"
    "Models cached via @st.cache_resource · No retraining on rerun unless triggered · "
    "Chennai IMD Alert System"
    "</div>",
    unsafe_allow_html=True,
)
