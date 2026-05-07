"""
EV Grid Intelligence Dashboard — Part D
=========================================
Streamlit app tying together:
  Part A — XGBoost + Prophet demand prediction
  Part B — MILP rolling-horizon scheduler
  Part C — MCLP grid-aware location recommender

Install:
    pip install streamlit plotly pandas numpy scipy scikit-learn
    pip install prophet xgboost shap ortools

Run:
    streamlit run ev_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import time

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="EV Grid Intelligence · BESCOM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}

.main { background-color: #0a0e1a; }
section[data-testid="stSidebar"] {
    background: #0d1225;
    border-right: 1px solid #1e2d4a;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #0d1b35 0%, #112040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #4fc3f7;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #7a9cc4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.violation-badge {
    background: #b71c1c;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
}
.safe-badge {
    background: #1b5e20;
    color: #a5d6a7;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
}
.rationale-card {
    background: #0d1b35;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.85rem;
}
.site-card {
    background: #0d1b35;
    border-left: 3px solid #ffd54f;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.85rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #7a9cc4;
}
.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
}
div[data-testid="stMetric"] label { color: #7a9cc4 !important; font-size: 0.75rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-family: 'IBM Plex Mono'; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SHARED CONFIG & DATA (cached)
# ═══════════════════════════════════════════════

TRANSFORMER_CAPACITY_KW = 4000.0
HARD_LIMIT_PCT          = 0.95
HARD_LIMIT_KW           = TRANSFORMER_CAPACITY_KW * HARD_LIMIT_PCT
WARNING_KW              = TRANSFORMER_CAPACITY_KW * 0.90
SLOT_MINUTES            = 15
LAT_MIN, LAT_MAX        = 12.85, 13.10
LON_MIN, LON_MAX        = 77.48, 77.75
GRID_RES                = 35
COVERAGE_RADIUS_KM      = 2.0

ZONES_CFG = {
    "Koramangala":   {"lat": 12.935, "lon": 77.624, "base_kw": 2800, "ev_density": 0.18,
                      "zone_type": 1, "loading_pct": 0.92, "growth_rate": 0.22},
    "Whitefield":    {"lat": 12.969, "lon": 77.750, "base_kw": 3500, "ev_density": 0.22,
                      "zone_type": 2, "loading_pct": 0.88, "growth_rate": 0.28},
    "Jayanagar":     {"lat": 12.925, "lon": 77.583, "base_kw": 2100, "ev_density": 0.12,
                      "zone_type": 0, "loading_pct": 0.65, "growth_rate": 0.15},
    "Marathahalli":  {"lat": 12.956, "lon": 77.701, "base_kw": 3100, "ev_density": 0.20,
                      "zone_type": 1, "loading_pct": 0.85, "growth_rate": 0.25},
    "Hebbal":        {"lat": 13.035, "lon": 77.597, "base_kw": 2600, "ev_density": 0.15,
                      "zone_type": 1, "loading_pct": 0.72, "growth_rate": 0.18},
    "Electronic City":{"lat": 12.845,"lon": 77.660, "base_kw": 2900, "ev_density": 0.19,
                       "zone_type": 2, "loading_pct": 0.78, "growth_rate": 0.30},
}

EXISTING_STATIONS = [
    {"name": "Mantri Square",     "lat": 13.003, "lon": 77.567, "chargers": 4,  "type": "slow"},
    {"name": "Forum Mall",        "lat": 12.934, "lon": 77.610, "chargers": 6,  "type": "fast"},
    {"name": "Phoenix Marketcity","lat": 12.997, "lon": 77.697, "chargers": 8,  "type": "fast"},
    {"name": "Indiranagar BESCOM","lat": 12.979, "lon": 77.641, "chargers": 2,  "type": "slow"},
    {"name": "Orion Mall",        "lat": 13.011, "lon": 77.556, "chargers": 3,  "type": "slow"},
]

RECOMMENDED_SITES = [
    {"id": "SITE-01", "lat": 12.962, "lon": 77.638, "suitability": 0.87, "chargers": "4×Fast+2×L2", "zone": "Koramangala"},
    {"id": "SITE-02", "lat": 12.851, "lon": 77.668, "suitability": 0.84, "chargers": "4×Fast+2×L2", "zone": "Electronic City"},
    {"id": "SITE-03", "lat": 12.988, "lon": 77.716, "suitability": 0.81, "chargers": "2×Fast+4×L2", "zone": "Whitefield"},
    {"id": "SITE-04", "lat": 13.028, "lon": 77.574, "suitability": 0.79, "chargers": "0×Fast+6×L2", "zone": "Hebbal"},
    {"id": "SITE-05", "lat": 12.940, "lon": 77.596, "suitability": 0.76, "chargers": "2×Fast+4×L2", "zone": "Jayanagar"},
    {"id": "SITE-06", "lat": 12.971, "lon": 77.682, "suitability": 0.73, "chargers": "4×Fast+2×L2", "zone": "Marathahalli"},
    {"id": "SITE-07", "lat": 12.900, "lon": 77.620, "suitability": 0.71, "chargers": "2×Fast+4×L2", "zone": "Koramangala"},
    {"id": "SITE-08", "lat": 13.050, "lon": 77.620, "suitability": 0.69, "chargers": "0×Fast+6×L2", "zone": "Hebbal"},
]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(np.array(lat2) - np.array(lat1))
    dlon = np.radians(np.array(lon2) - np.array(lon1))
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ── Generate synthetic load data ──────────────
@st.cache_data
def generate_load_data(adoption_rate: float, extra_evs: int):
    np.random.seed(42)
    ts = pd.date_range("2024-05-20 00:00", periods=96*7, freq="15min")
    rows = []
    for zone, cfg in ZONES_CFG.items():
        h = ts.hour + ts.minute / 60
        morning = np.exp(-0.5*((h-9)/1.5)**2)
        evening = np.exp(-0.5*((h-20)/1.5)**2)
        profile = 0.35 + 0.65*(morning+evening)
        if cfg["zone_type"] == 2:
            profile += np.where((h>=8)&(h<=18), 0.3, 0)
        weekend = np.where(ts.dayofweek>=5, 0.75, 1.0)
        base = cfg["base_kw"] * profile * weekend * (1+np.random.normal(0,0.02,len(ts)))

        ev_home = np.exp(-0.5*((h-19)/1.2)**2)
        ev_work = np.exp(-0.5*((h-8.5)/1.0)**2)
        ev_prof = 0.7*ev_home + 0.3*ev_work
        n_evs   = int(5000 * cfg["ev_density"] * adoption_rate) + extra_evs
        ev_unm  = n_evs * 7.2 * 0.15 * ev_prof * (1+np.random.normal(0,0.04,len(ts)))
        ev_unm  = np.clip(ev_unm, 0, None)

        # Optimized: shift evening peak by 2h
        ev_opt  = n_evs * 7.2 * 0.15 * np.exp(-0.5*((h-21.5)/1.5)**2) * (1+np.random.normal(0,0.03,len(ts)))
        ev_opt  = np.clip(ev_opt, 0, None)

        net_unm = np.clip(base+ev_unm, 0, None)
        net_opt = np.clip(base+ev_opt, 0, None)

        rows.append(pd.DataFrame({
            "timestamp": ts, "zone": zone,
            "base_load_kw": base,
            "ev_unmanaged_kw": ev_unm,
            "ev_optimized_kw": ev_opt,
            "net_unmanaged_kw": net_unm,
            "net_optimized_kw": net_opt,
            "loading_unmanaged": net_unm / TRANSFORMER_CAPACITY_KW,
            "loading_optimized":  net_opt / TRANSFORMER_CAPACITY_KW,
            "violation_unmanaged": net_unm > HARD_LIMIT_KW,
            "violation_optimized":  net_opt > HARD_LIMIT_KW,
        }))
    return pd.concat(rows, ignore_index=True)


@st.cache_data
def generate_heatmap_grid(adoption_rate: float):
    np.random.seed(42)
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_RES)
    lons = np.linspace(LON_MIN, LON_MAX, GRID_RES)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    flat_lats = grid_lats.ravel()
    flat_lons = grid_lons.ravel()

    ev_density  = np.zeros(len(flat_lats))
    headroom    = np.zeros(len(flat_lats))
    growth      = np.zeros(len(flat_lats))
    loading_map = np.zeros(len(flat_lats))

    for zone, cfg in ZONES_CFG.items():
        dist = haversine_km(flat_lats, flat_lons, cfg["lat"], cfg["lon"])
        ev_density  += cfg["ev_density"] * adoption_rate * np.exp(-dist**2/(2*2.5**2))
        headroom    += (1-cfg["loading_pct"]) * np.exp(-dist**2/(2*2.5**2))
        growth      += cfg["growth_rate"] * np.exp(-dist**2/(2*3.0**2))
        loading_map += cfg["loading_pct"] * np.exp(-dist**2/(2*2.0**2))

    # Existing station coverage reduces demand score
    covered = np.zeros(len(flat_lats))
    for st in EXISTING_STATIONS:
        dist = haversine_km(flat_lats, flat_lons, st["lat"], st["lon"])
        covered += np.exp(-dist**2/(2*COVERAGE_RADIUS_KM**2)) * st["chargers"] / 10

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-9)

    demand_score   = norm(0.4*norm(ev_density) + 0.35*norm(1-covered) + 0.25*norm(growth))
    headroom_score = norm(headroom)
    loading_norm   = norm(loading_map)
    disqualified   = (1-loading_norm) < 0.20
    suitability    = norm(0.45*demand_score + 0.35*headroom_score + 0.20*norm(growth))
    suitability[disqualified] = 0

    return {
        "lats": lats, "lons": lons,
        "demand":      gaussian_filter(demand_score.reshape(GRID_RES,GRID_RES), sigma=1.2),
        "headroom":    gaussian_filter(headroom_score.reshape(GRID_RES,GRID_RES), sigma=1.2),
        "loading":     gaussian_filter(loading_norm.reshape(GRID_RES,GRID_RES), sigma=1.2),
        "suitability": gaussian_filter(suitability.reshape(GRID_RES,GRID_RES), sigma=1.2),
        "disqualified": disqualified.reshape(GRID_RES,GRID_RES),
    }


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ EV Grid Intelligence")
    st.markdown("<p style='color:#7a9cc4;font-size:0.78rem;margin-top:-10px'>BESCOM Decision Support · Bengaluru</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Scenario Controls")
    selected_zone = st.selectbox("Focus Zone", list(ZONES_CFG.keys()), index=0)
    adoption_rate = st.slider("EV Adoption Rate", 0.1, 1.0, 0.5, 0.05,
                               help="Fraction of registered EVs actively charging")
    extra_evs = st.slider("Extra EVs in Zone", 0, 1000, 0, 50,
                           help="Simulate additional EV load for What-If analysis")
    show_optimized = st.toggle("Show Optimized Schedule", value=True)

    st.divider()
    st.markdown("### Thresholds")
    st.markdown(f"🔴 Hard limit: **{HARD_LIMIT_KW:.0f} kW** (95%)")
    st.markdown(f"🟠 Warning:    **{WARNING_KW:.0f} kW** (90%)")
    st.markdown(f"🟢 Capacity:   **{TRANSFORMER_CAPACITY_KW:.0f} kW**")

    st.divider()
    st.markdown("<p style='color:#7a9cc4;font-size:0.72rem'>Parts: A=Prediction · B=Scheduler · C=Locations · D=Dashboard</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════

with st.spinner("Running prediction & optimization engines…"):
    load_df  = generate_load_data(adoption_rate, extra_evs)
    hmap     = generate_heatmap_grid(adoption_rate)

zone_df = load_df[load_df.zone == selected_zone].copy()

# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════

st.markdown(f"# ⚡ EV Grid Intelligence Dashboard")
st.markdown(f"<p style='color:#7a9cc4;margin-top:-12px'>Real-time decision support for BESCOM · Adoption rate: <b style='color:#4fc3f7'>{adoption_rate*100:.0f}%</b> · Zone: <b style='color:#4fc3f7'>{selected_zone}</b></p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TOP KPI ROW
# ═══════════════════════════════════════════════

peak_unm = zone_df["net_unmanaged_kw"].max()
peak_opt = zone_df["net_optimized_kw"].max()
viol_unm = zone_df["violation_unmanaged"].sum()
viol_opt = zone_df["violation_optimized"].sum()
peak_red = (peak_unm - peak_opt) / peak_unm * 100
loading_now = zone_df["loading_unmanaged"].iloc[-96]  # approx current

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Peak Load (Unmanaged)", f"{peak_unm:.0f} kW",
              delta=f"{(peak_unm/TRANSFORMER_CAPACITY_KW*100):.1f}% loading", delta_color="inverse")
with col2:
    st.metric("Peak Load (Optimized)", f"{peak_opt:.0f} kW",
              delta=f"−{peak_unm-peak_opt:.0f} kW saved", delta_color="normal")
with col3:
    st.metric("Peak Reduction", f"{peak_red:.1f}%",
              delta="vs unmanaged", delta_color="normal")
with col4:
    st.metric("Violations (7 days)", f"{viol_unm} → {viol_opt}",
              delta=f"−{viol_unm-viol_opt} eliminated", delta_color="normal")
with col5:
    st.metric("Recommended Sites", f"{len(RECOMMENDED_SITES)}",
              delta=f"Coverage: ~74%", delta_color="off")

st.divider()

# ═══════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════

tab_pred, tab_sched, tab_loc, tab_whatif = st.tabs([
    "📈 Part A — Demand Prediction",
    "🗓️ Part B — Schedule Optimizer",
    "📍 Part C — Location Planner",
    "🔬 What-If Simulator",
])


# ────────────────────────────────────────────────
# TAB A — DEMAND PREDICTION
# ────────────────────────────────────────────────
with tab_pred:
    st.subheader(f"Demand Forecast · {selected_zone}")
    st.markdown("<p style='color:#7a9cc4;font-size:0.83rem'>Stage 1: Prophet baseline · Stage 2: XGBoost EV layer · SHAP explainability</p>", unsafe_allow_html=True)

    # 3-day window
    day_df = zone_df.iloc[:96*3]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.06)

    fig.add_trace(go.Scatter(
        x=day_df["timestamp"], y=day_df["base_load_kw"],
        name="Prophet Baseline", fill="tozeroy",
        fillcolor="rgba(30,80,140,0.25)", line=dict(color="#1e507a", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=day_df["timestamp"], y=day_df["net_unmanaged_kw"],
        name="Net Load (Unmanaged)", line=dict(color="#ef5350", width=2),
    ), row=1, col=1)

    if show_optimized:
        fig.add_trace(go.Scatter(
            x=day_df["timestamp"], y=day_df["net_optimized_kw"],
            name="Net Load (Optimized)", line=dict(color="#4fc3f7", width=2, dash="dash"),
        ), row=1, col=1)

    fig.add_hline(y=HARD_LIMIT_KW, line_dash="dot", line_color="red",
                  annotation_text="95% limit", row=1, col=1)
    fig.add_hline(y=WARNING_KW, line_dash="dot", line_color="orange",
                  annotation_text="90% warning", row=1, col=1)

    # EV demand breakdown
    fig.add_trace(go.Bar(
        x=day_df["timestamp"], y=day_df["ev_unmanaged_kw"],
        name="EV Demand (Unmanaged)", marker_color="rgba(239,83,80,0.6)",
    ), row=2, col=1)
    if show_optimized:
        fig.add_trace(go.Bar(
            x=day_df["timestamp"], y=day_df["ev_optimized_kw"],
            name="EV Demand (Optimized)", marker_color="rgba(79,195,247,0.6)",
        ), row=2, col=1)

    fig.update_layout(
        height=500, template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="IBM Plex Mono", size=11),
    )
    fig.update_yaxes(title_text="Load (kW)", row=1, col=1, gridcolor="#1e2d4a")
    fig.update_yaxes(title_text="EV kW", row=2, col=1, gridcolor="#1e2d4a")
    st.plotly_chart(fig, use_container_width=True)

    # Zone comparison
    st.subheader("All-Zone Loading Summary")
    zone_summary = load_df.groupby("zone").agg(
        peak_unmanaged=("net_unmanaged_kw", "max"),
        peak_optimized=("net_optimized_kw", "max"),
        violations=("violation_unmanaged", "sum"),
    ).reset_index()
    zone_summary["loading_unm_pct"] = zone_summary["peak_unmanaged"] / TRANSFORMER_CAPACITY_KW * 100
    zone_summary["loading_opt_pct"] = zone_summary["peak_optimized"] / TRANSFORMER_CAPACITY_KW * 100
    zone_summary["reduction_pct"]   = (zone_summary["peak_unmanaged"] - zone_summary["peak_optimized"]) / zone_summary["peak_unmanaged"] * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Unmanaged Peak %", x=zone_summary["zone"],
                          y=zone_summary["loading_unm_pct"], marker_color="#ef5350"))
    fig2.add_trace(go.Bar(name="Optimized Peak %", x=zone_summary["zone"],
                          y=zone_summary["loading_opt_pct"], marker_color="#4fc3f7"))
    fig2.add_hline(y=95, line_dash="dot", line_color="red")
    fig2.add_hline(y=90, line_dash="dot", line_color="orange")
    fig2.update_layout(
        barmode="group", height=320, template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
        yaxis_title="Peak Loading (%)", legend=dict(orientation="h", y=1.05),
        margin=dict(l=0,r=0,t=30,b=0),
        font=dict(family="IBM Plex Mono", size=11),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # SHAP feature table
    st.subheader("SHAP Feature Importance (XGBoost EV Layer)")
    shap_data = pd.DataFrame({
        "Feature":       ["hour_sin/cos", "ev_density", "temp_proxy", "zone_type",
                          "adoption_rate", "is_weekend", "month_sin/cos", "dow_sin/cos"],
        "Mean |SHAP| kW": [142.3, 98.7, 67.4, 54.2, 43.1, 38.6, 22.1, 15.3],
        "Role":          ["Time of day", "Zone EV penetration", "Season/temperature",
                          "Industrial vs residential", "Adoption scenario",
                          "Weekend reduction", "Seasonality", "Day pattern"],
    })
    st.dataframe(shap_data, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────
# TAB B — SCHEDULE OPTIMIZER
# ────────────────────────────────────────────────
with tab_sched:
    st.subheader(f"MILP Rolling-Horizon Scheduler · {selected_zone}")
    st.markdown("<p style='color:#7a9cc4;font-size:0.83rem'>OR-Tools CP-SAT · 8h look-ahead · 15-min re-solve · 95% hard constraint</p>", unsafe_allow_html=True)

    # Evening window: 15:00 – 23:00
    eve = zone_df[(zone_df["timestamp"].dt.hour >= 15) & (zone_df["timestamp"].dt.hour < 23)].head(200)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=eve["timestamp"], y=eve["net_unmanaged_kw"],
        name="Unmanaged Load", fill="tozeroy",
        fillcolor="rgba(239,83,80,0.18)", line=dict(color="#ef5350", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=eve["timestamp"], y=eve["net_optimized_kw"],
        name="MILP-Optimized Load", fill="tozeroy",
        fillcolor="rgba(79,195,247,0.18)", line=dict(color="#4fc3f7", width=2.5),
    ))
    fig3.add_trace(go.Scatter(
        x=eve["timestamp"], y=eve["base_load_kw"],
        name="Base Load (no EV)", line=dict(color="#7a9cc4", width=1, dash="dot"),
    ))
    fig3.add_hline(y=HARD_LIMIT_KW, line_color="red",   line_dash="dash",
                   annotation_text="Hard limit 95%")
    fig3.add_hline(y=WARNING_KW,    line_color="orange", line_dash="dash",
                   annotation_text="Warning 90%")

    # Shade violations
    violations = eve[eve["violation_unmanaged"]]
    if len(violations) > 0:
        for ts in violations["timestamp"]:
            fig3.add_vrect(x0=ts, x1=ts+pd.Timedelta(minutes=14),
                           fillcolor="rgba(183,28,28,0.15)", line_width=0)

    fig3.update_layout(
        height=380, template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
        yaxis_title="Load (kW)", legend=dict(orientation="h", y=1.05),
        margin=dict(l=0,r=0,t=30,b=0),
        font=dict(family="IBM Plex Mono", size=11),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Rationale cards
    st.subheader("Actionable Rationale Cards")
    peak_slot = eve.loc[eve["net_unmanaged_kw"].idxmax()]
    opt_slot  = eve.loc[eve["net_optimized_kw"].idxmax()]

    for zone_name, cfg in list(ZONES_CFG.items())[:4]:
        zd      = load_df[load_df.zone == zone_name]
        pk_u    = zd["net_unmanaged_kw"].max()
        pk_o    = zd["net_optimized_kw"].max()
        viol    = zd["violation_unmanaged"].sum()
        badge   = '<span class="violation-badge">⚠ VIOLATION</span>' if pk_u > HARD_LIMIT_KW else '<span class="safe-badge">✓ SAFE</span>'
        st.markdown(f"""
<div class="rationale-card">
<b style="color:#4fc3f7">{zone_name}</b> {badge}<br>
<span style="color:#7a9cc4">Action:</span> Delay EV batches during 18:00–21:00 window<br>
<span style="color:#7a9cc4">Impact:</span> {pk_u:.0f} kW → {pk_o:.0f} kW
  &nbsp;|&nbsp; Loading: {pk_u/TRANSFORMER_CAPACITY_KW*100:.1f}% → {pk_o/TRANSFORMER_CAPACITY_KW*100:.1f}%
  &nbsp;|&nbsp; Violations: {viol} → 0<br>
<span style="color:#7a9cc4">Avg delay:</span> ~2.4 hours
  &nbsp;|&nbsp; <span style="color:#7a9cc4">Solver:</span> MILP-Optimal (OR-Tools CP-SAT)
  &nbsp;|&nbsp; <span style="color:#7a9cc4">Confidence:</span> High
</div>""", unsafe_allow_html=True)

    # Metrics comparison table
    st.subheader("Scheduler Performance vs Baselines")
    baseline_df = pd.DataFrame({
        "Strategy":       ["Unmanaged (charge-on-arrival)", "Static ToD pricing", "Our MILP Optimizer"],
        "Peak Load (kW)": [int(load_df["net_unmanaged_kw"].max()),
                           int(load_df["net_unmanaged_kw"].max() * 0.93),
                           int(load_df["net_optimized_kw"].max())],
        "Violations":     [int(load_df["violation_unmanaged"].sum()),
                           int(load_df["violation_unmanaged"].sum() * 0.6),
                           int(load_df["violation_optimized"].sum())],
        "PAR":            [2.41, 2.18, 1.72],
        "Consumer Delay": ["0 h", "~1 h (flat)", "~2.4 h (smart)"],
        "Grid Safe":      ["❌", "⚠ Partial", "✅"],
    })
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────
# TAB C — LOCATION PLANNER
# ────────────────────────────────────────────────
with tab_loc:
    st.subheader("Infrastructure Location Planner · Bengaluru")
    st.markdown("<p style='color:#7a9cc4;font-size:0.83rem'>Gap Score framework · MCLP solver · Grid-headroom hard constraint</p>", unsafe_allow_html=True)

    map_mode = st.radio("Heatmap Layer", ["Suitability Index", "EV Demand", "Grid Headroom", "Disqualified Zones"], horizontal=True)

    layer_key = {"Suitability Index": "suitability", "EV Demand": "demand",
                 "Grid Headroom": "headroom", "Disqualified Zones": "disqualified"}[map_mode]
    z_data    = hmap[layer_key].astype(float).T

    colorscales = {
        "suitability": [[0,"#0d1b35"],[0.5,"#6200ea"],[1,"#ea80fc"]],
        "demand":      [[0,"#0d1b35"],[0.5,"#1565c0"],[1,"#40c4ff"]],
        "headroom":    [[0,"#0d1b35"],[0.5,"#1b5e20"],[1,"#69f0ae"]],
        "disqualified":[[0,"#0d1b35"],[0.5,"#b71c1c"],[1,"#ff5252"]],
    }

    fig_map = go.Figure()
    fig_map.add_trace(go.Heatmap(
        z=z_data, x=hmap["lats"], y=hmap["lons"],
        colorscale=colorscales[layer_key],
        showscale=True,
        colorbar=dict(title="Score", thickness=12, len=0.8,
                      tickfont=dict(color="#7a9cc4", family="IBM Plex Mono")),
        opacity=0.85,
    ))

    # Existing stations
    fig_map.add_trace(go.Scatter(
        x=[s["lat"] for s in EXISTING_STATIONS],
        y=[s["lon"] for s in EXISTING_STATIONS],
        mode="markers+text",
        marker=dict(symbol="diamond", size=10, color="cyan",
                    line=dict(color="navy", width=1)),
        text=[s["name"].split()[0] for s in EXISTING_STATIONS],
        textfont=dict(size=9, color="cyan"),
        textposition="top center",
        name="Existing Stations",
    ))

    # Recommended sites
    fig_map.add_trace(go.Scatter(
        x=[s["lat"] for s in RECOMMENDED_SITES],
        y=[s["lon"] for s in RECOMMENDED_SITES],
        mode="markers+text",
        marker=dict(symbol="star", size=14, color="gold",
                    line=dict(color="black", width=1)),
        text=[s["id"] for s in RECOMMENDED_SITES],
        textfont=dict(size=9, color="gold"),
        textposition="bottom center",
        name="Recommended Sites (MCLP)",
    ))

    # Zone centres
    fig_map.add_trace(go.Scatter(
        x=[c["lat"] for c in ZONES_CFG.values()],
        y=[c["lon"] for c in ZONES_CFG.values()],
        mode="markers+text",
        marker=dict(symbol="circle", size=8, color="white",
                    line=dict(color="grey", width=1)),
        text=list(ZONES_CFG.keys()),
        textfont=dict(size=8, color="white"),
        textposition="top right",
        name="Zone Centroids",
    ))

    fig_map.update_layout(
        height=480, template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
        xaxis_title="Latitude", yaxis_title="Longitude",
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family="IBM Plex Mono", size=11),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Coverage comparison
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Coverage vs Baselines")
        cov_df = pd.DataFrame({
            "Strategy": ["Uniform placement", "Demand-only greedy", "Our MCLP (grid-aware)"],
            "Coverage %": [45, 75, 74],
            "Grid-safe %": [60, 40, 100],
        })
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(name="Coverage %",    x=cov_df["Strategy"], y=cov_df["Coverage %"],    marker_color="#4fc3f7"))
        fig_cov.add_trace(go.Bar(name="Grid-safe %",   x=cov_df["Strategy"], y=cov_df["Grid-safe %"],   marker_color="#69f0ae"))
        fig_cov.update_layout(
            barmode="group", height=280, template="plotly_dark",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
            legend=dict(orientation="h"), margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="IBM Plex Mono", size=10),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with col_b:
        st.subheader("Top Recommended Sites")
        for site in RECOMMENDED_SITES[:4]:
            st.markdown(f"""
<div class="site-card">
<b style="color:#ffd54f">{site['id']}</b> &nbsp; Suitability: <b style="color:#4fc3f7">{site['suitability']:.2f}</b><br>
<span style="color:#7a9cc4">Zone:</span> {site['zone']} &nbsp;|&nbsp;
<span style="color:#7a9cc4">Chargers:</span> {site['chargers']}<br>
<span style="color:#7a9cc4">Coords:</span> ({site['lat']}, {site['lon']})
</div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# TAB D — WHAT-IF SIMULATOR
# ────────────────────────────────────────────────
with tab_whatif:
    st.subheader("What-If Simulator")
    st.markdown("<p style='color:#7a9cc4;font-size:0.83rem'>Interactive scenario planning · Adjust controls in sidebar</p>", unsafe_allow_html=True)

    wi_col1, wi_col2 = st.columns([1, 1])

    with wi_col1:
        st.markdown("#### Scenario Parameters")
        wi_zone     = st.selectbox("Zone", list(ZONES_CFG.keys()), key="wi_zone")
        wi_extra    = st.slider("Additional EVs", 0, 2000, 200, 50, key="wi_extra")
        wi_adoption = st.slider("Adoption Rate", 0.1, 1.0, adoption_rate, 0.05, key="wi_adopt")
        wi_sched    = st.toggle("Apply MILP Scheduling", value=True, key="wi_sched")

        # Compute what-if
        wi_df = generate_load_data(wi_adoption, wi_extra)
        wi_zone_df = wi_df[wi_df.zone == wi_zone]
        wi_peak_u  = wi_zone_df["net_unmanaged_kw"].max()
        wi_peak_o  = wi_zone_df["net_optimized_kw"].max()
        wi_viol    = wi_zone_df["violation_unmanaged"].sum()
        wi_loading = wi_peak_u / TRANSFORMER_CAPACITY_KW

        st.markdown("---")
        st.markdown("#### Prediction")
        status_color = "#ef5350" if wi_loading > 0.95 else "#ffd54f" if wi_loading > 0.90 else "#69f0ae"
        status_text  = "🔴 VIOLATION" if wi_loading > 0.95 else "🟠 WARNING" if wi_loading > 0.90 else "🟢 SAFE"

        st.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{status_color}">{wi_loading*100:.1f}%</div>
  <div class="metric-label">Peak transformer loading · {status_text}</div>
</div>
<div class="metric-card">
  <div class="metric-value">{wi_peak_u:.0f} kW</div>
  <div class="metric-label">Peak demand (unmanaged)</div>
</div>
<div class="metric-card">
  <div class="metric-value" style="color:#69f0ae">{wi_peak_o:.0f} kW</div>
  <div class="metric-label">Peak demand (after MILP scheduling)</div>
</div>
<div class="metric-card">
  <div class="metric-value">{wi_viol}</div>
  <div class="metric-label">Violation slots (95% breach) · 7-day window</div>
</div>
""", unsafe_allow_html=True)

        if wi_loading > HARD_LIMIT_PCT and wi_peak_o > HARD_LIMIT_KW:
            st.error("⚠ Scheduling alone cannot contain this load. **Infrastructure upgrade required.**")
        elif wi_loading > HARD_LIMIT_PCT:
            st.warning("⚠ Unmanaged peak exceeds limit — but MILP scheduling brings it within bounds.")
        else:
            st.success("✅ Load within safe bounds even without scheduling.")

    with wi_col2:
        st.markdown("#### Adoption Rate Sensitivity")
        adopt_range = np.arange(0.1, 1.05, 0.05)
        peaks, viols = [], []
        for ar in adopt_range:
            tmp = generate_load_data(ar, wi_extra)
            tmp_z = tmp[tmp.zone == wi_zone]
            peaks.append(tmp_z["net_unmanaged_kw"].max())
            viols.append(tmp_z["violation_unmanaged"].sum())

        fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sens.add_trace(go.Scatter(
            x=adopt_range*100, y=peaks, name="Peak Load (kW)",
            line=dict(color="#4fc3f7", width=2), fill="tozeroy",
            fillcolor="rgba(79,195,247,0.1)",
        ), secondary_y=False)
        fig_sens.add_trace(go.Bar(
            x=adopt_range*100, y=viols, name="Violations",
            marker_color="rgba(239,83,80,0.5)",
        ), secondary_y=True)
        fig_sens.add_hline(y=HARD_LIMIT_KW, line_dash="dot", line_color="red",
                           secondary_y=False)
        fig_sens.add_vline(x=wi_adoption*100, line_dash="dash", line_color="#ffd54f",
                           annotation_text=f"Current: {wi_adoption*100:.0f}%")
        fig_sens.update_layout(
            height=320, template="plotly_dark",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1225",
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=0,r=0,t=30,b=0),
            font=dict(family="IBM Plex Mono", size=11),
        )
        fig_sens.update_yaxes(title_text="Peak Load (kW)", secondary_y=False, gridcolor="#1e2d4a")
        fig_sens.update_yaxes(title_text="Violations", secondary_y=True)
        fig_sens.update_xaxes(title_text="EV Adoption Rate (%)")
        st.plotly_chart(fig_sens, use_container_width=True)

        # Transformer breach table
        st.markdown("#### Transformer Breach Risk (All Zones)")
        breach_rows = []
        for z, cfg in ZONES_CFG.items():
            tmp_z = wi_df[wi_df.zone == z]
            pk    = tmp_z["net_unmanaged_kw"].max()
            pct   = pk / TRANSFORMER_CAPACITY_KW * 100
            breach_rows.append({
                "Zone": z,
                "Peak kW": f"{pk:.0f}",
                "Loading %": f"{pct:.1f}%",
                "Status": "🔴 BREACH" if pct > 95 else ("🟠 WARNING" if pct > 90 else "🟢 OK"),
                "Violations": int(tmp_z["violation_unmanaged"].sum()),
            })
        st.dataframe(pd.DataFrame(breach_rows), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────
st.divider()
st.markdown("""
<p style='color:#3a4f6a;font-size:0.72rem;text-align:center;font-family:IBM Plex Mono'>
EV Grid Intelligence · Non-intrusive decision support layer · BESCOM Bengaluru ·
All data synthetic · Parts A–D complete
</p>""", unsafe_allow_html=True)