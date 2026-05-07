"""
EV Grid Stress Prediction Pipeline
===================================
Two-stage forecast:
  Stage 1 — Facebook Prophet: 15-min baseline load per transformer
  Stage 2 — XGBoost: injects EV-specific demand on top of baseline
  SHAP explainability on every XGBoost prediction

Install dependencies:
    pip install prophet xgboost shap scikit-learn pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ZONES = {
    "Koramangala":  {"base_kw": 2800, "ev_density": 0.18, "zone_type": 1},  # residential-commercial
    "Whitefield":   {"base_kw": 3500, "ev_density": 0.22, "zone_type": 2},  # industrial-tech
    "Jayanagar":    {"base_kw": 2100, "ev_density": 0.12, "zone_type": 0},  # residential
    "Marathahalli": {"base_kw": 3100, "ev_density": 0.20, "zone_type": 1},
    "Hebbal":       {"base_kw": 2600, "ev_density": 0.15, "zone_type": 1},
}

TRANSFORMER_CAPACITY_KW = 4000   # 95 % hard limit = 3800 kW
CAPACITY_WARNING_PCT    = 0.90   # flag above this
CAPACITY_HARD_LIMIT_PCT = 0.95

START_DATE  = "2024-01-01"
END_DATE    = "2024-06-30"
FREQ        = "15min"

# Bengaluru public holidays (approx)
HOLIDAYS = pd.DataFrame({
    "holiday": ["Republic Day", "Ugadi", "Ram Navami", "Labour Day",
                "Independence Day", "Gandhi Jayanti", "Diwali", "Christmas"],
    "ds": pd.to_datetime(["2024-01-26", "2024-04-09", "2024-04-17", "2024-05-01",
                           "2024-08-15", "2024-10-02", "2024-11-01", "2024-12-25"]),
})

# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def make_time_index(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)


def base_load_curve(ts: pd.DatetimeIndex, base_kw: float, zone_type: int) -> np.ndarray:
    """Diurnal load profile with zone-specific shape."""
    h = ts.hour + ts.minute / 60

    # Morning peak ~8-10 AM, evening peak ~7-9 PM
    morning  = np.exp(-0.5 * ((h - 9.0) / 1.5) ** 2)
    evening  = np.exp(-0.5 * ((h - 20.0) / 1.5) ** 2)
    night_dip = 0.35 + 0.65 * (morning + evening)

    if zone_type == 2:   # industrial: strong daytime load
        daytime = np.where((h >= 8) & (h <= 18), 0.4, 0.0)
        profile = night_dip + daytime
    else:
        profile = night_dip

    # Weekend reduction
    weekend_factor = np.where(ts.dayofweek >= 5, 0.75, 1.0)

    # Seasonal: slightly higher in summer (Apr-Jun) due to AC
    seasonal = 1 + 0.08 * np.sin(2 * np.pi * (ts.dayofyear - 90) / 365)

    # Gaussian noise
    noise = np.random.normal(0, 0.03, len(ts))

    load = base_kw * profile * weekend_factor * seasonal * (1 + noise)
    return np.clip(load, base_kw * 0.2, base_kw * 1.1)


def ev_demand(ts: pd.DatetimeIndex, ev_density: float, adoption_rate: float = 0.5) -> np.ndarray:
    """Simulate EV charging demand (unmanaged, charge-at-arrival behaviour)."""
    h = ts.hour + ts.minute / 60

    # Two charging peaks: 6-8 PM (home arrival) and 8-10 AM (workplace)
    home_peak      = np.exp(-0.5 * ((h - 19.0) / 1.2) ** 2)
    workplace_peak = np.exp(-0.5 * ((h - 8.5)  / 1.0) ** 2)
    ev_profile     = 0.7 * home_peak + 0.3 * workplace_peak

    # Temperature proxy: hot days → more AC → less range → more frequent charging
    temp_factor = 1 + 0.05 * np.sin(2 * np.pi * (ts.dayofyear - 90) / 365)

    # Weekend: more leisure driving → charging spread
    weekend = np.where(ts.dayofweek >= 5, 1.15, 1.0)

    # Simulated number of active EVs in zone
    n_evs    = int(5000 * ev_density * adoption_rate)
    avg_kw   = 7.2   # typical L2 charger (kW)
    concurrency = 0.15  # fraction charging simultaneously

    noise  = np.random.normal(0, 0.05, len(ts))
    demand = (n_evs * avg_kw * concurrency * ev_profile
              * temp_factor * weekend * (1 + noise))
    return np.clip(demand, 0, None)


def generate_dataset(zone_name: str, cfg: dict) -> pd.DataFrame:
    ts  = make_time_index(START_DATE, END_DATE, FREQ)
    bl  = base_load_curve(ts, cfg["base_kw"], cfg["zone_type"])
    ev  = ev_demand(ts, cfg["ev_density"])
    net = bl + ev

    df = pd.DataFrame({
        "ds":           ts,
        "zone":         zone_name,
        "zone_type":    cfg["zone_type"],
        "base_load_kw": bl,
        "ev_load_kw":   ev,
        "net_load_kw":  net,
        "capacity_kw":  TRANSFORMER_CAPACITY_KW,
        "loading_pct":  net / TRANSFORMER_CAPACITY_KW,
        "violation":    (net / TRANSFORMER_CAPACITY_KW) > CAPACITY_HARD_LIMIT_PCT,
        # time features
        "hour":         ts.hour,
        "minute":       ts.minute,
        "dayofweek":    ts.dayofweek,
        "month":        ts.month,
        "dayofyear":    ts.dayofyear,
        "is_weekend":   (ts.dayofweek >= 5).astype(int),
        # cyclical encoding
        "hour_sin":     np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":     np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin":      np.sin(2 * np.pi * ts.dayofweek / 7),
        "dow_cos":      np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin":    np.sin(2 * np.pi * ts.month / 12),
        "month_cos":    np.cos(2 * np.pi * ts.month / 12),
        # EV features
        "ev_density":   cfg["ev_density"],
        "adoption_rate": 0.5,
        "temp_proxy":   20 + 8 * np.sin(2 * np.pi * (ts.dayofyear - 90) / 365)
                        + np.random.normal(0, 1.5, len(ts)),
    })
    return df


print("Generating synthetic data for all zones...")
all_data = pd.concat([generate_dataset(z, c) for z, c in ZONES.items()], ignore_index=True)
print(f"  Total rows: {len(all_data):,}  |  Zones: {all_data.zone.nunique()}")

# ─────────────────────────────────────────────
# 2.  STAGE 1 — PROPHET BASELINE FORECAST
# ─────────────────────────────────────────────
from prophet import Prophet

def train_prophet(zone_df: pd.DataFrame) -> tuple:
    """Train Prophet on base_load_kw; return model + forecast DataFrame."""
    prophet_df = zone_df[["ds", "base_load_kw"]].rename(columns={"base_load_kw": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=HOLIDAYS,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.90,
    )
    # Add custom Bengaluru seasonality
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)

    # 80/20 temporal split
    split = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split]
    test_df  = prophet_df.iloc[split:]

    m.fit(train_df)
    forecast = m.predict(test_df[["ds"]])
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"yhat": "baseline_pred", "yhat_lower": "bl_lower", "yhat_upper": "bl_upper"}
    )
    mae = np.mean(np.abs(test_df["y"].values - forecast["baseline_pred"].values))
    return m, forecast, test_df, mae


print("\nTraining Prophet models per zone...")
prophet_results = {}
for zone in ZONES:
    zdf = all_data[all_data.zone == zone].copy()
    model, forecast, test, mae = train_prophet(zdf)
    prophet_results[zone] = {"model": model, "forecast": forecast, "test": test, "mae": mae}
    print(f"  {zone:15s}  MAE = {mae:.1f} kW")

# ─────────────────────────────────────────────
# 3.  STAGE 2 — XGBOOST EV DEMAND LAYER
# ─────────────────────────────────────────────
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

XGB_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "zone_type", "ev_density", "adoption_rate", "temp_proxy",
    "base_load_kw",           # Prophet baseline passed as feature
]
TARGET = "ev_load_kw"

le = LabelEncoder()
all_data["zone_enc"] = le.fit_transform(all_data["zone"])

split_date = pd.Timestamp("2024-05-15")
train_df = all_data[all_data.ds < split_date].copy()
test_df  = all_data[all_data.ds >= split_date].copy()

xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)

print("\nTraining XGBoost EV demand model...")
xgb_model.fit(
    train_df[XGB_FEATURES], train_df[TARGET],
    eval_set=[(test_df[XGB_FEATURES], test_df[TARGET])],
    verbose=False,
)

test_df["ev_pred"] = xgb_model.predict(test_df[XGB_FEATURES])
test_df["net_pred"] = test_df["base_load_kw"] + test_df["ev_pred"]
test_df["predicted_loading_pct"] = test_df["net_pred"] / TRANSFORMER_CAPACITY_KW
test_df["predicted_violation"]   = test_df["predicted_loading_pct"] > CAPACITY_HARD_LIMIT_PCT

mae_ev  = mean_absolute_error(test_df[TARGET], test_df["ev_pred"])
mae_net = mean_absolute_error(test_df["net_load_kw"], test_df["net_pred"])
rmse    = np.sqrt(mean_squared_error(test_df["net_load_kw"], test_df["net_pred"]))
viol_recall = (
    (test_df["violation"] & test_df["predicted_violation"]).sum()
    / test_df["violation"].sum()
    if test_df["violation"].sum() > 0 else float("nan")
)

print(f"\n  XGBoost Results (test set: {split_date.date()} → {END_DATE})")
print(f"  EV demand MAE   : {mae_ev:.1f} kW")
print(f"  Net load MAE    : {mae_net:.1f} kW")
print(f"  Net load RMSE   : {rmse:.1f} kW")
print(f"  Violation recall: {viol_recall:.1%}")

# ─────────────────────────────────────────────
# 4.  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
import shap

print("\nComputing SHAP values (this may take ~30s)...")
explainer   = shap.TreeExplainer(xgb_model)
sample      = test_df[XGB_FEATURES].sample(min(2000, len(test_df)), random_state=RANDOM_SEED)
shap_values = explainer.shap_values(sample)
shap_df     = pd.DataFrame(shap_values, columns=XGB_FEATURES)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

print("\n  Top feature contributions to EV demand prediction:")
for feat, val in mean_abs_shap.head(6).items():
    print(f"    {feat:20s}  mean |SHAP| = {val:.2f} kW")

# ─────────────────────────────────────────────
# 5.  VIOLATION ALERTS
# ─────────────────────────────────────────────
print("\n── Capacity Violation Alerts ──────────────────────────────────")
alerts = test_df[test_df["predicted_loading_pct"] > CAPACITY_WARNING_PCT].copy()
alerts = alerts.sort_values("predicted_loading_pct", ascending=False)

for zone, grp in alerts.groupby("zone"):
    worst = grp.iloc[0]
    print(f"  ⚠  {zone:15s}  peak predicted loading = {worst['predicted_loading_pct']:.1%}"
          f"  at {worst['ds'].strftime('%Y-%m-%d %H:%M')}")

# ─────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────
print("\nGenerating plots...")
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("EV Grid Stress — Prediction Pipeline Results", fontsize=15, fontweight="bold")
palette = sns.color_palette("tab10", n_colors=len(ZONES))

# --- Plot 1: Actual vs Predicted (Koramangala, 1 week) ---
ax = axes[0, 0]
zone_test = test_df[test_df.zone == "Koramangala"].set_index("ds")
week = zone_test.iloc[:7*96]   # 7 days × 96 slots
ax.plot(week.index, week["net_load_kw"], label="Actual", lw=1.5, color="steelblue")
ax.plot(week.index, week["net_pred"],    label="Predicted", lw=1.5, color="coral", ls="--")
ax.axhline(TRANSFORMER_CAPACITY_KW * CAPACITY_HARD_LIMIT_PCT, color="red",
           lw=1, ls=":", label="95% limit")
ax.set_title("Koramangala — Actual vs Predicted (1 week)")
ax.set_ylabel("Load (kW)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# --- Plot 2: Loading % all zones ---
ax = axes[0, 1]
for i, (zone, color) in enumerate(zip(ZONES, palette)):
    zdf = test_df[test_df.zone == zone].set_index("ds")["predicted_loading_pct"] * 100
    daily_max = zdf.resample("D").max()
    ax.plot(daily_max.index, daily_max.values, label=zone, color=color, lw=1.2)
ax.axhline(95, color="red",    ls=":", lw=1.2, label="Hard limit 95%")
ax.axhline(90, color="orange", ls=":", lw=1.0, label="Warning 90%")
ax.set_title("Daily Peak Loading % — All Zones")
ax.set_ylabel("Loading (%)")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# --- Plot 3: EV vs Base load share ---
ax = axes[1, 0]
zone_means = test_df.groupby("zone")[["base_load_kw", "ev_pred"]].mean()
zone_means.plot(kind="bar", ax=ax, color=["steelblue", "coral"], edgecolor="white", width=0.6)
ax.set_title("Average Load Composition by Zone")
ax.set_ylabel("kW")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
ax.legend(["Base Load", "EV Demand (predicted)"], fontsize=8)

# --- Plot 4: Diurnal EV demand profile ---
ax = axes[1, 1]
test_df["hour_slot"] = test_df["hour"] + test_df["minute"] / 60
for zone, color in zip(ZONES, palette):
    hourly = test_df[test_df.zone == zone].groupby("hour_slot")["ev_pred"].mean()
    ax.plot(hourly.index, hourly.values, label=zone, color=color, lw=1.4)
ax.set_title("Average Diurnal EV Demand Profile")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Predicted EV Load (kW)")
ax.set_xticks(range(0, 25, 3))
ax.legend(fontsize=7)

# --- Plot 5: SHAP bar chart ---
ax = axes[2, 0]
mean_abs_shap.head(10).sort_values().plot(kind="barh", ax=ax, color="mediumpurple")
ax.set_title("Feature Importance (mean |SHAP|)")
ax.set_xlabel("mean |SHAP value| (kW)")

# --- Plot 6: Scatter actual vs predicted ---
ax = axes[2, 1]
sample_scatter = test_df.sample(min(3000, len(test_df)), random_state=42)
ax.scatter(sample_scatter["net_load_kw"], sample_scatter["net_pred"],
           alpha=0.15, s=8, color="steelblue")
lims = [sample_scatter["net_load_kw"].min(), sample_scatter["net_load_kw"].max()]
ax.plot(lims, lims, "r--", lw=1.2, label="Perfect prediction")
ax.set_xlabel("Actual net load (kW)")
ax.set_ylabel("Predicted net load (kW)")
ax.set_title(f"Actual vs Predicted — All Zones  (MAE={mae_net:.0f} kW)")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = "ev_prediction_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {plot_path}")

# ─────────────────────────────────────────────
# 7.  SAMPLE RATIONALE CARD (structured output)
# ─────────────────────────────────────────────
worst_zone = test_df.loc[test_df["predicted_loading_pct"].idxmax()]

print("\n── Sample Prediction Rationale Card ───────────────────────────")
print(f"  Zone            : {worst_zone['zone']}")
print(f"  Timestamp       : {worst_zone['ds']}")
print(f"  Baseline load   : {worst_zone['base_load_kw']:.0f} kW")
print(f"  EV demand pred  : {worst_zone['ev_pred']:.0f} kW")
print(f"  Net load pred   : {worst_zone['net_pred']:.0f} kW")
print(f"  Capacity        : {TRANSFORMER_CAPACITY_KW} kW")
print(f"  Loading         : {worst_zone['predicted_loading_pct']:.1%}")
print(f"  Violation flag  : {'⚠ YES' if worst_zone['predicted_violation'] else 'NO'}")
top_shap = mean_abs_shap.head(3)
print(f"  Top SHAP drivers: {', '.join(f'{f} ({v:.1f} kW)' for f, v in top_shap.items())}")
print(f"  Model MAE       : {mae_net:.1f} kW  |  Violation recall: {viol_recall:.1%}")
print("────────────────────────────────────────────────────────────────")

print("\nDone! Next steps:")
print("  1. Run the scheduling optimizer (Part B — MILP via OR-Tools)")
print("  2. Plug in real SCADA CSV → replace generate_dataset() with pd.read_csv()")
print("  3. Tune TRANSFORMER_CAPACITY_KW per transformer from your topology data")
