"""
app.py
======
Weather Prediction System - Direct Weather Forecasting & AI Prediction Dashboard.

Focuses directly on:
1. Real-time live weather conditions for selected cities.
2. AI-driven Next-Day Weather Prediction (Rainfall probability & Max Temperature).
3. Physics-based Numerical Weather Prediction (NWP) Forecast for tomorrow.
4. 7-Day upcoming weather forecast with interactive temperature and precipitation trends.
5. Interactive What-If weather simulator.
"""

from typing import Dict, Any, Optional
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Local modules
from data_loader import (
    load_raw_dataset,
    prepare_engineered_dataset,
    get_latest_inference_features,
    CITIES_METADATA,
    MODEL_FEATURE_NAMES,
)
from predict import get_predictor, check_models_exist
from api_client import fetch_live_city_weather, convert_live_weather_to_ml_features


# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Weather Forecasting & AI Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    /* Hero Banner */
    .weather-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0284c7 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
    }
    .weather-hero h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.1rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .weather-hero p {
        color: #e0f2fe;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .weather-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .weather-card:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }

    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }

    .card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }

    .card-subtext {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Prediction Outcome Cards */
    .pred-card-ml {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
        border: 2px solid #86efac;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.08);
        height: 100%;
    }
    .pred-card-nwp {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
        border: 2px solid #7dd3fc;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.08);
        height: 100%;
    }

    .badge-ml {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
    }
    .badge-nwp {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
    }

    .badge-rain-yes {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-rain-no {
        background-color: #fef3c7;
        color: #b45309;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    .consensus-banner {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 18px;
        margin-bottom: 24px;
        font-size: 0.94rem;
        color: #1e293b;
    }

    /* 7-day forecast item card */
    .forecast-day-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .forecast-day-card:hover {
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED LOADERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading historical dataset...")
def get_cached_dataset() -> pd.DataFrame:
    raw_df = load_raw_dataset("data/india_2000_2024_daily_weather.csv")
    return prepare_engineered_dataset(raw_df)


@st.cache_resource(show_spinner="Loading trained ML models...")
def get_cached_predictor():
    return get_predictor()


# Verify models existence
models_ok, model_msg = check_models_exist()
if not models_ok:
    st.error(f"❌ {model_msg}. Please run `python train_models.py` to generate the models.")
    st.stop()

try:
    df_all = get_cached_dataset()
    predictor = get_cached_predictor()
except Exception as e:
    st.error(f"Failed initializing system: {str(e)}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. SIDEBAR: CITY SELECTION & METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌦️ Weather Forecast & AI")
    st.caption("Next-Day Prediction & Real-Time Weather")

    st.markdown("---")
    st.subheader("📍 Target City")
    cities = list(CITIES_METADATA.keys())
    selected_city = st.selectbox(
        "Choose City",
        options=cities,
        index=0,
        help="Select any of the 10 supported Indian metropolitan cities."
    )
    city_meta = CITIES_METADATA[selected_city]

    # City info card
    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; margin: 12px 0;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase;">Location Details</div>
        <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{selected_city}</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">{city_meta['state']} ({city_meta['region']})</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
            🌍 {city_meta['lat']}°N, {city_meta['lon']}°E<br>
            🌤️ {city_meta['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Weather Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.82rem; color: #64748b; line-height: 1.5;">
        <strong>How it works:</strong><br>
        • <strong>Live Weather:</strong> Real-time atmospheric conditions from Open-Meteo.<br>
        • <strong>AI Prediction:</strong> Random Forest model trained on 25 years of meteorological data.<br>
        • <strong>Forecast:</strong> 7-Day high-resolution Numerical Weather Prediction.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. FETCH DATA & RUN PREDICTION
# -----------------------------------------------------------------------------
now_dt = datetime.datetime.now()
formatted_time = now_dt.strftime("%A, %B %d, %Y • %I:%M %p")

# Fetch live Open-Meteo weather
with st.spinner(f"Fetching current weather and computing AI predictions for {selected_city}..."):
    live_res = fetch_live_city_weather(selected_city)

is_live = live_res.get("success", False)

if is_live:
    curr = live_res["current"]
    today_stats = live_res["today"]
    tmr = live_res["tomorrow"]
    f_df = live_res["forecast_table"]
    
    # Run ML inference on live weather features
    ml_features = convert_live_weather_to_ml_features(live_res)
    ml_pred = predictor.predict(ml_features)
else:
    # Fallback to historical dataset record if API is unreachable
    st.warning("⚠️ Live API temporarily unavailable. Using latest recorded dataset observations.")
    ml_features, meta = get_latest_inference_features(df_all, selected_city)
    ml_pred = predictor.predict(ml_features)
    curr = {
        "condition_icon": "🌤️",
        "condition_text": "Dataset Record",
        "temperature": float(ml_features["temperature_2m_max"].iloc[0]),
        "apparent_temperature": float(ml_features["temperature_2m_max"].iloc[0]),
        "humidity": 65,
        "wind_speed": float(ml_features["wind_speed_10m_max"].iloc[0]),
        "precipitation": float(ml_features["rain_sum"].iloc[0]),
        "time": "Dec 31, 2024"
    }
    tmr = {
        "date": "Next Day",
        "condition_icon": "🌦️",
        "condition_text": "ML Predicted",
        "temp_max": ml_pred["temp_prediction"],
        "temp_min": ml_pred["temp_prediction"] - 8.0,
        "rain_sum": 0.0,
        "rain_prob_max": ml_pred["rain_probability_pct"]
    }
    f_df = pd.DataFrame()


# -----------------------------------------------------------------------------
# 5. HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="weather-hero">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1>{curr['condition_icon']} Weather in {selected_city}</h1>
            <p>{city_meta['state']}, India &nbsp;•&nbsp; {formatted_time}</p>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span style="background: rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 9999px; font-weight: 600; font-size: 0.88rem;">
                🟢 Live & Prediction Active
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. CURRENT LIVE WEATHER SECTION
# -----------------------------------------------------------------------------
st.markdown("### 🕒 Current Atmospheric Conditions")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="weather-card">
        <div class="card-title">Weather Condition</div>
        <div class="card-value" style="font-size: 1.45rem;">{curr['condition_icon']} {curr['condition_text']}</div>
        <div class="card-subtext">Real-time status</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="weather-card">
        <div class="card-title">Current Temperature</div>
        <div class="card-value">{curr['temperature']:.1f} °C</div>
        <div class="card-subtext">Feels like: {curr['apparent_temperature']:.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="weather-card">
        <div class="card-title">Relative Humidity</div>
        <div class="card-value">{curr['humidity']} %</div>
        <div class="card-subtext">Atmospheric moisture</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="weather-card">
        <div class="card-title">Wind Speed</div>
        <div class="card-value">{curr['wind_speed']:.1f} <span style="font-size: 1rem; font-weight: 500;">km/h</span></div>
        <div class="card-subtext">Surface velocity (10m)</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="weather-card">
        <div class="card-title">Today's Precipitation</div>
        <div class="card-value">{curr['precipitation']:.1f} <span style="font-size: 1rem; font-weight: 500;">mm</span></div>
        <div class="card-subtext">Rainfall measured today</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. TOMORROW'S WEATHER PREDICTION & FORECAST
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"### 🔮 Tomorrow's Weather Prediction & Forecast ({tmr['date']})")

is_rain_ml = ml_pred["rain_prediction"] == 1
ml_badge = '<span class="badge-rain-yes">🌧️ Rain Expected</span>' if is_rain_ml else '<span class="badge-rain-no">☀️ No Rain Expected</span>'

rain_prob_nwp = tmr.get("rain_prob_max", 0) or 0
is_rain_nwp = (tmr.get("rain_sum", 0) > 0.0) or (rain_prob_nwp > 50)
nwp_badge = '<span class="badge-rain-yes">🌧️ Rain Forecasted</span>' if is_rain_nwp else '<span class="badge-rain-no">☀️ No Rain Expected</span>'

col_pred_ml, col_pred_nwp = st.columns(2)

with col_pred_ml:
    st.markdown(f"""
    <div class="pred-card-ml">
        <div class="badge-ml">🤖 Machine Learning Prediction</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #166534; margin-bottom: 12px;">
            AI Next-Day Weather Forecast
        </div>
        <div style="margin-bottom: 14px;">
            {ml_badge}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px;">
            <div>
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Rainfall Probability</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{ml_pred['rain_probability_pct']}%</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Predicted Max Temp</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444;">{ml_pred['temp_prediction']} °C</div>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #475569; background: rgba(255,255,255,0.7); padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
            💡 <strong>Model Engine:</strong> Random Forest trained on 25 years (2000–2024) of Indian meteorological data using seasonal lag variables.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_pred_nwp:
    st.markdown(f"""
    <div class="pred-card-nwp">
        <div class="badge-nwp">🌐 Atmospheric Forecast (NWP)</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #075985; margin-bottom: 12px;">
            Numerical Weather Simulation
        </div>
        <div style="margin-bottom: 14px;">
            {nwp_badge}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px;">
            <div>
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Forecasted Rain</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{tmr.get('rain_sum', 0):.1f} <span style="font-size: 1.1rem; font-weight: 600;">mm</span></div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Expected High / Low</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{tmr['temp_max']:.1f}° <span style="font-size: 1.2rem; color: #64748b; font-weight: 600;">/ {tmr['temp_min']:.1f}°</span></div>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #475569; background: rgba(255,255,255,0.7); padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
            🌤️ <strong>Atmospheric Condition:</strong> {tmr.get('condition_icon', '⛅')} {tmr.get('condition_text', 'Fair')} &nbsp;•&nbsp; Rain chance: <strong>{rain_prob_nwp}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Consensus banner
if is_rain_ml == is_rain_nwp:
    consensus_text = f"✅ <strong>High Forecast Agreement:</strong> Both the Machine Learning model ({ml_pred['rain_probability_pct']}% chance) and Numerical Weather Prediction agree on <strong>{'Rain' if is_rain_ml else 'No Rain'}</strong> for tomorrow in {selected_city}."
else:
    consensus_text = f"ℹ️ <strong>Forecast Note:</strong> The ML model predicts <strong>{'Rain' if is_rain_ml else 'No Rain'}</strong> ({ml_pred['rain_probability_pct']}%), while Open-Meteo NWP forecasts <strong>{'Rain' if is_rain_nwp else 'Fair conditions'}</strong> ({rain_prob_nwp}%). Atmospheric dynamics are actively transitioning."

st.markdown(f"""
<div class="consensus-banner">
    {consensus_text}
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. 7-DAY WEATHER FORECAST & TRENDS
# -----------------------------------------------------------------------------
if not f_df.empty:
    st.markdown("---")
    st.markdown(f"### 📅 7-Day Weather Forecast & Temperature Trends for {selected_city}")

    # Daily cards grid
    f_cols = st.columns(len(f_df))
    for idx, (_, row) in enumerate(f_df.iterrows()):
        with f_cols[idx]:
            is_today = idx == 0
            day_label = "Today" if is_today else str(row.get("day_name", ""))[:3]
            cond_icon = row.get("condition_icon") or "🌤️"
            cond_text = row.get("condition_text") or row.get("condition", "Fair")
            t_max = float(row.get("temp_max", 0.0) or 0.0)
            t_min = float(row.get("temp_min", 0.0) or 0.0)
            rain_prob = row.get("rain_prob_max", row.get("rain_prob", 0))
            rain_prob_val = int(rain_prob) if rain_prob is not None and not pd.isna(rain_prob) else 0
            date_str = str(row.get("date", ""))
            date_display = date_str[5:] if len(date_str) >= 5 else date_str

            st.markdown(f"""
            <div class="forecast-day-card" style="{'border: 2px solid #0284c7; background: #f0f9ff;' if is_today else ''}">
                <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">{day_label}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 6px;">{date_display}</div>
                <div style="font-size: 1.8rem; margin: 4px 0;">{cond_icon}</div>
                <div style="font-size: 0.8rem; color: #334155; height: 32px; line-height: 1.2; display: flex; align-items: center; justify-content: center;">
                    {cond_text}
                </div>
                <div style="margin-top: 8px; font-weight: 800; font-size: 0.95rem; color: #ef4444;">
                    {t_max:.0f}° <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 600;">{t_min:.0f}°</span>
                </div>
                <div style="font-size: 0.72rem; color: #0284c7; margin-top: 4px;">
                    💧 {rain_prob_val}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Interactive 7-Day Plotly Chart
    fig_7d = go.Figure()
    
    # Rain bars
    fig_7d.add_trace(go.Bar(
        x=f_df["day_name"] + " (" + f_df["date"].str[5:] + ")",
        y=f_df["rain_sum"],
        name="Precipitation (mm)",
        yaxis="y2",
        marker_color="rgba(14, 165, 233, 0.4)"
    ))
    
    # Max temperature line
    fig_7d.add_trace(go.Scatter(
        x=f_df["day_name"] + " (" + f_df["date"].str[5:] + ")",
        y=f_df["temp_max"],
        name="Max Temp (°C)",
        line=dict(color="#ef4444", width=3),
        mode="lines+markers+text",
        text=[f"{v:.0f}°C" for v in f_df["temp_max"]],
        textposition="top center"
    ))
    
    # Min temperature line
    fig_7d.add_trace(go.Scatter(
        x=f_df["day_name"] + " (" + f_df["date"].str[5:] + ")",
        y=f_df["temp_min"],
        name="Min Temp (°C)",
        line=dict(color="#3b82f6", width=2.5),
        mode="lines+markers+text",
        text=[f"{v:.0f}°C" for v in f_df["temp_min"]],
        textposition="bottom center"
    ))

    fig_7d.update_layout(
        title=f"7-Day Temperature Curve & Expected Rain ({selected_city})",
        xaxis_title="Day & Date",
        yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig_7d, use_container_width=True)


# -----------------------------------------------------------------------------
# 9. INTERACTIVE WHAT-IF WEATHER SIMULATOR (OPTIONAL EXPANDER)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🎛️ Test Custom Weather Inputs (What-If AI Simulator)", expanded=False):
    st.write("Enter hypothetical weather parameters below to test how the Machine Learning model predicts rain and temperature:")

    with st.form("custom_simulator_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            s_max_temp = st.number_input("Today's Max Temperature (°C)", value=32.0, min_value=-10.0, max_value=55.0)
            s_min_temp = st.number_input("Today's Min Temperature (°C)", value=22.0, min_value=-20.0, max_value=45.0)
            s_rain_today = st.number_input("Today's Rainfall (mm)", value=2.5, min_value=0.0, max_value=500.0)
            s_wind_speed = st.number_input("Today's Max Wind Speed (km/h)", value=12.0, min_value=0.0, max_value=250.0)
        with col_s2:
            s_prev_rain = st.number_input("Yesterday's Rainfall (mm)", value=0.0, min_value=0.0, max_value=500.0)
            s_prev_temp = st.number_input("Yesterday's Max Temperature (°C)", value=31.0, min_value=-10.0, max_value=55.0)
            s_month = st.selectbox("Month", options=list(range(1, 13)), index=datetime.datetime.now().month - 1, format_func=lambda m: datetime.date(2024, m, 1).strftime("%B"))
            s_day_of_year = st.number_input("Day of Year (1–366)", min_value=1, max_value=366, value=datetime.datetime.now().timetuple().tm_yday)

        sim_submit = st.form_submit_button("Run AI Prediction", use_container_width=True)

    if sim_submit:
        sim_input = pd.DataFrame([[
            s_max_temp, s_min_temp, s_rain_today, s_wind_speed,
            s_prev_rain, s_prev_temp, s_month, s_day_of_year
        ]], columns=MODEL_FEATURE_NAMES)
        sim_res = predictor.predict(sim_input)

        c_sim1, c_sim2 = st.columns(2)
        with c_sim1:
            st.metric("Simulated Rain Tomorrow", sim_res["rain_label"], f"{sim_res['rain_probability_pct']}% probability")
        with c_sim2:
            st.metric("Simulated Max Temperature", f"{sim_res['temp_prediction']} °C")


# -----------------------------------------------------------------------------
# 10. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.82rem;'>"
    "Weather Forecasting & Prediction System • Powered by Machine Learning & Open-Meteo Global NWP"
    "</div>",
    unsafe_allow_html=True
)
