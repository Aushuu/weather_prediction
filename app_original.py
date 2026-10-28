import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="centered"
)

# Load trained models
rain_data = joblib.load("weather_models/rainfall_model.pkl")
temp_data = joblib.load("weather_models/temperature_model.pkl")

rain_model = rain_data["model"]
temp_model = temp_data["model"]
features = rain_data["features"]

st.title("🌦️ Weather Prediction System")
st.write("Predict tomorrow's rainfall and maximum temperature.")
st.info(
    "This is a machine learning demonstration. "
    "Predictions are estimates, not official weather forecasts."
)

st.subheader("Enter the latest available weather measurements")

with st.form("weather_form"):
    col1, col2 = st.columns(2)

    with col1:
        max_temp = st.number_input(
            "Today's maximum temperature (°C)",
            value=30.0, min_value=-20.0, max_value=60.0
        )
        min_temp = st.number_input(
            "Today's minimum temperature (°C)",
            value=20.0, min_value=-30.0, max_value=50.0
        )
        rain_today = st.number_input(
            "Today's rainfall (mm)",
            value=0.0, min_value=0.0, max_value=500.0
        )
        wind_speed = st.number_input(
            "Today's maximum wind speed (km/h)",
            value=10.0, min_value=0.0, max_value=300.0
        )

    with col2:
        previous_rain = st.number_input(
            "Previous day's rainfall (mm)",
            value=0.0, min_value=0.0, max_value=500.0
        )
        previous_temp = st.number_input(
            "Previous day's maximum temperature (°C)",
            value=29.0, min_value=-20.0, max_value=60.0
        )
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=9,
            format_func=lambda m: pd.Timestamp(
                2025, m, 1
            ).strftime("%B")
        )
        day_of_year = st.number_input(
            "Day of year (1–366)",
            min_value=1, max_value=366, value=280
        )

    submitted = st.form_submit_button("Predict Tomorrow's Weather")

if submitted:
    # Build input using the exact training feature names
    input_values = {
        "temperature_2m_max": max_temp,
        "temperature_2m_min": min_temp,
        "rain_sum": rain_today,
        "wind_speed_10m_max": wind_speed,
        "previous_rain": previous_rain,
        "previous_max_temp": previous_temp,
        "month": month,
        "day_of_year": day_of_year
    }

    X_new = pd.DataFrame(
        [[input_values[f] for f in features]],
        columns=features
    )

    rain_prediction = rain_model.predict(X_new)[0]
    rain_probability = rain_model.predict_proba(X_new)[0][1]
    temp_prediction = temp_model.predict(X_new)[0]

    st.subheader("Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Rain tomorrow",
            "Yes 🌧️" if rain_prediction == 1 else "No ☀️"
        )
        st.write(
            f"Model-estimated rain probability: "
            f"{rain_probability:.1%}"
        )

    with result2:
        st.metric(
            "Predicted maximum temperature",
            f"{temp_prediction:.1f} °C"
        )

    st.caption(
        "Rain probability is the model's estimate, not a "
        "calibrated probability from an operational forecast."
    )
