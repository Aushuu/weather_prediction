"""
data_loader.py
==============
Data loading, validation, and feature preparation for the Weather Prediction System.

This module handles:
1. Loading the 2000-2024 Indian historical daily weather dataset.
2. Chronological sorting, lag feature generation, and leakage-free validation.
3. City metadata (coordinates for Open-Meteo API integration).
4. Extracting historical and latest available records for inference.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Metadata for the 10 Indian cities present in the dataset
CITIES_METADATA: Dict[str, Dict[str, Any]] = {
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "region": "Northern Plains",
        "state": "National Capital Territory",
        "description": "Subtropical steppe climate with hot summers and cool winters."
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "region": "Konkan Coast",
        "state": "Maharashtra",
        "description": "Tropical wet and dry climate with intense monsoon rainfall."
    },
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "region": "Eastern India",
        "state": "West Bengal",
        "description": "Tropical wet-and-dry climate influenced by Bay of Bengal depressions."
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "region": "Coromandel Coast",
        "state": "Tamil Nadu",
        "description": "Tropical wet and dry climate, receiving peak rainfall from Northeast monsoon."
    },
    "Bangalore": {
        "lat": 12.9716,
        "lon": 77.5946,
        "region": "Deccan Plateau",
        "state": "Karnataka",
        "description": "Moderate tropical savanna climate with cooler nights due to elevation."
    },
    "Hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        "region": "Central Deccan",
        "state": "Telangana",
        "description": "Tropical wet and dry climate bordering a semi-arid climate."
    },
    "Ahmedabad": {
        "lat": 23.0225,
        "lon": 72.5714,
        "region": "Western India",
        "state": "Gujarat",
        "description": "Hot semi-arid climate with high summer heat and distinct monsoon season."
    },
    "Pune": {
        "lat": 18.5204,
        "lon": 73.8567,
        "region": "Western Ghats Rain Shadow",
        "state": "Maharashtra",
        "description": "Hot semi-arid climate with pleasant winters and moderate monsoon showers."
    },
    "Jaipur": {
        "lat": 26.9124,
        "lon": 75.7873,
        "region": "North-West India",
        "state": "Rajasthan",
        "description": "Hot semi-arid climate near Thar Desert with wide diurnal temperature shifts."
    },
    "Lucknow": {
        "lat": 26.8467,
        "lon": 80.9462,
        "region": "Gangetic Plain",
        "state": "Uttar Pradesh",
        "description": "Humid subtropical climate with dry, cold winters and sweltering summer monsoons."
    }
}

# The 8 exact features expected by the trained Random Forest models
MODEL_FEATURE_NAMES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum",
    "wind_speed_10m_max",
    "previous_rain",
    "previous_max_temp",
    "month",
    "day_of_year"
]

# WMO Weather Interpretation Codes (WMO 4501)
WMO_CODE_DESCRIPTIONS: Dict[int, Tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    71: ("Slight snow fall", "🌨️"),
    73: ("Moderate snow fall", "🌨️"),
    75: ("Heavy snow fall", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def get_wmo_info(code: Optional[int]) -> Tuple[str, str]:
    """Return human-readable description and emoji icon for a WMO weather code."""
    if code is None or pd.isna(code):
        return ("Unknown Weather", "❓")
    return WMO_CODE_DESCRIPTIONS.get(int(code), ("Variable Conditions", "🌤️"))


def load_raw_dataset(file_path: str = "data/india_2000_2024_daily_weather.csv") -> pd.DataFrame:
    """
    Load the historical CSV dataset and apply initial cleaning.
    
    Steps:
    - Verifies file existence.
    - Parses dates and sorts chronologically per city.
    - Removes any duplicate date entries per city.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please ensure the CSV file is in the data/ folder."
        )

    df = pd.read_csv(path)
    
    # Required columns verification
    expected_cols = {"city", "date", "temperature_2m_max", "temperature_2m_min", "rain_sum", "wind_speed_10m_max"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Standardize dates and sort strictly chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Drop duplicate records if any exist
    df = df.drop_duplicates(subset=["city", "date"]).reset_index(drop=True)

    return df


def prepare_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag features and target variables in a strictly leak-free chronological manner.
    
    Lag Features:
    - previous_rain: rain_sum of day t-1
    - previous_max_temp: temperature_2m_max of day t-1
    - month: month of day t (1..12)
    - day_of_year: day of year of day t (1..366)
    
    Targets for day t+1:
    - rain_tomorrow: 1 if rain_sum(t+1) > 0.0 else 0
    - temp_tomorrow: temperature_2m_max(t+1)
    """
    df = df.copy()

    # Create lag features strictly within each city group to avoid city boundary bleeding
    df["previous_rain"] = df.groupby("city")["rain_sum"].shift(1)
    df["previous_max_temp"] = df.groupby("city")["temperature_2m_max"].shift(1)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Next-day targets for evaluation/retraining
    df["rain_tomorrow"] = (df.groupby("city")["rain_sum"].shift(-1) > 0.0).astype("Int64")
    df["temp_tomorrow"] = df.groupby("city")["temperature_2m_max"].shift(-1)

    return df


def get_city_records(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter records for a specific city, sorted by date."""
    city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)
    if city_df.empty:
        raise ValueError(f"City '{city}' not found in dataset. Available: {list(CITIES_METADATA.keys())}")
    return city_df


def get_latest_inference_features(df: pd.DataFrame, city: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract the most recent record for the chosen city and construct the exact 8-feature
    input DataFrame ready for the trained Random Forest models.
    
    Returns:
    - X_input: DataFrame containing the 8 features in the exact expected order.
    - metadata: Dictionary containing record date, city, today's observations, etc.
    """
    city_df = get_city_records(df, city)
    
    # Ensure lag features are present
    if "previous_rain" not in city_df.columns:
        city_df = prepare_engineered_dataset(city_df)

    # Get the latest row with non-null lag values
    valid_rows = city_df.dropna(subset=["previous_rain", "previous_max_temp"])
    if valid_rows.empty:
        raise ValueError(f"No valid records found for city {city}.")

    latest_row = valid_rows.iloc[-1]
    
    feature_dict = {
        "temperature_2m_max": float(latest_row["temperature_2m_max"]),
        "temperature_2m_min": float(latest_row["temperature_2m_min"]),
        "rain_sum": float(latest_row["rain_sum"]),
        "wind_speed_10m_max": float(latest_row["wind_speed_10m_max"]),
        "previous_rain": float(latest_row["previous_rain"]),
        "previous_max_temp": float(latest_row["previous_max_temp"]),
        "month": int(latest_row["month"]),
        "day_of_year": int(latest_row["day_of_year"])
    }

    X_input = pd.DataFrame([[feature_dict[f] for f in MODEL_FEATURE_NAMES]], columns=MODEL_FEATURE_NAMES)

    meta = {
        "city": city,
        "date": latest_row["date"],
        "date_str": latest_row["date"].strftime("%Y-%m-%d"),
        "features": feature_dict,
        "weather_code": latest_row.get("weather_code", None),
        "apparent_temp_max": latest_row.get("apparent_temperature_max", None),
        "apparent_temp_min": latest_row.get("apparent_temperature_min", None)
    }

    return X_input, meta


def get_historical_record_for_date(
    df: pd.DataFrame, city: str, target_date: pd.Timestamp
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Retrieve features for a specific historical date and the actual observed next-day weather
    (if available in the dataset) to allow users to verify model performance on past dates.
    """
    city_df = get_city_records(df, city)
    if "previous_rain" not in city_df.columns:
        city_df = prepare_engineered_dataset(city_df)

    match = city_df[city_df["date"] == target_date]
    if match.empty:
        return None, None, None

    row_idx = match.index[0]
    row = match.iloc[0]

    if pd.isna(row["previous_rain"]) or pd.isna(row["previous_max_temp"]):
        return None, None, None

    feature_dict = {
        "temperature_2m_max": float(row["temperature_2m_max"]),
        "temperature_2m_min": float(row["temperature_2m_min"]),
        "rain_sum": float(row["rain_sum"]),
        "wind_speed_10m_max": float(row["wind_speed_10m_max"]),
        "previous_rain": float(row["previous_rain"]),
        "previous_max_temp": float(row["previous_max_temp"]),
        "month": int(row["month"]),
        "day_of_year": int(row["day_of_year"])
    }

    X_input = pd.DataFrame([[feature_dict[f] for f in MODEL_FEATURE_NAMES]], columns=MODEL_FEATURE_NAMES)

    meta = {
        "city": city,
        "date": row["date"],
        "date_str": row["date"].strftime("%Y-%m-%d"),
        "features": feature_dict,
        "weather_code": row.get("weather_code", None)
    }

    # Ground truth for next day if available
    ground_truth = None
    if row_idx + 1 < len(city_df):
        next_row = city_df.iloc[row_idx + 1]
        if (next_row["date"] - row["date"]).days == 1:
            ground_truth = {
                "date": next_row["date"],
                "date_str": next_row["date"].strftime("%Y-%m-%d"),
                "rain_sum": float(next_row["rain_sum"]),
                "rained": bool(next_row["rain_sum"] > 0.0),
                "temperature_2m_max": float(next_row["temperature_2m_max"]),
                "temperature_2m_min": float(next_row["temperature_2m_min"]),
                "weather_code": next_row.get("weather_code", None)
            }

    return X_input, meta, ground_truth
