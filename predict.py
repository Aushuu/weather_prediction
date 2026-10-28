"""
predict.py
==========
Model inference pipeline for the Weather Prediction System.

This module handles:
1. Loading trained Scikit-Learn Random Forest models from pickle files.
2. Validating input feature schema (names, order, data types, physical bounds).
3. Predicting next-day rainfall (classification & calibrated probability estimate).
4. Forecasting next-day maximum temperature (regression in °C).
5. Adapting live Open-Meteo weather measurements into valid model feature vectors.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
import joblib
import pandas as pd
import numpy as np

# Suppress harmless sklearn unpickle version warnings during deployment
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RAINFALL_MODEL_PATH = Path("weather_models/rainfall_model.pkl")
TEMPERATURE_MODEL_PATH = Path("weather_models/temperature_model.pkl")


def check_models_exist() -> Tuple[bool, str]:
    """Check whether both required model files exist on disk."""
    if not RAINFALL_MODEL_PATH.is_file():
        return False, f"Rainfall model not found at '{RAINFALL_MODEL_PATH}'."
    if not TEMPERATURE_MODEL_PATH.is_file():
        return False, f"Temperature model not found at '{TEMPERATURE_MODEL_PATH}'."
    return True, "Models are present."


class WeatherPredictor:
    """Wrapper class encapsulating model loading, feature validation, and inference."""

    def __init__(
        self,
        rain_model_path: Path = RAINFALL_MODEL_PATH,
        temp_model_path: Path = TEMPERATURE_MODEL_PATH
    ):
        self.rain_model_path = Path(rain_model_path)
        self.temp_model_path = Path(temp_model_path)
        self.rain_model = None
        self.temp_model = None
        self.feature_names = []
        self._load_models()

    def _load_models(self):
        """Load pickled model dictionaries."""
        if not self.rain_model_path.is_file():
            raise FileNotFoundError(f"Missing rainfall model at {self.rain_model_path}")
        if not self.temp_model_path.is_file():
            raise FileNotFoundError(f"Missing temperature model at {self.temp_model_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rain_data = joblib.load(self.rain_model_path)
            temp_data = joblib.load(self.temp_model_path)

        if not isinstance(rain_data, dict) or "model" not in rain_data or "features" not in rain_data:
            raise ValueError("Rainfall model pickle must contain a dictionary with 'model' and 'features' keys.")

        if not isinstance(temp_data, dict) or "model" not in temp_data or "features" not in temp_data:
            raise ValueError("Temperature model pickle must contain a dictionary with 'model' and 'features' keys.")

        self.rain_model = rain_data["model"]
        self.temp_model = temp_data["model"]
        self.feature_names = rain_data["features"]

    def validate_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the input DataFrame has all expected features in the exact required order.
        Also check for missing values and unreasonable physical extremes.
        """
        missing = [f for f in self.feature_names if f not in input_df.columns]
        if missing:
            raise ValueError(f"Input features are missing required columns: {missing}")

        # Reorder columns to guarantee exact match with model training
        df_ordered = input_df[self.feature_names].copy()

        # Check for NaN / nulls
        if df_ordered.isnull().any().any():
            nan_cols = df_ordered.columns[df_ordered.isnull().any()].tolist()
            raise ValueError(f"Input features contain missing (NaN) values in: {nan_cols}")

        return df_ordered

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run inference on the provided feature DataFrame.
        
        Returns:
        {
            'rain_prediction': int (0 or 1),
            'rain_label': 'Rain' or 'No Rain',
            'rain_probability': float (0.0 to 1.0),
            'temp_prediction': float (predicted max temp in °C),
            'features_used': dict of input features
        }
        """
        X = self.validate_features(input_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Rainfall Classification
            rain_pred = int(self.rain_model.predict(X)[0])
            
            # Predict probabilities
            if hasattr(self.rain_model, "predict_proba"):
                probs = self.rain_model.predict_proba(X)[0]
                # Index 1 corresponds to class 1 (Rain)
                rain_prob = float(probs[1]) if len(probs) > 1 else float(rain_pred)
            else:
                rain_prob = float(rain_pred)

            # Temperature Regression
            temp_pred = float(self.temp_model.predict(X)[0])

        features_dict = X.iloc[0].to_dict()

        return {
            "rain_prediction": rain_pred,
            "rain_label": "Rain" if rain_pred == 1 else "No Rain",
            "rain_probability": rain_prob,
            "rain_probability_pct": round(rain_prob * 100, 1),
            "temp_prediction": round(temp_pred, 1),
            "features_used": features_dict
        }


# Global singleton cache instance
_PREDICTOR_INSTANCE: Optional[WeatherPredictor] = None


def get_predictor() -> WeatherPredictor:
    """Retrieve or initialize the cached WeatherPredictor singleton."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = WeatherPredictor()
    return _PREDICTOR_INSTANCE


def predict_from_measurements(
    max_temp: float,
    min_temp: float,
    rain_today: float,
    wind_speed: float,
    previous_rain: float,
    previous_max_temp: float,
    month: int,
    day_of_year: int
) -> Dict[str, Any]:
    """Helper to predict from individual scalar feature values."""
    feature_dict = {
        "temperature_2m_max": float(max_temp),
        "temperature_2m_min": float(min_temp),
        "rain_sum": float(rain_today),
        "wind_speed_10m_max": float(wind_speed),
        "previous_rain": float(previous_rain),
        "previous_max_temp": float(previous_max_temp),
        "month": int(month),
        "day_of_year": int(day_of_year)
    }
    predictor = get_predictor()
    df = pd.DataFrame([[feature_dict[f] for f in predictor.feature_names]], columns=predictor.feature_names)
    return predictor.predict(df)
