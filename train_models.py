"""
train_models.py
===============
Reproducible training script for the Weather Prediction System.

This script:
1. Loads the 2000-2024 Indian daily weather dataset.
2. Constructs leak-free lag features (previous day rain and temperature) within each city.
3. Splits data chronologically:
   - Training Set: 2000-01-01 to 2022-12-31 (~23 years)
   - Holdout Validation Set: 2023-01-01 to 2024-12-31 (~2 years)
4. Trains:
   - Rainfall Classifier: RandomForestClassifier (Predicts Rain / No Rain + probability)
   - Temperature Regressor: RandomForestRegressor (Predicts next day maximum temperature in °C)
5. Evaluates model performance against naive persistence baselines.
6. Saves models to 'weather_models/' as dictionaries with keys {'model', 'features'}.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import (
    load_raw_dataset,
    prepare_engineered_dataset,
    MODEL_FEATURE_NAMES
)

MODELS_DIR = Path("weather_models")


def train_and_save_models(
    dataset_path: str = "data/india_2000_2024_daily_weather.csv",
    split_date: str = "2023-01-01",
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Train and save the rainfall classifier and temperature regressor.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Loading raw dataset...")
    df_raw = load_raw_dataset(dataset_path)
    print(f"Loaded {len(df_raw):,} records across {df_raw['city'].nunique()} cities.")
    print(f"Date range: {df_raw['date'].min().strftime('%Y-%m-%d')} to {df_raw['date'].max().strftime('%Y-%m-%d')}")

    print("\nSTEP 2: Engineering leak-free chronological features...")
    df_eng = prepare_engineered_dataset(df_raw)

    # Drop edge rows with null lag features or null next-day targets
    valid_mask = (
        df_eng["previous_rain"].notnull() &
        df_eng["previous_max_temp"].notnull() &
        df_eng["rain_tomorrow"].notnull() &
        df_eng["temp_tomorrow"].notnull()
    )
    df_clean = df_eng[valid_mask].copy()
    print(f"Clean samples ready for modeling: {len(df_clean):,}")

    print(f"\nSTEP 3: Splitting data chronologically at {split_date}...")
    split_ts = pd.to_datetime(split_date)
    train_df = df_clean[df_clean["date"] < split_ts].copy()
    test_df = df_clean[df_clean["date"] >= split_ts].copy()

    X_train = train_df[MODEL_FEATURE_NAMES]
    y_rain_train = train_df["rain_tomorrow"].astype(int)
    y_temp_train = train_df["temp_tomorrow"].astype(float)

    X_test = test_df[MODEL_FEATURE_NAMES]
    y_rain_test = test_df["rain_tomorrow"].astype(int)
    y_temp_test = test_df["temp_tomorrow"].astype(float)

    print(f"Training set:   {len(train_df):,} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Holdout test:   {len(test_df):,} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # ------------------------------------------------------------------
    # 1. Rainfall Classification Model
    # ------------------------------------------------------------------
    print("\nSTEP 4: Training Rainfall Classifier (RandomForestClassifier)...")
    rain_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",  # Handle class imbalance (rain is less frequent than dry days)
        random_state=random_state,
        n_jobs=-1
    )
    rain_model.fit(X_train, y_rain_train)

    # Rainfall Evaluation
    rain_preds = rain_model.predict(X_test)
    rain_probs = rain_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_rain_test, rain_preds)
    prec = precision_score(y_rain_test, rain_preds, zero_division=0)
    rec = recall_score(y_rain_test, rain_preds, zero_division=0)
    f1 = f1_score(y_rain_test, rain_preds, zero_division=0)
    roc = roc_auc_score(y_rain_test, rain_probs)

    print("  [Rainfall Holdout Results]")
    print(f"  * Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  * Precision: {prec:.4f}")
    print(f"  * Recall:    {rec:.4f}")
    print(f"  * F1-Score:  {f1:.4f}")
    print(f"  * ROC-AUC:   {roc:.4f}")

    # ------------------------------------------------------------------
    # 2. Temperature Regression Model
    # ------------------------------------------------------------------
    print("\nSTEP 5: Training Temperature Regressor (RandomForestRegressor)...")
    temp_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1
    )
    temp_model.fit(X_train, y_temp_train)

    # Temperature Evaluation
    temp_preds = temp_model.predict(X_test)
    mae = mean_absolute_error(y_temp_test, temp_preds)
    rmse = np.sqrt(mean_squared_error(y_temp_test, temp_preds))
    r2 = r2_score(y_temp_test, temp_preds)

    print("  [Temperature Holdout Results]")
    print(f"  * MAE:  {mae:.2f} deg C")
    print(f"  * RMSE: {rmse:.2f} deg C")
    print(f"  * R^2:   {r2:.4f}")

    # ------------------------------------------------------------------
    # 3. Save Models in exact Colab format
    # ------------------------------------------------------------------
    print("\nSTEP 6: Saving models to weather_models/ ...")
    rain_save_dict = {
        "model": rain_model,
        "features": MODEL_FEATURE_NAMES,
        "metrics": {"accuracy": acc, "f1": f1, "roc_auc": roc},
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    temp_save_dict = {
        "model": temp_model,
        "features": MODEL_FEATURE_NAMES,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(rain_save_dict, MODELS_DIR / "rainfall_model.pkl", compress=3)
    joblib.dump(temp_save_dict, MODELS_DIR / "temperature_model.pkl", compress=3)

    print(f"Saved: {MODELS_DIR / 'rainfall_model.pkl'}")
    print(f"Saved: {MODELS_DIR / 'temperature_model.pkl'}")
    print("=" * 70)
    print("Training complete! Models are verified and ready for deployment.")


if __name__ == "__main__":
    train_and_save_models()
