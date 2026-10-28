"""
evaluation.py
=============
Model evaluation and baseline benchmarking module.

This module provides:
1. Chronological train/holdout split (Train: 2000-2022, Holdout: 2023-2024).
2. Strict leakage prevention (lag features computed per city, targets shifted into next day).
3. Standard ML metrics for Rainfall (Accuracy, Precision, Recall, F1, ROC-AUC)
   and Temperature (MAE, RMSE, R²).
4. Naive baseline comparisons:
   - Persistence baseline (Tomorrow = Today)
   - Majority class baseline (Always predict No Rain)
   - Climatology baseline (Historical city-month average temperature)
5. Standalone CLI evaluation runner and cached report generator for the UI.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import (
    load_raw_dataset,
    prepare_engineered_dataset,
    MODEL_FEATURE_NAMES
)
from predict import get_predictor


def create_chronological_split(
    df: pd.DataFrame,
    split_date: str = "2023-01-01"
) -> tuple:
    """
    Split the dataset chronologically at split_date.
    Records before split_date are training data; records on or after are holdout test data.
    """
    df_eng = prepare_engineered_dataset(df)
    
    # Filter rows where both lag features and targets are valid
    valid_mask = (
        df_eng["previous_rain"].notnull() &
        df_eng["previous_max_temp"].notnull() &
        df_eng["rain_tomorrow"].notnull() &
        df_eng["temp_tomorrow"].notnull()
    )
    clean_df = df_eng[valid_mask].copy()

    split_ts = pd.to_datetime(split_date)
    train_df = clean_df[clean_df["date"] < split_ts].copy()
    test_df = clean_df[clean_df["date"] >= split_ts].copy()

    return train_df, test_df


def evaluate_models_on_holdout(
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    city: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate the loaded models on the holdout test set (optionally filtered by city)
    and compare against naive persistence & climatology baselines.
    """
    eval_df = test_df.copy()
    if city:
        eval_df = eval_df[eval_df["city"] == city].copy()

    if eval_df.empty:
        raise ValueError("Evaluation DataFrame is empty for the specified filters.")

    predictor = get_predictor()
    X_test = eval_df[MODEL_FEATURE_NAMES]
    y_rain_true = eval_df["rain_tomorrow"].astype(int).values
    y_temp_true = eval_df["temp_tomorrow"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Model predictions
        rain_pred = predictor.rain_model.predict(X_test)
        if hasattr(predictor.rain_model, "predict_proba"):
            rain_proba = predictor.rain_model.predict_proba(X_test)[:, 1]
        else:
            rain_proba = rain_pred.astype(float)
        temp_pred = predictor.temp_model.predict(X_test)

    # -------------------------------------------------------------
    # 1. Rainfall Classification Metrics
    # -------------------------------------------------------------
    rain_metrics = {
        "accuracy": float(accuracy_score(y_rain_true, rain_pred)),
        "precision": float(precision_score(y_rain_true, rain_pred, zero_division=0)),
        "recall": float(recall_score(y_rain_true, rain_pred, zero_division=0)),
        "f1": float(f1_score(y_rain_true, rain_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_rain_true, rain_proba)),
        "confusion_matrix": confusion_matrix(y_rain_true, rain_pred).tolist(),
        "support": int(len(y_rain_true)),
        "rain_days_pct": float(np.mean(y_rain_true) * 100)
    }

    # Baselines for Rainfall
    # Persistence: predict tomorrow's rain = today's rain (rain_sum > 0)
    baseline_rain_persistence = (eval_df["rain_sum"].values > 0.0).astype(int)
    # Majority class: always predict 0 (No Rain)
    baseline_rain_majority = np.zeros_like(y_rain_true)

    rain_baselines = {
        "persistence_accuracy": float(accuracy_score(y_rain_true, baseline_rain_persistence)),
        "persistence_f1": float(f1_score(y_rain_true, baseline_rain_persistence, zero_division=0)),
        "majority_accuracy": float(accuracy_score(y_rain_true, baseline_rain_majority)),
        "majority_f1": float(f1_score(y_rain_true, baseline_rain_majority, zero_division=0)),
    }

    # -------------------------------------------------------------
    # 2. Temperature Regression Metrics
    # -------------------------------------------------------------
    temp_mae = float(mean_absolute_error(y_temp_true, temp_pred))
    temp_rmse = float(np.sqrt(mean_squared_error(y_temp_true, temp_pred)))
    temp_r2 = float(r2_score(y_temp_true, temp_pred))

    temp_metrics = {
        "mae": temp_mae,
        "rmse": temp_rmse,
        "r2": temp_r2,
        "support": int(len(y_temp_true))
    }

    # Baselines for Temperature
    # Persistence: predict tomorrow's max temp = today's max temp
    baseline_temp_persistence = eval_df["temperature_2m_max"].values
    pers_mae = float(mean_absolute_error(y_temp_true, baseline_temp_persistence))
    pers_rmse = float(np.sqrt(mean_squared_error(y_temp_true, baseline_temp_persistence)))
    pers_r2 = float(r2_score(y_temp_true, baseline_temp_persistence))

    # Climatology Baseline: monthly mean temperature per city from training data (or test if train not provided)
    ref_df = train_df if train_df is not None else eval_df
    monthly_clim = ref_df.groupby(["city", "month"])["temperature_2m_max"].mean().to_dict()
    clim_preds = np.array([
        monthly_clim.get((c, m), 30.0)
        for c, m in zip(eval_df["city"].values, eval_df["month"].values)
    ])
    clim_mae = float(mean_absolute_error(y_temp_true, clim_preds))
    clim_rmse = float(np.sqrt(mean_squared_error(y_temp_true, clim_preds)))
    clim_r2 = float(r2_score(y_temp_true, clim_preds))

    temp_baselines = {
        "persistence_mae": pers_mae,
        "persistence_rmse": pers_rmse,
        "persistence_r2": pers_r2,
        "climatology_mae": clim_mae,
        "climatology_rmse": clim_rmse,
        "climatology_r2": clim_r2
    }

    return {
        "city": city or "All Cities (Consolidated)",
        "rain_metrics": rain_metrics,
        "rain_baselines": rain_baselines,
        "temp_metrics": temp_metrics,
        "temp_baselines": temp_baselines,
        "date_range": (
            eval_df["date"].min().strftime("%Y-%m-%d"),
            eval_df["date"].max().strftime("%Y-%m-%d")
        )
    }


def print_evaluation_report(results: Dict[str, Any]):
    """Format and print an executive model evaluation summary."""
    print("=" * 70)
    print(f" WEATHER MODEL EVALUATION REPORT: {results['city'].upper()}")
    print(f" Holdout Evaluation Period: {results['date_range'][0]} to {results['date_range'][1]}")
    print("=" * 70)

    rm = results["rain_metrics"]
    rb = results["rain_baselines"]
    print("\n[RAINFALL PREDICTION (CLASSIFIER)]")
    print(f"  * Total Test Samples: {rm['support']} (Rain occurrence: {rm['rain_days_pct']:.1f}%)")
    print(f"  * Random Forest Accuracy:  {rm['accuracy']:.3f} ({rm['accuracy']*100:.1f}%)")
    print(f"  * Random Forest Precision: {rm['precision']:.3f}")
    print(f"  * Random Forest Recall:    {rm['recall']:.3f}")
    print(f"  * Random Forest F1-Score:  {rm['f1']:.3f}")
    print(f"  * Random Forest ROC-AUC:   {rm['roc_auc']:.3f}")
    print("\n  Baseline Comparison:")
    print(f"    - Persistence Model Accuracy: {rb['persistence_accuracy']:.3f} (F1: {rb['persistence_f1']:.3f})")
    print(f"    - Majority Class Accuracy:   {rb['majority_accuracy']:.3f} (F1: {rb['majority_f1']:.3f})")

    tm = results["temp_metrics"]
    tb = results["temp_baselines"]
    print("\n[MAXIMUM TEMPERATURE FORECASTING (REGRESSOR)]")
    print(f"  * Random Forest MAE:   {tm['mae']:.2f} deg C")
    print(f"  * Random Forest RMSE:  {tm['rmse']:.2f} deg C")
    print(f"  * Random Forest R^2:   {tm['r2']:.3f}")
    print("\n  Baseline Comparison:")
    print(f"    - Persistence Baseline MAE:   {tb['persistence_mae']:.2f} deg C (RMSE: {tb['persistence_rmse']:.2f} deg C, R^2: {tb['persistence_r2']:.3f})")
    print(f"    - Monthly Climatology MAE:    {tb['climatology_mae']:.2f} deg C (RMSE: {tb['climatology_rmse']:.2f} deg C, R^2: {tb['climatology_r2']:.3f})")
    print("=" * 70)


if __name__ == "__main__":
    print("Loading historical dataset for chronological evaluation...")
    df = load_raw_dataset()
    train_df, test_df = create_chronological_split(df, split_date="2023-01-01")
    print(f"Train samples (2000-2022): {len(train_df)}")
    print(f"Holdout samples (2023-2024): {len(test_df)}")

    res = evaluate_models_on_holdout(test_df, train_df)
    print_evaluation_report(res)
