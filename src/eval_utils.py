"""
Evaluation utilities for the Zomato delivery-time regression project.
Shared across the modeling, tuning, and analysis notebooks (02, 03).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error


def compute_metrics(y_true, y_pred) -> dict:
    """Standard regression metrics for delivery-time prediction, per plan Section 13."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_true - y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "within_5min_pct": (abs_err <= 5).mean() * 100,
        "within_10min_pct": (abs_err <= 10).mean() * 100,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: {model_name: {"metrics": {...}, "train_time_sec": float}}
    Returns a tidy comparison DataFrame sorted by MAE (ascending), per plan Section 14.
    """
    rows = []
    for model_name, res in results.items():
        rows.append({"Model": model_name, **res["metrics"], "Training Time (s)": res.get("train_time_sec")})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
