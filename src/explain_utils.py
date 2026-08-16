"""
Explainability utilities for the Zomato delivery-time regression project.
predict_delivery() combines the final model, calibrated quantile models, and SHAP into a single
user-facing explanation, per plan Section 21. Used by notebook 03 and later app.py.
"""
import pandas as pd


def predict_delivery(order: dict, model, quantile_models: dict, correction_margin: float,
                      explainer, feature_names, top_n_factors: int = 3) -> dict:
    """
    order: dict of raw feature values matching the training schema (single delivery).
    model: fitted final point-estimate pipeline (e.g. tuned XGBoost).
    quantile_models: {"lower_10": pipeline, "upper_90": pipeline} — calibrated via correction_margin.
    explainer: shap.TreeExplainer built on model.named_steps["model"].
    feature_names: output of model.named_steps["preprocess"].get_feature_names_out().

    Returns point estimate, calibrated 80% interval, and top SHAP factors in each direction.
    Note: SHAP factors are row-specific and contextual — a factor's direction here can differ
    from that feature's average/global relationship with the target (see notebook 3 case studies).
    """
    order_df = pd.DataFrame([order])

    point_pred = model.predict(order_df)[0]
    lower = quantile_models["lower_10"].predict(order_df)[0] - correction_margin
    upper = quantile_models["upper_90"].predict(order_df)[0] + correction_margin

    order_transformed = model.named_steps["preprocess"].transform(order_df)
    shap_vals = explainer.shap_values(order_transformed)[0]
    shap_series = pd.Series(shap_vals, index=feature_names).sort_values()

    factors_decreasing = shap_series.head(top_n_factors)
    factors_increasing = shap_series.tail(top_n_factors)[::-1]

    return {
        "predicted_minutes": round(float(point_pred), 1),
        "interval_minutes": (round(float(lower), 1), round(float(upper), 1)),
        "factors_increasing": [(name, round(float(val), 2)) for name, val in factors_increasing.items()],
        "factors_decreasing": [(name, round(float(val), 2)) for name, val in factors_decreasing.items()],
    }
