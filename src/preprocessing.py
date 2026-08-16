"""
Shared preprocessing pipeline builder for the Zomato delivery-time regression project.
Used across notebooks 02/03 (and later app.py) so every model trains on an identically
imputed/encoded feature space — the only difference between model families is whether
numeric scaling is applied.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw lat/long dropped from the modeling feature set on purpose: distance_km already captures
# the geographic signal in an interpretable way, and City captures the coarse regional grouping.
# The raw coordinates showed ~0.01-0.14 correlation with the target in notebook 1 — keeping them
# adds noise without adding much the model can't already get from distance_km + City.

NUMERIC_FEATURES = [
    "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
    "multiple_deliveries", "Vehicle_condition",
    "order_hour", "order_day_of_week", "order_month",
    "is_weekend", "is_peak_hour", "coords_corrupted",
]

CATEGORICAL_FEATURES = [
    "Weather_conditions", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "Festival", "order_period",
]

# City gets its own branch: missingness is informative (see notebook 1 addendum — target mean
# differs meaningfully when City is missing), so it's imputed with an explicit "Unknown" category
# rather than most_frequent, which would silently hide that signal inside the majority class.
CITY_FEATURE = ["City"]


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """
    scale_numeric=True  -> for linear/regularized models (Linear, Ridge, ElasticNet)
    scale_numeric=False -> for tree-based models (RF, GB, HistGB, XGBoost) — scaling is a no-op
                            for them, so skip it to save compute and keep feature importances
                            in original units.

    SimpleImputer(add_indicator=True) on the numeric block auto-adds a missingness flag for any
    numeric column that had NaNs at fit time (multiple_deliveries, Age, Ratings, and the
    order_hour-derived columns for rows where Time_Orderd was missing) — preserves the
    informative-missingness signal found in notebook 1 without hand-building flag columns.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    city_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("city", city_transformer, CITY_FEATURE),
    ])
