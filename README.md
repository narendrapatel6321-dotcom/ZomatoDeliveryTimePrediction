# Zomato Delivery Time Prediction

A regression pipeline that predicts food delivery time (in minutes) **at the moment an order is placed** — before a courier is even assigned — using order, courier, geographic, weather, traffic, and temporal signals. Built on a public 45,584-row Kaggle dataset spanning Feb 11 – Apr 6, 2022.

The project goes beyond a standard point-estimate regression: it includes a leakage audit, fair (non-default) hyperparameter comparisons across 7 algorithms, calibrated prediction intervals via conformalized quantile regression, SHAP-based per-order explanations, and an explicit temporal-generalization stress test.

## 1. Project Overview

**Business problem**: A delivery platform needs to quote a customer an accurate ETA the instant an order is confirmed — before pickup time, courier assignment confirmation, or any downstream operational data exists. Point estimates alone are insufficient for this: a single number of "37 minutes" hides how confident the system actually is.

**Solution**: A tuned XGBoost regressor trained exclusively on features available at order-placement time, paired with a conformalized quantile regression layer that produces calibrated 80% prediction intervals (e.g. *"37 minutes, likely range 31–46"*) instead of a bare point estimate. A SHAP-backed explanation function (`predict_delivery()`) surfaces the top factors driving each individual prediction.

**Result**: MAE of 3.17 minutes (R² = 0.82) on a held-out test set, with 98.5% of predictions landing within ±10 minutes of the actual delivery time — against a naive mean-baseline MAE of 7.69 minutes.

## 2. Tech Stack

- **Data manipulation**: Pandas, NumPy
- **Modeling**: scikit-learn (`Pipeline`, `ColumnTransformer`, linear models, ensemble models), XGBoost
- **Hyperparameter optimization**: Optuna (TPE sampler)
- **Explainability**: SHAP (`TreeExplainer`), scikit-learn permutation importance
- **Uncertainty quantification**: XGBoost native quantile regression (`reg:quantileerror`) + a custom conformalized quantile regression (CQR) calibration layer
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab, Google Drive (artifact persistence), Kaggle API
- **Deployment (planned)**: Streamlit

## 3. Data Engineering & Preprocessing

### Data quality issues found and resolved
| Issue | Resolution |
|---|---|
| 431 rows with sign-flipped (negative) GPS coordinates | Corrected via `abs()` — India's coordinates are all positive, so this was a recoverable sign error |
| 3,640 rows (8.0%) with unrecoverable `(0, 0)` restaurant coordinates | Kept in the dataset (not dropped); `distance_km` set to `NaN`, flagged via a `coords_corrupted` indicator column, imputed downstream inside the pipeline |
| 3 mixed time-string formats in `Time_Orderd` / `Time_Order_picked` (`HH:MM`, `HH:MM:SS` with midnight rollover, Excel day-fraction serials) | Single regex-based parser normalizing all formats to minutes-since-midnight |
| 53 rows with `Delivery_person_Ratings = 6.0` (invalid on a 0–5 scale) | Set to `NaN` and imputed — treated as a hard sentinel, not boundary noise, so clipping to 5.0 was rejected as dishonest |
| `City` (2.6% missing) and `multiple_deliveries` (2.2% missing) | Missingness confirmed **informative**, not random — target mean differs by 3.6–4.3 minutes between missing/present groups. `City` imputed with an explicit `"Unknown"` category rather than mode; `multiple_deliveries` imputed via `SimpleImputer(add_indicator=True)` to preserve the missingness signal as a model feature |

### Feature engineering
- **Geographic**: Haversine great-circle distance (`distance_km`) computed from restaurant/delivery coordinates, validated against sanity checks rather than trusted blindly
- **Temporal**: hour, day-of-week, month, weekend flag, and a 6-bucket `order_period` (morning lull / lunch peak / afternoon lull / evening / dinner peak / late night) — **thresholds derived empirically from mean delivery time per hour**, not arbitrarily chosen
- **Raw lat/long dropped** from the final feature set — near-zero linear correlation with the target (≤0.014) and redundant with `distance_km` + `City`; kept only as the source for distance computation
- **Categorical encoding**: `OneHotEncoder` inside a `ColumnTransformer`, fit train-fold-only inside an `sklearn.Pipeline` to prevent leakage

### Feature leakage audit
Every feature was explicitly classified as available-at-order-time vs. leakage. `Time_Order_picked` and any feature derived from it (`pickup_time_min`, `prep_time_min`) were **excluded** — that information is only known after a courier has already picked up the order, which is after the prediction moment this project targets. Correlation of all retained numeric features against the target was sanity-checked post-hoc (max 0.39) to confirm no hidden leakage slipped through.

### Feature-group ablation (evidence for what actually matters)
| Feature set | Validation MAE | Δ vs. previous |
|---|---|---|
| Raw features only | 3.969 | — |
| + Geographic | 3.250 | **−18.4%** |
| + Temporal | 3.239 | −0.3% |
| + Hand-crafted interaction terms (distance×traffic, distance×weather) | 3.256 | **+0.5% (worse)** |

Hand-crafted interaction features were dropped from the final model — tree-based learners already capture these interactions implicitly through split combinations, confirmed via SHAP dependence plots showing exactly this interaction emerging on its own.

## 4. Model Tournament

**Baselines**: mean predictor, median predictor, and a minimal distance+traffic linear model — established as the bar every real model had to clear.

**Algorithms evaluated**: Linear Regression, Ridge (`RidgeCV`), ElasticNet (`ElasticNetCV`), Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost.

**Fair-comparison methodology**: initial runs on library defaults produced a misleading picture — `GradientBoostingRegressor`'s default `max_depth=3` and `ElasticNet`'s default `alpha=1.0` were artificially handicapping those models relative to peers with more generous defaults. All models were re-run with deliberately chosen, comparably-generous hyperparameters (matched training budgets for boosting methods, CV-selected regularization strength for linear models) before any model was ruled out.

**Hyperparameter tuning**: Optuna (TPE sampler, 50 trials each, validation MAE objective) applied to the top 2 candidates only — **XGBoost and HistGradientBoosting**. Random Forest was excluded from tuning despite a marginally top MAE: it took ~20x longer to train (71.7s vs. 3.6–3.7s) for a performance difference (0.008 MAE) smaller than noise — a poor use of a 50-trial tuning budget.

Tuned search space (XGBoost): `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`.

## 5. Results & Evaluation

### Model comparison (validation set, fair hyperparameters)
| Model | MAE | RMSE | R² | Within ±10 min | Training Time |
|---|---|---|---|---|---|
| Mean baseline | 7.686 | 9.508 | -0.000 | 68.1% | <0.01s |
| Median baseline | 7.658 | 9.512 | -0.001 | 71.9% | <0.01s |
| Distance + Traffic (linear) | 6.684 | 8.359 | 0.227 | 76.7% | 0.05s |
| ElasticNet (CV-tuned) | 4.860 | 6.100 | 0.588 | 90.5% | 4.5s |
| Ridge (CV-tuned) | 4.840 | 6.074 | 0.592 | 90.6% | 0.5s |
| Linear Regression | 4.840 | 6.074 | 0.592 | 90.6% | 0.2s |
| Gradient Boosting | 3.324 | 4.169 | 0.808 | 98.1% | 27.8s |
| Random Forest | 3.223 | 4.086 | 0.815 | 98.1% | 71.7s |
| XGBoost (untuned) | 3.231 | 4.057 | 0.818 | 98.4% | 3.6s |
| HistGradientBoosting (untuned) | 3.231 | 4.055 | 0.818 | 98.3% | 3.7s |
| **XGBoost (Optuna-tuned)** | **3.158** | — | — | — | — |
| HistGradientBoosting (Optuna-tuned) | 3.160 | — | — | — | — |

### Final model: XGBoost (Optuna-tuned)
XGBoost and HistGradientBoosting finished essentially tied post-tuning (validation MAE 3.158 vs. 3.160; mean absolute MAE difference of only 0.034 minutes across 19 error segments — traffic, weather, city, time-of-day). XGBoost was selected on practical grounds: marginally better overall MAE and more mature SHAP/deployment tooling. HistGradientBoosting is documented as an equally valid alternative, not a rejected one.

### Test set performance (final, held out until this evaluation)
| Split | MAE | RMSE | R² | MedAE | Within ±5 min | Within ±10 min |
|---|---|---|---|---|---|---|
| **Random split (test)** | **3.173** | **4.006** | **0.818** | **2.686** | **80.0%** | **98.5%** |
| Temporal split (test — later, unseen period) | 3.775 | 4.925 | 0.732 | 3.076 | 73.7% | 95.1% |

**Temporal generalization**: performance degrades 19% (relative) on genuinely future, unseen time periods. Root cause identified: `distance_km` drifts ~7% higher between the earlier training period (9.66 km mean) and the later test period (10.35 km mean), while every other numeric feature stays stable — Festival/traffic/weather distributions were checked and ruled out as causes. This is reported as a real limitation, not smoothed over: a model this accurate on historical data can still require periodic retraining against distributional drift in production.

### Prediction intervals (conformalized quantile regression)
Raw XGBoost quantile models (10th/50th/90th percentile) under-covered systematically (71.8% actual vs. 80% target) — the point-model's tuned regularization was over-shrinking the tail quantiles. A CQR calibration layer, fit on the validation set only, corrected this:

| Metric | Raw quantile model | CQR-calibrated |
|---|---|---|
| Test set coverage | — | **79.9%** (target: 80%) |
| Test set avg. interval width | — | 10.21 min |

### Explainability findings
- **Geographic distance is the dominant driver** (18.4% relative MAE reduction alone).
- **Festival status has the largest single categorical effect** (+20 min mean) but was *underweighted by permutation importance* due to severe class imbalance (~2% positive) — SHAP and native gain-based importance both correctly surfaced it; a documented lesson on not trusting permutation importance alone for rare categorical features.
- **Courier age shows a sharp threshold effect around 30**, not a linear relationship — visible directly in SHAP dependence plots.
- **Compound missing data degrades reliability**: rows missing both `Road_traffic_density` and `Weather_conditions` simultaneously (which happens almost exclusively together, suggesting a shared root cause) show 71% higher mean absolute error than the rest of the test set.

## 6. How to Run

This project runs in **Google Colab** with **Google Drive** for artifact persistence and **GitHub** for source control of the `src/` package — it is not a local `python train.py` pipeline.

```bash
# 1. Clone the repo (inside a Colab cell) to access src/
!git clone https://github.com/<your-username>/zomato-delivery-time-prediction.git
import sys; sys.path.append("/content/zomato-delivery-time-prediction/src")

# 2. Authenticate and download the dataset (inside a Colab cell)
from google.colab import files
files.upload()  # kaggle.json from kaggle.com/settings -> API -> Create New Token
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d saurabhbadole/zomato-delivery-operations-analytics-dataset -p data/raw --unzip

# 3. Mount Drive for artifact persistence
from google.colab import drive
drive.mount('/content/drive')
```

Then run the three notebooks **in order**:

1. **`01_data_preparation.ipynb`** — audit, EDA, cleaning, feature engineering, leakage audit, train/val/test split → saves `prepared_data.parquet` + `feature_metadata.json` to Drive
2. **`02_modeling.ipynb`** — baselines, fair-hyperparameter model comparison, Optuna tuning, feature ablation → saves tuned model candidates + comparison tables to Drive
3. **`03_analysis_and_uncertainty.ipynb`** — SHAP, residual/error analysis, model selection, quantile regression + CQR calibration, `predict_delivery()`, temporal generalization, case studies → saves the final model + all analysis artifacts to Drive

**Dependencies** (installed inline per-notebook, no separate environment file — this is a Colab-native project):
```bash
pip install kaggle xgboost optuna shap --quiet
```
