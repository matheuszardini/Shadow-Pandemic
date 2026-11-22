"""
Random Forest Regressor for predicting country-year DV homicide rate during
the pandemic period, compared against a linear (OLS-equivalent) baseline
using Leave-One-Country-Out cross-validation (to avoid within-country leakage
across the repeated 2020-2022 observations).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

DATA_DIR = "../outputs/data"
FIG_DIR = "../outputs/figures"
panel = pd.read_csv(f"{DATA_DIR}/analysis_panel.csv")

# ---------------------------------------------------------------------------
# Build modelling dataset at country-year granularity (pandemic years only)
# ---------------------------------------------------------------------------
baseline = panel[panel.period == "pre-pandemic"].groupby("iso2")["dv_homicide_rate_f"].mean().rename("baseline_rate")
ml_df = panel[panel.period == "pandemic"].merge(baseline, on="iso2", how="left")
feature_cols = ["stringency_avg", "gov_response_avg", "containment_health_avg",
                 "economic_support_avg", "stay_home_avg", "year", "baseline_rate"]
ml_df = ml_df.dropna(subset=feature_cols + ["dv_homicide_rate_f"])
print("ML dataset shape:", ml_df.shape, " | countries:", ml_df["iso2"].nunique())

X = ml_df[feature_cols].values
y = ml_df["dv_homicide_rate_f"].values
groups = ml_df["iso2"].values

logo = LeaveOneGroupOut()

# ---------------------------------------------------------------------------
# Linear baseline (OLS-equivalent) under LOGO-CV
# ---------------------------------------------------------------------------
lin = LinearRegression()
lin_pred = cross_val_predict(lin, X, y, cv=logo, groups=groups)
lin_r2 = r2_score(y, lin_pred)
lin_mae = mean_absolute_error(y, lin_pred)
lin_rmse = mean_squared_error(y, lin_pred) ** 0.5

# ---------------------------------------------------------------------------
# Random Forest Regressor under LOGO-CV
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3,
                            random_state=42)
rf_pred = cross_val_predict(rf, X, y, cv=logo, groups=groups)
rf_r2 = r2_score(y, rf_pred)
rf_mae = mean_absolute_error(y, rf_pred)
rf_rmse = mean_squared_error(y, rf_pred) ** 0.5

# Naive baseline: always predict each country's own pre-pandemic mean
naive_pred = ml_df["baseline_rate"].values
naive_r2 = r2_score(y, naive_pred)
naive_mae = mean_absolute_error(y, naive_pred)
naive_rmse = mean_squared_error(y, naive_pred) ** 0.5

print("\n=== Leave-One-Country-Out Cross-Validation Results ===")
for name, r2, mae, rmse in [
    ("Naive (predict own pre-pandemic baseline)", naive_r2, naive_mae, naive_rmse),
    ("Linear Regression (OLS-equivalent)", lin_r2, lin_mae, lin_rmse),
    ("Random Forest Regressor", rf_r2, rf_mae, rf_rmse),
]:
    print(f"{name:45s} R2={r2:7.4f}  MAE={mae:7.4f}  RMSE={rmse:7.4f}")

results_df = pd.DataFrame({
    "model": ["Naive baseline", "Linear Regression", "Random Forest"],
    "R2": [naive_r2, lin_r2, rf_r2],
    "MAE": [naive_mae, lin_mae, rf_mae],
    "RMSE": [naive_rmse, lin_rmse, rf_rmse],
})
results_df.to_csv(f"{DATA_DIR}/ml_model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Fit RF on full data (for feature importance reporting only -- not for the
# performance numbers above, which come from the unbiased LOGO-CV)
# ---------------------------------------------------------------------------
rf_full = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                 random_state=42)
rf_full.fit(X, y)
importances = pd.Series(rf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest feature importances (fit on full pandemic-period data):")
print(importances)
importances.to_csv(f"{DATA_DIR}/rf_feature_importances.csv")

fig, ax = plt.subplots(figsize=(9, 6))
importances.sort_values().plot(kind="barh", color="#c0392b", ax=ax)
ax.set_xlabel("Feature importance (Gini importance)")
ax.set_title("Random Forest Feature Importances\n(Predicting DV Homicide Rate, 2020-2022)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_rf_feature_importance.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Predicted vs actual plot (LOGO-CV predictions) for both models
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, pred, title, r2 in [(axes[0], lin_pred, "Linear Regression", lin_r2),
                              (axes[1], rf_pred, "Random Forest", rf_r2)]:
    ax.scatter(y, pred, alpha=0.7, color="#2980b9")
    lims = [min(y.min(), pred.min()) - 0.1, max(y.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Actual DV homicide rate")
    ax.set_ylabel("Predicted (LOGO-CV)")
    ax.set_title(f"{title}\n(LOGO-CV R² = {r2:.3f})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_predicted_vs_actual.png", dpi=200)
plt.close()

print("\nDone.")
