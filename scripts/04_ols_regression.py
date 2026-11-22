"""OLS regression: lockdown stringency vs. change in DV homicide rate."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

DATA_DIR = "../outputs/data"
FIG_DIR = "../outputs/figures"
panel = pd.read_csv(f"{DATA_DIR}/analysis_panel.csv")

# ---------------------------------------------------------------------------
# Build country-level cross-section: baseline 2019, pandemic-period average
# (2020-2022), and the *variation* (delta) in DV homicide rate.
# ---------------------------------------------------------------------------
base19 = panel[panel.year == 2019].set_index("iso2")["dv_homicide_rate_f"]
pand_mean = panel[panel.period == "pandemic"].groupby("iso2")["dv_homicide_rate_f"].mean()
stringency = panel[panel.period == "pandemic"].groupby("iso2")["stringency_avg"].mean()
gov_resp = panel[panel.period == "pandemic"].groupby("iso2")["gov_response_avg"].mean()
contain = panel[panel.period == "pandemic"].groupby("iso2")["containment_health_avg"].mean()
econ = panel[panel.period == "pandemic"].groupby("iso2")["economic_support_avg"].mean()
stay_home = panel[panel.period == "pandemic"].groupby("iso2")["stay_home_avg"].mean()

cs = pd.concat([base19.rename("baseline_2019"), pand_mean.rename("pandemic_mean"),
                stringency.rename("stringency_avg"), gov_resp.rename("gov_response_avg"),
                contain.rename("containment_health_avg"), econ.rename("economic_support_avg"),
                stay_home.rename("stay_home_avg")], axis=1).dropna()
cs["delta_dv_rate"] = cs["pandemic_mean"] - cs["baseline_2019"]
cs["pct_change_dv_rate"] = 100 * cs["delta_dv_rate"] / cs["baseline_2019"].replace(0, np.nan)
cs.to_csv(f"{DATA_DIR}/country_cross_section.csv")
print("Cross-section n =", len(cs))
print(cs.describe())

# ---------------------------------------------------------------------------
# Model A: simple bivariate OLS  delta_dv_rate ~ stringency_avg
# ---------------------------------------------------------------------------
X = sm.add_constant(cs["stringency_avg"])
y = cs["delta_dv_rate"]
modelA = sm.OLS(y, X).fit()
print("\n===== Model A: delta_dv_rate ~ stringency_avg =====")
print(modelA.summary())

with open(f"{DATA_DIR}/ols_model_A_summary.txt", "w") as f:
    f.write(str(modelA.summary()))

# ---------------------------------------------------------------------------
# Model B: multiple OLS with additional OxCGRT policy indices
# ---------------------------------------------------------------------------
Xb = sm.add_constant(cs[["stringency_avg", "economic_support_avg", "stay_home_avg"]])
modelB = sm.OLS(y, Xb).fit()
print("\n===== Model B: delta_dv_rate ~ stringency + economic_support + stay_home =====")
print(modelB.summary())
with open(f"{DATA_DIR}/ols_model_B_summary.txt", "w") as f:
    f.write(str(modelB.summary()))

# ---------------------------------------------------------------------------
# Diagnostics figure: residuals vs fitted for Model A
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(modelA.fittedvalues, modelA.resid, alpha=0.7, color="#c0392b")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Model A Residuals vs Fitted")

sm.qqplot(modelA.resid, line="45", ax=axes[1])
axes[1].set_title("Model A Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_ols_diagnostics.png", dpi=200)
plt.close()

print("\nModel A R2:", modelA.rsquared, "Adj R2:", modelA.rsquared_adj)
print("Model B R2:", modelB.rsquared, "Adj R2:", modelB.rsquared_adj)
