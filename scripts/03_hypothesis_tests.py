"""Hypothesis testing: pre-pandemic (2018-19) vs pandemic (2020-22) DV rates."""
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = "../outputs/data"
panel = pd.read_csv(f"{DATA_DIR}/analysis_panel.csv")

def paired_country_table(value_col):
    pre = panel[panel.period == "pre-pandemic"].groupby("iso2")[value_col].mean()
    pan = panel[panel.period == "pandemic"].groupby("iso2")[value_col].mean()
    both = pd.concat([pre.rename("pre"), pan.rename("pandemic")], axis=1).dropna()
    return both

results = {}

# ---------------------------------------------------------------------------
# Test 1: Wilcoxon signed-rank test on DV homicide rate (intimate partner/family)
# ---------------------------------------------------------------------------
t1 = paired_country_table("dv_homicide_rate_f")
w1 = stats.wilcoxon(t1["pandemic"], t1["pre"], alternative="greater")
results["dv_homicide_wilcoxon"] = {
    "n_countries": len(t1), "pre_mean": t1["pre"].mean(), "pandemic_mean": t1["pandemic"].mean(),
    "pre_median": t1["pre"].median(), "pandemic_median": t1["pandemic"].median(),
    "statistic": w1.statistic, "p_value": w1.pvalue,
    "n_increased": (t1["pandemic"] > t1["pre"]).sum(),
}

# ---------------------------------------------------------------------------
# Test 2: Wilcoxon signed-rank test on sexual violence rate (secondary indicator)
# ---------------------------------------------------------------------------
t2 = paired_country_table("sexual_violence_rate_f")
w2 = stats.wilcoxon(t2["pandemic"], t2["pre"], alternative="greater")
results["sexual_violence_wilcoxon"] = {
    "n_countries": len(t2), "pre_mean": t2["pre"].mean(), "pandemic_mean": t2["pandemic"].mean(),
    "pre_median": t2["pre"].median(), "pandemic_median": t2["pandemic"].median(),
    "statistic": w2.statistic, "p_value": w2.pvalue,
    "n_increased": (t2["pandemic"] > t2["pre"]).sum(),
}

# ---------------------------------------------------------------------------
# Test 3 (robustness): Mann-Whitney U (independent samples) on total homicide
# rate, all country-year observations rather than country means
# ---------------------------------------------------------------------------
g1 = panel[panel.period == "pre-pandemic"]["dv_homicide_rate_f"].dropna()
g2 = panel[panel.period == "pandemic"]["dv_homicide_rate_f"].dropna()
mw = stats.mannwhitneyu(g2, g1, alternative="greater")
results["dv_homicide_mannwhitney_obslevel"] = {
    "n_pre_obs": len(g1), "n_pandemic_obs": len(g2),
    "statistic": mw.statistic, "p_value": mw.pvalue,
}

# ---------------------------------------------------------------------------
# Effect size: matched-pairs rank-biserial correlation for Wilcoxon test 1
# ---------------------------------------------------------------------------
diffs = t1["pandemic"] - t1["pre"]
n = len(diffs)
r_rb = w1.statistic / (n * (n + 1) / 2) * 2 - 1  # rank-biserial approx from W statistic
results["dv_homicide_wilcoxon"]["effect_size_rank_biserial_approx"] = r_rb

for k, v in results.items():
    print(f"\n=== {k} ===")
    for kk, vv in v.items():
        print(f"  {kk}: {vv}")

pd.DataFrame(results).T.to_csv(f"{DATA_DIR}/hypothesis_test_results.csv")
t1.to_csv(f"{DATA_DIR}/paired_dv_homicide_pre_vs_pandemic.csv")
