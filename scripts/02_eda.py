"""EDA & visualisations for the Shadow Pandemic project."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

DATA_DIR = "../outputs/data"
FIG_DIR = "../outputs/figures"

panel = pd.read_csv(f"{DATA_DIR}/analysis_panel.csv")

desc = panel.groupby("period")["dv_homicide_rate_f"].agg(["count", "mean", "median", "std", "min", "max"])
desc.to_csv(f"{DATA_DIR}/desc_stats_by_period.csv")
print("Descriptive stats by period:")
print(desc)

eu_desc = panel[panel.year.between(2015, 2024)].groupby("year")["dv_homicide_rate_f"].agg(["mean", "median", "std"])
eu_desc.to_csv(f"{DATA_DIR}/desc_stats_by_year.csv")
print("EU-wide average DV homicide rate by year:")
print(eu_desc)

fig, ax = plt.subplots(figsize=(10, 6))
yearly = panel.groupby("year")["dv_homicide_rate_f"].mean()
ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, color="#c0392b")
ax.axvspan(2020, 2022, color="grey", alpha=0.15, label="Pandemic period (2020-2022)")
ax.set_xlabel("Year")
ax.set_ylabel("Mean DV homicide rate\n(female victims, per 100,000 population)")
ax.set_title("Average Intimate-Partner/Family Homicide Rate of Women\nAcross European Countries, 2015-2024")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_timeseries.png", dpi=200)
plt.close()

piv = panel.pivot_table(index="country", columns="year", values="dv_homicide_rate_f")
piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(piv, cmap="Reds", linewidths=0.4, linecolor="white", vmax=2.0,
            cbar_kws={"label": "DV homicide rate per 100,000 (capped at 2.0 for readability)"}, ax=ax)
ax.set_title("Intimate-Partner/Family Homicide Rate of Women by Country and Year\n(micro-states e.g. Liechtenstein show extreme swings from very low case counts)")
ax.set_xlabel("Year")
ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_heatmap.png", dpi=200)
plt.close()

pand = panel[panel.period == "pandemic"].dropna(subset=["stringency_avg", "dv_homicide_rate_f"])
fig, ax = plt.subplots(figsize=(9, 7))
sns.regplot(data=pand, x="stringency_avg", y="dv_homicide_rate_f", ax=ax,
            scatter_kws={"alpha": 0.6, "s": 60}, line_kws={"color": "#c0392b"})
ax.set_xlabel("Mean Annual Government Stringency Index (0-100)")
ax.set_ylabel("DV homicide rate (female, per 100,000)")
ax.set_title("Lockdown Stringency vs. Domestic-Violence Homicide Rate\n(EU/EEA countries, 2020-2022)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_scatter_stringency.png", dpi=200)
plt.close()

box_df = panel[panel.period.isin(["pre-pandemic", "pandemic"])]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=box_df, x="period", y="dv_homicide_rate_f", hue="period",
            order=["pre-pandemic", "pandemic"], palette=["#2980b9", "#c0392b"],
            legend=False, ax=ax)
sns.stripplot(data=box_df, x="period", y="dv_homicide_rate_f", order=["pre-pandemic", "pandemic"],
              color="black", alpha=0.4, size=4, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("DV homicide rate (female, per 100,000)")
ax.set_title("Distribution of DV Homicide Rates:\nPre-pandemic (2018-19) vs Pandemic (2020-22)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_boxplot_period.png", dpi=200)
plt.close()

subset_countries = ["Germany", "France", "Italy", "Spain", "Poland", "Sweden",
                     "Austria", "Netherlands", "Greece", "Hungary", "Ireland", "Croatia"]
sub = panel[panel.country.isin(subset_countries)]
g = sns.relplot(data=sub, x="year", y="dv_homicide_rate_f", col="country", col_wrap=4,
                 kind="line", marker="o", height=2.6, color="#c0392b", facet_kws={"sharey": False})
for ax in g.axes.flat:
    ax.axvspan(2020, 2022, color="grey", alpha=0.15)
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Rate /100k")
g.fig.suptitle("DV Homicide Rate Trajectories by Country", y=1.03)
g.savefig(f"{FIG_DIR}/fig5_small_multiples.png", dpi=200, bbox_inches="tight")
plt.close()

corr_vars = ["dv_homicide_rate_f", "homicide_rate_f_total", "rape_rate_f",
             "sexual_violence_rate_f", "stringency_avg", "gov_response_avg",
             "containment_health_avg", "economic_support_avg", "stay_home_avg"]
corr = panel[panel.period == "pandemic"][corr_vars].corr()
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix of Violence Indicators and\nOxCGRT Policy Indices (2020-2022)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_correlation_heatmap.png", dpi=200)
plt.close()

print("All figures saved to", FIG_DIR)
