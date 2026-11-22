"""
The Shadow Pandemic - Data Cleaning & Integration Pipeline
Loads Eurostat crim_hom_vrel, crim_hom_soff and OxCGRT datasets, filters to
domestic-violence-relevant categories, aggregates OxCGRT to annual country
level, and merges everything into a single analysis-ready panel dataset.
"""
import pandas as pd
import numpy as np

DATA_DIR = "../data"
OUT_DIR = "../outputs/data"

ISO2_TO_ISO3 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR", "CH": "CHE",
    "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IS": "ISL", "IT": "ITA", "LI": "LIE", "LT": "LTU", "LU": "LUX", "LV": "LVA",
    "ME": "MNE", "MK": "MKD", "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL",
    "PT": "PRT", "RO": "ROU", "RS": "SRB", "SE": "SWE", "SI": "SVN", "SK": "SVK",
    "TR": "TUR", "XK": "XKX",
}
EXCLUDE_GEO = {"EU_V", "UKC-L", "UKM", "UKN"}


def load_vrel():
    df = pd.read_csv(f"{DATA_DIR}/crim_hom_vrel_linear_2_0.csv")
    df = df[~df["geo"].isin(EXCLUDE_GEO)]
    sub = df[(df["pers_cat"] == "IPTN_FAM") & (df["sex"] == "F") & (df["unit"] == "P_HTHAB")]
    sub = sub[["geo", "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"geo": "iso2", "TIME_PERIOD": "year", "OBS_VALUE": "dv_homicide_rate_f"}
    )
    return sub.dropna(subset=["dv_homicide_rate_f"])


def load_soff():
    df = pd.read_csv(f"{DATA_DIR}/crim_hom_soff_linear_2_0.csv")
    df = df[~df["geo"].isin(EXCLUDE_GEO)]
    sub = df[(df["leg_stat"] == "PER_VICT") & (df["sex"] == "F") & (df["unit"] == "P_HTHAB")]
    piv = sub.pivot_table(index=["geo", "TIME_PERIOD"], columns="iccs", values="OBS_VALUE").reset_index()
    piv = piv.rename(columns={
        "geo": "iso2", "TIME_PERIOD": "year",
        "ICCS0101": "homicide_rate_f_total",
        "ICCS03011": "rape_rate_f",
        "ICCS03012": "sexual_violence_rate_f",
    })
    return piv


def load_oxcgrt_annual():
    cols = ["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average",
            "GovernmentResponseIndex_Average", "ContainmentHealthIndex_Average",
            "EconomicSupportIndex", "C6M_Stay at home requirements"]
    df = pd.read_csv(f"{DATA_DIR}/OxCGRT_nat_latest.csv", usecols=cols)
    df = df[df["Jurisdiction"] == "NAT_TOTAL"].copy()
    df["year"] = df["Date"].astype(str).str[:4].astype(int)
    agg = df.groupby(["CountryCode", "year"]).agg(
        stringency_avg=("StringencyIndex_Average", "mean"),
        gov_response_avg=("GovernmentResponseIndex_Average", "mean"),
        containment_health_avg=("ContainmentHealthIndex_Average", "mean"),
        economic_support_avg=("EconomicSupportIndex", "mean"),
        stay_home_avg=("C6M_Stay at home requirements", "mean"),
    ).reset_index()
    return agg.rename(columns={"CountryCode": "iso3"})


def build_panel():
    vrel = load_vrel()
    soff = load_soff()
    ox = load_oxcgrt_annual()

    panel = pd.merge(vrel, soff, on=["iso2", "year"], how="left")
    panel["iso3"] = panel["iso2"].map(ISO2_TO_ISO3)
    panel = pd.merge(panel, ox, on=["iso3", "year"], how="left")

    pre_years = panel["year"] < 2020
    panel.loc[pre_years, ["stringency_avg", "gov_response_avg",
                           "containment_health_avg", "economic_support_avg",
                           "stay_home_avg"]] = panel.loc[pre_years, ["stringency_avg",
                           "gov_response_avg", "containment_health_avg",
                           "economic_support_avg", "stay_home_avg"]].fillna(0)

    country_names = {
        "AL": "Albania", "AT": "Austria", "BA": "Bosnia and Herzegovina", "BE": "Belgium",
        "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany",
        "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
        "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IS": "Iceland",
        "IT": "Italy", "LI": "Liechtenstein", "LT": "Lithuania", "LU": "Luxembourg",
        "LV": "Latvia", "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta",
        "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
        "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "TR": "Turkey",
    }
    panel["country"] = panel["iso2"].map(country_names)

    def period(y):
        if 2018 <= y <= 2019:
            return "pre-pandemic"
        if 2020 <= y <= 2022:
            return "pandemic"
        return "other"
    panel["period"] = panel["year"].apply(period)

    panel = panel.sort_values(["iso2", "year"]).reset_index(drop=True)
    return panel


panel = build_panel()
panel.to_csv(f"{OUT_DIR}/analysis_panel.csv", index=False)
print("Panel shape:", panel.shape)
print("Countries:", panel['iso2'].nunique())
print("Year range:", panel['year'].min(), "-", panel['year'].max())
panel.head(15)
