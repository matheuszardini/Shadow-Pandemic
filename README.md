# The Shadow Pandemic: Impact of COVID-19 Lockdowns on Domestic Violence in Europe

**Data Analytics | Python | Statistical Analysis | Machine Learning**

## Project Overview

The COVID-19 pandemic introduced significant social and economic challenges across Europe. Among the concerns raised during this period was the potential increase in domestic violence, often referred to as the "Shadow Pandemic".

This project investigates the relationship between COVID-19 lockdown measures and domestic violence-related homicide trends across European countries.

The analysis combines crime statistics and COVID-19 policy indicators to explore patterns, compare periods, conduct statistical hypothesis testing, and evaluate predictive modelling approaches.

## Research Objectives

- Examine domestic violence-related homicide trends across European countries.
- Explore changes in homicide rates before and during the COVID-19 pandemic.
- Investigate the relationship between lockdown stringency and changes in domestic violence-related homicide rates.
- Apply statistical methods to evaluate observed differences and relationships.
- Explore the predictive potential of machine learning models.

## Methodology

The project follows a reproducible data analytics pipeline:

1. **Data Cleaning and Integration** – Processing and combining Eurostat crime datasets and COVID-19 policy indicators.
2. **Exploratory Data Analysis (EDA)** – Identifying trends, distributions, country-level differences, and temporal patterns.
3. **Statistical Hypothesis Testing** – Applying Wilcoxon Signed-Rank and Mann–Whitney U tests.
4. **Regression Analysis** – Using Ordinary Least Squares (OLS) regression to investigate associations between lockdown stringency and changes in domestic violence-related homicide rates.
5. **Machine Learning** – Developing a Random Forest Regressor and evaluating generalisation using Leave-One-Country-Out cross-validation.

## Tools & Technologies

| Technology | Application |
|---|---|
| Python | Data analysis and modelling |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualisation |
| SciPy | Statistical hypothesis testing |
| Statsmodels | OLS regression |
| Scikit-learn | Machine learning and model evaluation |
| Jupyter Notebook | Reproducible analysis |

## Repository Structure

```text
Shadow-Pandemic/
│
├── notebooks/
│   └── Shadow_Pandemic_Analysis.ipynb
│
├── outputs/
│   ├── data/
│   └── figures/
│
├── scripts/
│   ├── 01_data_pipeline.py
│   ├── 02_eda.py
│   ├── 03_hypothesis_tests.py
│   ├── 04_ols_regression.py
│   └── 05_ml_model.py
│
└── README.md
```

## Reproducibility

The Jupyter Notebook contains the analysis workflow, while the Python scripts organise the main analytical stages.

To reproduce the analysis:

1. Clone this repository.
2. Install the required Python dependencies.
3. Open the notebook in Jupyter.
4. Execute the analysis cells in sequence.

## Data Sources

The analysis uses:

- **Eurostat** – Crime and criminal justice statistics.
- **Oxford COVID-19 Government Response Tracker (OxCGRT)** – COVID-19 government policy response indicators.

## Academic Context

Developed as part of the Higher Diploma in Data Analytics.

**Author:** Matheus Zardini

---

*This project is an academic data analytics investigation. Statistical associations should not be interpreted as evidence of causation.*
