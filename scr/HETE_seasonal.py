# HETE_SEASON_NOIV.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the quarterly panel
# -----------------------------
df = pd.read_csv("data/merged_air_travel_data.csv")

df = df.rename(columns={'Real price': 'real_price'})
df['passengers'] = df['passengers'].replace(',', '', regex=True).astype(float)

df = df[(df['real_price'] > 0) & (df['passengers'] > 0)]
df['Year'] = df['Year'].astype(int)
df['quarter'] = df['quarter'].astype(int)

df['route_id'] = df['city1'] + " -> " + df['city2']

# Price delayed by a quarter
df = df.sort_values(['route_id', 'Year', 'quarter'])
df['price_lag1'] = df.groupby('route_id')['real_price'].shift(1)
df = df.dropna(subset=['price_lag1'])

df['ln_pax'] = np.log(df['passengers'])
df['ln_price_lag1'] = np.log(df['price_lag1'])

# -----------------------------
# 2. Estimation FE (route + year) per quarter
# -----------------------------
elasticities = []
ses = []
quarters = [1, 2, 3, 4]

for q in quarters:
    sub = df[df['quarter'] == q].copy()
    if sub.empty:
        elasticities.append(np.nan)
        ses.append(np.nan)
        continue

    # OLS with FE(route + year), SE cluster route
    model = smf.ols(
        'ln_pax ~ C(route_id) + C(Year) + ln_price_lag1',
        data=sub
    ).fit(
        cov_type='cluster',
        cov_kwds={'groups': sub['route_id']}
    )

    beta = model.params['ln_price_lag1']
    se = model.bse['ln_price_lag1']

    elasticities.append(beta)
    ses.append(se)
 

# -----------------------------
# 3. Graphs
# -----------------------------
plt.figure(figsize=(6,4))
plt.bar(quarters, elasticities, yerr=ses, capsize=4)
plt.xticks(quarters, ["Q1","Q2","Q3","Q4"])
plt.axhline(0, linestyle='--', color='grey')
plt.ylabel("Elasticity (beta)")
plt.xlabel("Quarter")
plt.title("Price elasticity by season (FE-OLS, no IV)")
plt.tight_layout()
plt.show()