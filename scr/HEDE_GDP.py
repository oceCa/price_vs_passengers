import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/merged_air_travel_data.csv")

df = df.rename(columns={'Real price': 'real_price'})
df['passengers'] = df['passengers'].replace(',', '', regex=True).astype(float)

df = df[(df['real_price'] > 0) & (df['passengers'] > 0)]
df['Year'] = df['Year'].astype(int)

# -----------------------------
# 2. annual panel city1 -> city2
# -----------------------------
annual = (
    df.groupby(['Year', 'city1', 'city2'], as_index=False)
      .agg(
          passengers=('passengers', 'sum'),
          real_price=('real_price', 'mean')
      )
)

annual = annual[annual['passengers'] > 0]
annual['route_id'] = annual['city1'] + " -> " + annual['city2']

# Price delayed by one year
annual = annual.sort_values(['route_id', 'Year'])
annual['price_lag1'] = annual.groupby('route_id')['real_price'].shift(1)
annual = annual.dropna(subset=['price_lag1'])

annual['ln_pax'] = np.log(annual['passengers'])
annual['ln_price_lag1'] = np.log(annual['price_lag1'])

# -----------------------------
# 3. Extract the original state (City, ST)
# -----------------------------
annual['origin_state'] = annual['city1'].str.extract(r',\s*([A-Z]{2})')[0]

# -----------------------------
# 4. Internal table: GDP per capita (2025)
# Source: BEA / Wikipedia (2025 values)
# -----------------------------
state_gdp_pc = {
    "CA": 104916, "TX": 86987, "NY": 117332, "FL": 73784, "IL": 90449, "PA": 78544,
    "OH": 78120, "GA": 78754, "WA": 108468, "NJ": 90272, "NC": 75876, "MA": 110561,
    "VA": 86747, "MI": 71083, "CO": 93026, "AZ": 73203, "TN": 75748, "MD": 87021,
    "IN": 76004, "MN": 86371, "WI": 75605, "MO": 72108, "CT": 100235, "SC": 63711,
    "OR": 77916, "LA": 71642, "AL": 61846, "UT": 86506, "KY": 64110, "OK": 64719,
    "NV": 80880, "IA": 79631, "KS": 79513, "AR": 60276, "NE": 93145, "DC": 263220,
    "MS": 53061, "NM": 66229, "ID": 63991, "NH": 85518, "HI": 80325, "WV": 60783,
    "DE": 98055, "ME": 69803, "RI": 74594, "MT": 66379, "ND": 95982, "SD": 80685,
    "AK": 95147, "WY": 90335, "VT": 70131,
}

state_gdp = pd.DataFrame(list(state_gdp_pc.items()), columns=['origin_state', 'gdp_pc'])

annual = annual.merge(state_gdp, on='origin_state', how='left')
annual = annual.dropna(subset=['gdp_pc'])

# -----------------------------
# 5. Create 6 wealth groups (quantiles)
# -----------------------------
annual['gdp_group'] = pd.qcut(annual['gdp_pc'], 6, labels=False)

# (Optional) Display the minimum/maximum GDP limits per group
gdp_ranges = (
    annual.groupby('gdp_group')['gdp_pc']
          .agg(['min', 'max', 'count'])
          .round(0).astype(int)
)
print("\nGDP per capita ranges by group (in $):")
print(gdp_ranges)

# -----------------------------
# 6. Estimation FE-OLS per groupe of GDP
#    ln_pax ~ ln_price_lag1 + FE(route) + FE(year)
# -----------------------------
elasticities = []
ses = []
groups = sorted(annual['gdp_group'].dropna().unique())

for g in groups:
    sub = annual[annual['gdp_group'] == g].copy()
    if sub.empty:
        elasticities.append(np.nan)
        ses.append(np.nan)
        continue

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
# 7. Graphs
# -----------------------------
plt.figure(figsize=(7,4))
plt.errorbar(groups, elasticities, yerr=ses, fmt='o-', capsize=4)
plt.xticks(groups, [f"G{int(g)+1}" for g in groups])
plt.axhline(0, linestyle='--', color='grey')
plt.ylabel("Elasticity (beta)")
plt.xlabel("GDP per capita group (poor → rich)")
plt.title("Price elasticity by origin-state income (FE-OLS, no IV)")
plt.tight_layout()
plt.show()