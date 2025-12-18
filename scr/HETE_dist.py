import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data (NO distance column needed)
# -----------------------------
df = pd.read_csv("data/merged_air_travel_data.csv")

# Keep real price
df = df.rename(columns={'Real price': 'real_price'})

# Passengers might be string with commas
df['passengers'] = df['passengers'].replace(',', '', regex=True).astype(float)

# -----------------------------
# 2. Basic cleaning
# -----------------------------
df = df[(df['real_price'] > 0) & (df['passengers'] > 0)]
df['Year'] = df['Year'].astype(int)

# -----------------------------
# 3. Create distance_km from nsmiles
#    nsmiles = non-stop miles (already a distance measure)
# -----------------------------
df = df[df['nsmiles'] > 0].copy()
df['distance_km'] = df['nsmiles'] * 1.60934

# -----------------------------
# 4. Annual panel city1 -> city2
# -----------------------------
annual = (
    df.groupby(['Year', 'city1', 'city2'], as_index=False)
      .agg(
          passengers=('passengers', 'sum'),
          real_price=('real_price', 'mean'),
          distance_km=('distance_km', 'mean')
      )
)

annual = annual[annual['passengers'] > 0]
annual['route_id'] = annual['city1'] + " -> " + annual['city2']

# Price lagged by one year
annual = annual.sort_values(['route_id', 'Year'])
annual['price_lag1'] = annual.groupby('route_id')['real_price'].shift(1)
annual = annual.dropna(subset=['price_lag1'])

annual['ln_pax'] = np.log(annual['passengers'])
annual['ln_price_lag1'] = np.log(annual['price_lag1'])

# -----------------------------
# 5. Passenger-balanced distance groups (same as your logic)
# -----------------------------
route_stats = (
    annual.groupby('route_id', as_index=False)
          .agg(
              distance_km=('distance_km', 'mean'),
              pax_total=('passengers', 'sum')
          )
).sort_values('distance_km')

total_pax = route_stats['pax_total'].sum()
route_stats['cum_pax_share'] = route_stats['pax_total'].cumsum() / total_pax

route_stats['dist_group'] = np.floor(6 * route_stats['cum_pax_share']).astype(int)
route_stats['dist_group'] = route_stats['dist_group'].clip(upper=5)

annual = annual.merge(route_stats[['route_id', 'dist_group']], on='route_id', how='left')

# Summary (km range + pax per group)
dist_summary = (
    route_stats.groupby('dist_group')
              .agg(
                  n_routes=('route_id', 'count'),
                  pax_total=('pax_total', 'sum'),
                  km_min=('distance_km', 'min'),
                  km_max=('distance_km', 'max')
              )
              .reset_index()
)
dist_summary[['km_min','km_max']] = dist_summary[['km_min','km_max']].round(0).astype(int)

print("\nPassenger-weighted distance groups summary:")
print(dist_summary)

# -----------------------------
# 6. FE-OLS Estimation by Group
# -----------------------------
elasticities, ses = [], []
groups = sorted(annual['dist_group'].dropna().unique())

for g in groups:
    sub = annual[annual['dist_group'] == g].copy()
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

    elasticities.append(model.params['ln_price_lag1'])
    ses.append(model.bse['ln_price_lag1'])

    
# -----------------------------
# 7. Graph
# -----------------------------
labels = [f"G{int(g)+1}" for g in groups]

plt.figure(figsize=(8,4))
plt.bar(range(len(groups)), elasticities, yerr=ses, capsize=4)
plt.xticks(range(len(groups)), labels)
plt.axhline(0, linestyle='--', color='grey')
plt.ylabel("Elasticity (beta)")
plt.xlabel("Passenger-weighted distance group (short → long)")
plt.title("Price elasticity by distance (FE-OLS, passenger-weighted bins)")
plt.tight_layout()
plt.savefig("HETE_DIST_WEIGHTED_NOIV.png", dpi=300)
plt.show()