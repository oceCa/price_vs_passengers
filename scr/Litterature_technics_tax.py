import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

MERGED_CSV = "data/merged_air_travel_data.csv"

# 1. Charger la base brute (merged_air_travel_data.csv)
df = pd.read_csv(MERGED_CSV)

# 2. Nettoyer les colonnes clés
# Passagers (éventuelles virgules)
if "passengers" not in df.columns:
    raise KeyError("Column 'passengers' not found in merged_air_travel_data.csv")

df["passengers"] = (
    df["passengers"]
    .astype(str)
    .str.replace(",", "", regex=False)
)
df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")

# Année
if "Year" not in df.columns:
    raise KeyError("Column 'Year' not found in merged_air_travel_data.csv")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Prix réel
if "Real price" not in df.columns:
    raise KeyError("Column 'Real price' not found in merged_air_travel_data.csv")
df["Real price"] = pd.to_numeric(df["Real price"], errors="coerce")

# city1 / city2
if "city1" not in df.columns:
    raise KeyError("Column 'city1' not found in merged_air_travel_data.csv")
if "city2" not in df.columns:
    raise KeyError("Column 'city2' not found in merged_air_travel_data.csv")

# 3. Restreindre la fenêtre temporelle 1996–2025
df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]

# 4. Extraire l'État de départ depuis city1 (ex: "Miami, FL (Metropolitan Area)" -> "FL")
df["origin_state"] = df["city1"].str.extract(r',\s*([A-Z]{2})')[0]

# 5. Nettoyer les NA avant agrégation
df = df.dropna(subset=["Year", "origin_state", "city2", "passengers", "Real price"])

# 6. Recréer en mémoire l’équivalent de states_departures.csv :
#    agrégation Year + origin_state + city2
state_city_year = (
    df
    .groupby(["Year", "origin_state", "city2"])
    .apply(lambda g: pd.Series({
        "total_passengers": g["passengers"].sum(),
        # prix moyen pondéré par les passagers (à ce niveau state→city)
        "avg_price": (g["Real price"] * g["passengers"]).sum()
                      / g["passengers"].sum()
    }))
    .reset_index()
)

# 7. Ré-agréger au niveau state–year (comme dans ton code initial)
state_panel = (
    state_city_year
    .groupby(["origin_state", "Year"])
    .apply(lambda g: pd.Series({
        "total_passengers": g["total_passengers"].sum(),
        # moyenne pondérée des prix par les passagers de chaque state→city
        "avg_price": (g["avg_price"] * g["total_passengers"]).sum()
                      / g["total_passengers"].sum()
    }))
    .reset_index()
)

# 8. Logs
state_panel["log_passengers"] = np.log(state_panel["total_passengers"])
state_panel["log_price"] = np.log(state_panel["avg_price"])

# 9. Variables DiD
state_panel["CA"] = (state_panel["origin_state"] == "CA").astype(int)
state_panel["post"] = (state_panel["Year"] >= 2011).astype(int)  # année du cut à ajuster si besoin
state_panel["DID"] = state_panel["CA"] * state_panel["post"]

# 10. On enlève les NA
state_panel = state_panel.dropna(subset=["log_passengers", "DID"])

# 11. Régression DiD avec effets fixes état + année
model = smf.ols(
    formula="log_passengers ~ DID + C(origin_state) + C(Year)",
    data=state_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": state_panel["origin_state"]}  # SE clusterisés par état
)

print(model.summary())

# Notes perso (comme dans ton script original) :
# - Faire aussi une simple régression log_passengers ~ log_price pour avoir beta_1 et sa significativité.
# - Ajouter un lag sur le prix pour adresser la reverse causality (prix_t-1 -> passagers_t).
# - Discuter les sources d'endogénéité (météo, chocs locaux, etc.) qui biaisent beta_1.
# - Utiliser des fixed effects (année, état, voire route) pour capturer l’hétérogénéité inobservable.
