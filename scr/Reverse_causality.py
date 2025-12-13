# Reverse_causality_price_to_pax.py
# Objectif : tester la "reverse causality" =
# est-ce que le prix moyen des billets à l'année n-1
# influence le nombre de passagers à l'année n ?
#
# - Partie 1 : routes ville -> ville à partir de merged_air_travel_data.csv
#              (données trimestrielles agrégées à l'année)
# - Partie 2 : routes état -> ville à partir de merged_air_travel_data.csv
#              (agrégées à l'année comme dans states_departures.csv, mais en RAM)
#
# Pour chaque panel, on construit :
#   route_id, Year, total_passengers_t, avg_price_t
# puis un lag de prix par route :
#   price_lag1 = avg_price_{t-1}
# et on estime :
#   ln_pax_t ~ ln_price_lag1
# (OLS simple avec SE clusterisées + panel FE si linearmodels est installé)

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============= CONFIG =============

MERGED_CSV = "data/merged_air_travel_data.csv"   # city1, city2, Year, quarter, passengers, Real price, ...

MIN_PAX = 0   # tu peux le passer à 500 ou 1000 si tu veux

# ✅ OUTPUTS (tout ira ici)
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_reverse_causality")
RES_DIR = os.path.join(OUT_DIR, "results_reverse_causality")


# ============= HELPERS =============

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

def info(msg: str) -> None:
    print(f"\n[INFO] {msg}")

def to_numeric(s):
    """Conversion robuste en numérique."""
    return pd.to_numeric(s, errors="coerce")


# ============= PANEL VILLES (merged_air_travel_data) =============

def prepare_city_panel(path: str) -> pd.DataFrame:
    """
    1) Charge merged_air_travel_data.csv
    2) Agrège à l'année : somme des passagers, moyenne simple des prix par (Year, city1, city2)
    3) Construit route_id = city1 -> city2 avec un lag de prix (année n-1)
    4) Ajoute ln_pax_t et ln_price_lag1
    """
    info(f"Chargement des données trimestrielles (villes) depuis {path} ...")
    df = pd.read_csv(path)

    df["passengers"] = to_numeric(df["passengers"])
    df["Real price"] = to_numeric(df["Real price"])

    df = df.dropna(subset=["Year", "city1", "city2", "passengers", "Real price"])

    info("Agrégation au niveau annuel (city1, city2, Year) ...")
    grp_cols = ["Year", "city1", "city2"]
    ann = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              total_passengers=("passengers", "sum"),
              avg_price=("Real price", "mean")
          )
    )

    ann = ann[(ann["total_passengers"] > 0) & (ann["avg_price"] > 0)]

    if MIN_PAX > 0:
        ann = ann[ann["total_passengers"] >= MIN_PAX]
        info(f"Filtre flux avec total_passengers >= {MIN_PAX} -> {len(ann)} lignes restantes")

    ann["route_id"] = ann["city1"].astype(str) + " -> " + ann["city2"].astype(str)

    ann = ann.sort_values(["route_id", "Year"])
    ann["price_lag1"] = ann.groupby("route_id")["avg_price"].shift(1)
    ann = ann.dropna(subset=["price_lag1"])

    ann["ln_pax"] = np.log(ann["total_passengers"])
    ann["ln_price_lag1"] = np.log(ann["price_lag1"])

    info(f"Panel villes prêt : {len(ann)} observations avec lag de prix dispo.")
    return ann


# ============= PANEL ÉTATS (agrégé depuis merged_air_travel_data) =============

def build_state_city_year_from_merged(path: str) -> pd.DataFrame:
    """
    Sort un DataFrame avec colonnes:
        Year, origin_state, city2, total_passengers, avg_price (pondéré pax)
    """
    info(f"Construction du panel états (state → city → year) depuis {path} ...")
    df = pd.read_csv(path)

    if "passengers" not in df.columns:
        raise KeyError("Column 'passengers' not found in merged_air_travel_data.csv")

    df["passengers"] = (
        df["passengers"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["passengers"] = to_numeric(df["passengers"])

    if "Year" not in df.columns:
        raise KeyError("Column 'Year' not found in merged_air_travel_data.csv")
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]

    if "Real price" not in df.columns:
        raise KeyError("Column 'Real price' not found in merged_air_travel_data.csv")
    df["Real price"] = to_numeric(df["Real price"])

    if "city1" not in df.columns:
        raise KeyError("Column 'city1' not found in merged_air_travel_data.csv")
    if "city2" not in df.columns:
        raise KeyError("Column 'city2' not found in merged_air_travel_data.csv")

    df["origin_state"] = df["city1"].str.extract(r',\s*([A-Z]{2})')[0]
    df = df.dropna(subset=["Year", "origin_state", "city2", "passengers", "Real price"])

    state_city_year = (
        df
        .groupby(["Year", "origin_state", "city2"])
        .apply(lambda g: pd.Series({
            "total_passengers": g["passengers"].sum(),
            "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
        }))
        .reset_index()
    )

    info(f"Panel agrégé états → villes prêt : {len(state_city_year)} lignes.")
    return state_city_year


def prepare_state_panel(path: str) -> pd.DataFrame:
    info(f"Préparation du panel états (origin_state -> city2, annuel) depuis {path} ...")
    df = build_state_city_year_from_merged(path)

    df["total_passengers"] = to_numeric(df["total_passengers"])
    df["avg_price"] = to_numeric(df["avg_price"])

    df = df.dropna(subset=["Year", "origin_state", "city2", "total_passengers", "avg_price"])
    df = df[(df["total_passengers"] > 0) & (df["avg_price"] > 0)]

    if MIN_PAX > 0:
        df = df[df["total_passengers"] >= MIN_PAX]
        info(f"Filtre flux avec total_passengers >= {MIN_PAX} -> {len(df)} lignes restantes")

    df["route_id"] = df["origin_state"].astype(str) + " -> " + df["city2"].astype(str)

    df = df.sort_values(["route_id", "Year"])
    df["price_lag1"] = df.groupby("route_id")["avg_price"].shift(1)
    df = df.dropna(subset=["price_lag1"])

    df["ln_pax"] = np.log(df["total_passengers"])
    df["ln_price_lag1"] = np.log(df["price_lag1"])

    info(f"Panel états prêt : {len(df)} observations avec lag de prix dispo.")
    return df


# ============= ESTIMATIONS =============

def run_ols_lag(df: pd.DataFrame, label: str, cluster_col: str | None = None):
    """
    OLS : ln_pax_t ~ ln_price_lag1
    - SE clusterisées si cluster_col valide
    - sinon HC1
    """
    info(f"OLS lag model pour {label} : ln_pax_t ~ ln_price_lag1")

    if cluster_col is not None and cluster_col in df.columns:
        info(f"SE clusterisées sur {cluster_col}")
        model = smf.ols("ln_pax ~ ln_price_lag1", data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster_col], "use_correction": True}
        )
    else:
        info("SE robustes HC1 (pas de cluster_col valide fourni)")
        model = smf.ols("ln_pax ~ ln_price_lag1", data=df).fit(cov_type="HC1")

    print(model.summary())
    return model


def run_fe_panel(df: pd.DataFrame, label: str):
    """
    Panel FE (route + année) via linearmodels si dispo.
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        info(f"linearmodels n'est pas installé -> on saute le panel FE pour {label}.")
        return None

    info(f"Panel FE (route & année) pour {label} ...")
    panel = df.set_index(["route_id", "Year"]).sort_index()

    Y = panel["ln_pax"]
    X = sm.add_constant(panel[["ln_price_lag1"]])

    fe_mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)

    print(fe_res.summary)
    return fe_res


# ============= MAIN =============

def main():
    ensure_dirs()

    # ----- Niveau villes -----
    if os.path.exists(MERGED_CSV):
        city_panel = prepare_city_panel(MERGED_CSV)

        # ✅ sauvegarde du panel préparé
        city_out = os.path.join(RES_DIR, "panel_cities_lag.csv")
        city_panel.to_csv(city_out, index=False)
        info(f"Saved prepared city panel: {city_out}")

        if len(city_panel) > 0:
            ols_city = run_ols_lag(city_panel, "villes (city1 -> city2, annuel)", cluster_col="route_id")
            fe_city = run_fe_panel(city_panel, "villes (city1 -> city2, annuel)")

            # ✅ sauvegarde des coefficients (petit CSV)
            rows = [{
                "panel": "cities",
                "model": "OLS_cluster_route",
                "beta_ln_price_lag1": float(ols_city.params.get("ln_price_lag1", np.nan)),
                "se": float(ols_city.bse.get("ln_price_lag1", np.nan)),
                "pvalue": float(ols_city.pvalues.get("ln_price_lag1", np.nan)),
                "N": int(ols_city.nobs),
            }]
            if fe_city is not None:
                rows.append({
                    "panel": "cities",
                    "model": "FE_route_year",
                    "beta_ln_price_lag1": float(getattr(fe_city, "params", {}).get("ln_price_lag1", np.nan)),
                    "se": float(getattr(fe_city, "std_errors", {}).get("ln_price_lag1", np.nan)),
                    "pvalue": float(getattr(fe_city, "pvalues", {}).get("ln_price_lag1", np.nan)),
                    "N": int(fe_city.nobs) if hasattr(fe_city, "nobs") else np.nan,
                })
            res_city = pd.DataFrame(rows)
            out_res_city = os.path.join(RES_DIR, "results_cities.csv")
            res_city.to_csv(out_res_city, index=False)
            info(f"Saved city results: {out_res_city}")
        else:
            info("Panel villes vide après préparation (pas de lag de prix disponible).")
    else:
        info(f"Fichier {MERGED_CSV} introuvable, on saute la partie villes.")

    # ----- Niveau états -----
    if os.path.exists(MERGED_CSV):
        state_panel = prepare_state_panel(MERGED_CSV)

        # ✅ sauvegarde du panel préparé
        state_out = os.path.join(RES_DIR, "panel_states_lag.csv")
        state_panel.to_csv(state_out, index=False)
        info(f"Saved prepared state panel: {state_out}")

        if len(state_panel) > 0:
            ols_state = run_ols_lag(state_panel, "états (origin_state -> city2, annuel)", cluster_col="route_id")
            fe_state = run_fe_panel(state_panel, "états (origin_state -> city2, annuel)")

            rows = [{
                "panel": "states",
                "model": "OLS_cluster_route",
                "beta_ln_price_lag1": float(ols_state.params.get("ln_price_lag1", np.nan)),
                "se": float(ols_state.bse.get("ln_price_lag1", np.nan)),
                "pvalue": float(ols_state.pvalues.get("ln_price_lag1", np.nan)),
                "N": int(ols_state.nobs),
            }]
            if fe_state is not None:
                rows.append({
                    "panel": "states",
                    "model": "FE_route_year",
                    "beta_ln_price_lag1": float(getattr(fe_state, "params", {}).get("ln_price_lag1", np.nan)),
                    "se": float(getattr(fe_state, "std_errors", {}).get("ln_price_lag1", np.nan)),
                    "pvalue": float(getattr(fe_state, "pvalues", {}).get("ln_price_lag1", np.nan)),
                    "N": int(fe_state.nobs) if hasattr(fe_state, "nobs") else np.nan,
                })
            res_state = pd.DataFrame(rows)
            out_res_state = os.path.join(RES_DIR, "results_states.csv")
            res_state.to_csv(out_res_state, index=False)
            info(f"Saved state results: {out_res_state}")
        else:
            info("Panel états vide après préparation (pas de lag de prix disponible).")
    else:
        info(f"Fichier {MERGED_CSV} introuvable, on saute la partie états.")


if __name__ == "__main__":
    main()

# NOTE:
# les clustered standard errors sont au niveau de la route (route_id),
# pas au niveau de l’année ni de l’État seul.
