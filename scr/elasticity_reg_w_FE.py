#!/usr/bin/env python3
# Elasticity_cities_states_reg_w_FE_from_merged.py
# -----------------------------------------------------------
# Estimates price-demand elasticity from merged_air_travel_data.csv:
#   1) City-level (city1 -> city2 -> year):
#       - OLS log-log (route-clustered SE)
#       - MLR log-log with controls (if available)
#       - HDFE with 6 FE (o, d, t, od, ot, dt) if linearmodels available
#       - Two-way FE Panel (route, year)
#       - Log-log plot saved in outputs/
#
#   2) State-level reconstructed (origin_state -> city2 -> year):
#       - OLS log-log
#       - MLR log-log with controls (if available)
#       - HDFE 6 FE (o, d, t, od, ot, dt) if linearmodels available
#       - Two-way FE Panel (route, year)
#       - Log-log plot saved in outputs/
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant

# -----------------------
# Paths / I/O
# -----------------------
MERGED_CSV = "data/merged_air_travel_data.csv"

# All figures will be written here
OUTPUT_DIR = "outputs"


def _ensure_out_dir():
    """Create outputs directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------
# Helpers robustes communs
# -----------------------
def first_exact(df, candidates, required=False):
    """Trouve la 1re colonne parmi des noms EXACTS."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Colonnes candidates absentes: {candidates}")
    return None


def find_by_keywords(df, any_tokens, required=False):
    """Trouve la 1re colonne dont le nom (lowercase) contient au moins un des tokens."""
    any_tokens = [t.lower() for t in any_tokens]
    for c in df.columns:
        lc = c.lower()
        if any(t in lc for t in any_tokens):
            return c
    if required:
        raise KeyError(f"Aucune colonne ne contient l'un des tokens: {any_tokens}")
    return None


def to_numeric_safe(s):
    if s is None:
        return None
    return pd.to_numeric(s, errors="coerce")


def log_safe(series):
    return np.log(series.replace(0, np.nan))


def clean_positive(df, cols):
    for c in cols:
        df = df[df[c].notna() & (df[c] > 0)]
    return df


# -----------------------
# Load & nettoyage de merged_air_travel_data.csv
# -----------------------
def load_and_clean_merged(path=MERGED_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable.")
    print(f"\n[INFO] Chargement de {path} ...")
    df = pd.read_csv(path)

    # Passagers
    if "passengers" not in df.columns:
        raise KeyError("Colonne 'passengers' absente du CSV.")
    df["passengers"] = (
        df["passengers"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["passengers"] = to_numeric_safe(df["passengers"])

    # Prix réel
    if "Real price" not in df.columns:
        raise KeyError("Colonne 'Real price' absente du CSV.")
    df["Real price"] = to_numeric_safe(df["Real price"])

    # Année + villes
    for col in ["Year", "city1", "city2"]:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' absente du CSV.")
    df["Year"] = to_numeric_safe(df["Year"])

    # Nettoyage de base
    df = df.dropna(subset=["passengers", "Real price", "Year", "city1", "city2"])
    df = df[(df["passengers"] > 0) & (df["Real price"] > 0)]

    # Fenêtre temporelle
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]

    print(f"[INFO] Données nettoyées: {len(df)} lignes.")
    return df


# -----------------------
# Construction panel états → villes depuis merged (en mémoire)
# -----------------------
def build_state_city_year_from_merged(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Sort un DataFrame avec colonnes:
        Year, origin_state, city2, total_passengers, avg_price
    """
    print("\n[INFO] Construction du panel états (state → city → year) ...")
    df = df_merged.copy()

    # Extraire l'État de départ depuis city1 (ex: 'Los Angeles, CA')
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

    print(f"[INFO] Panel agrégé états → villes prêt : {len(state_city_year)} lignes.")
    return state_city_year


# ============================================================
# BLOC 1 : CITY-LEVEL (city1 -> city2 -> Year) avec HDFE & FE
# ============================================================
def run_cities_hdfe(df_merged: pd.DataFrame):
    _ensure_out_dir()

    print("\n" + "="*72)
    print("[CITIES] Elasticity_cities_reg_w_FE_from_merged")
    print("="*72)

    raw = df_merged.copy()

    # Détection colonnes
    col_pax_raw   = first_exact(raw, ["passengers"])
    col_price_raw = first_exact(raw, ["Real price", "real_price", "price", "fare"])
    col_city1     = first_exact(raw, ["city1"]) or find_by_keywords(raw, ["city1","orig","from","source"], required=True)
    col_city2     = first_exact(raw, ["city2"]) or find_by_keywords(raw, ["city2","dest","to","target"], required=True)
    col_year      = first_exact(raw, ["Year","year"]) or find_by_keywords(raw, ["year","yr"], required=True)

    # Nettoyage (re-sécurisé, même si déjà fait)
    raw[col_pax_raw]   = raw[col_pax_raw].astype(str).str.replace(",", "", regex=False)
    raw[col_pax_raw]   = to_numeric_safe(raw[col_pax_raw])
    raw[col_price_raw] = to_numeric_safe(raw[col_price_raw])

    raw = raw.dropna(subset=[col_pax_raw, col_price_raw, col_city1, col_city2, col_year])
    raw = raw[(raw[col_pax_raw] > 0) & (raw[col_price_raw] > 0)]
    raw = raw[(raw[col_year] >= 1996) & (raw[col_year] <= 2025)]

    # Agrégation route-année (passenger-weighted average price)
    grouped = (
        raw.groupby([col_year, col_city1, col_city2], as_index=False, sort=False)
           .apply(
               lambda g: pd.Series({
                   "total_passengers": g[col_pax_raw].sum(),
                   "avg_price": (g[col_price_raw] * g[col_pax_raw]).sum() / g[col_pax_raw].sum()
               }),
               include_groups=False
           )
           .reset_index(drop=True)
    )

    grouped = grouped[grouped["total_passengers"] >= 5000].copy()
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_price", "total_passengers"])

    if grouped.empty:
        raise ValueError("[CITIES] Aucune route ≥ 5000 passagers après nettoyage.")

    # Colonnes de base
    col_pax   = "total_passengers"
    col_price = "avg_price"

    # Contrôles potentiels
    col_dist     = first_exact(grouped, ["distance","Distance","dist_miles","dist_km"], required=False)
    col_carriers = first_exact(grouped, ["carriers","n_carriers","num_carriers"], required=False)
    col_month    = first_exact(grouped, ["month","Month"], required=False)
    col_pop_o    = first_exact(grouped, ["pop_origin","origin_pop","pop_o"], required=False)
    col_pop_d    = first_exact(grouped, ["pop_dest","destination_pop","pop_d"], required=False)

    for c in [col_dist, col_carriers, col_pop_o, col_pop_d]:
        if c:
            grouped[c] = to_numeric_safe(grouped[c])

    # Logs & IDs
    grouped["ln_pax"]   = np.log(grouped[col_pax])
    grouped["ln_price"] = np.log(grouped[col_price])
    if col_dist:  grouped["ln_dist"]   = log_safe(grouped[col_dist])
    if col_pop_o: grouped["ln_pop_o"]  = log_safe(grouped[col_pop_o])
    if col_pop_d: grouped["ln_pop_d"]  = log_safe(grouped[col_pop_d])

    grouped["origin"]      = grouped[col_city1].astype(str)
    grouped["destination"] = grouped[col_city2].astype(str)
    grouped["route_id"]    = grouped["origin"] + " -> " + grouped["destination"]
    year_col = col_year
    have_year = year_col is not None
    if have_year:
        year_str = grouped[year_col].astype(int).astype(str)
        grouped["origin_year"] = grouped["origin"]      + "_" + year_str
        grouped["dest_year"]   = grouped["destination"] + "_" + year_str

    # Cluster variable
    cluster_var = grouped["route_id"]
    print(f"\n[INFO][CITIES] Using clustered standard errors (groups = route_id)")

    # 1) OLS log–log simple
    X1 = add_constant(grouped[["ln_price"]])
    ols1 = sm.OLS(grouped["ln_pax"], X1, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )
    print("\n=== [CITIES] OLS log–log (cross-section) ===")
    print(ols1.summary())

    # 2) MLR avec contrôles
    controls = []
    if col_dist:     controls.append("ln_dist")
    if col_carriers: controls.append(col_carriers)
    if col_pop_o:    controls.append("ln_pop_o")
    if col_pop_d:    controls.append("ln_pop_d")

    X_cols = ["ln_price"] + controls
    if col_month and col_month in grouped.columns:
        d_m = pd.get_dummies(grouped[col_month].astype("Int64"), prefix="m", drop_first=True)
        X2 = pd.concat([grouped[X_cols], d_m], axis=1)
        month_dummies = list(d_m.columns)
    else:
        X2 = grouped[X_cols].copy()
        month_dummies = []

    X2 = add_constant(X2)
    mlr = sm.OLS(grouped["ln_pax"], X2, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )
    print("\n=== [CITIES] MLR log–log (with observed controls if any) ===")
    print(f"Controls used: {controls + month_dummies}")
    print(mlr.summary())

    # 3) HDFE 6 FE (o, d, t, od, ot, dt)
    have_hdfe = ("origin" in grouped.columns) and ("destination" in grouped.columns) and have_year
    try:
        if have_hdfe:
            from linearmodels.iv import AbsorbingLS
            print("\n=== [CITIES] HDFE log–log (6 FE: o, d, t, od, ot, dt) ===")
            y = grouped["ln_pax"]
            X_hdfe = grouped[["ln_price"]]
            absorb = pd.DataFrame({
                "o":  grouped["origin"].astype("category"),
                "d":  grouped["destination"].astype("category"),
                "t":  grouped[year_col].astype("category"),
                "od": grouped["route_id"].astype("category"),
                "ot": grouped["origin_year"].astype("category"),
                "dt": grouped["dest_year"].astype("category"),
            })
            hdfe_mod = AbsorbingLS(y, X_hdfe, absorb=absorb)
            hdfe_res = hdfe_mod.fit(cov_type="robust")
            print(hdfe_res.summary)
        else:
            print("\n[CITIES][HDFE] skipped: need origin, destination and year to build 6 FE.")
    except Exception as e:
        print("\n[CITIES][HDFE] AbsorbingLS unavailable or other error:")
        print(e)
        print("→ Install:  pip install linearmodels")

    # 4) Panel FE (route & année)
    try:
        from linearmodels.panel import PanelOLS
        panel = grouped.set_index(["route_id", year_col]).sort_index()
        fe_X_cols = ["ln_price"]
        if col_dist:     fe_X_cols.append("ln_dist")
        if col_carriers: fe_X_cols.append(col_carriers)
        if col_pop_o:    fe_X_cols.append("ln_pop_o")
        if col_pop_d:    fe_X_cols.append("ln_pop_d")

        Y = panel["ln_pax"]
        Xfe = add_constant(panel[fe_X_cols])
        fe_mod = PanelOLS(Y, Xfe, entity_effects=True, time_effects=True)
        fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)
        print("\n=== [CITIES] Two-way FE: route & year — clustered by route ===")
        print(f"Variables in X: {fe_X_cols}")
        print(fe_res.summary)
    except Exception as e:
        print("\n[CITIES][Panel FE] linearmodels indisponible ou autre erreur :")
        print(e)
        print("→ Fallback: statsmodels with route/year dummies + clustered SE.")
        fe_formula = f"ln_pax ~ ln_price + C(route_id) + C({year_col})"
        fe_res_fallback = smf.ols(fe_formula, data=grouped).fit(
            cov_type="cluster", cov_kwds={"groups": grouped["route_id"]}
        )
        print("\n=== [CITIES] FE fallback (statsmodels) — clustered by route ===")
        print(fe_res_fallback.summary())

    # 5) Graphique log–log (SAVE TO outputs/)
    x = grouped[col_price].to_numpy()
    y = grouped[col_pax].to_numpy()
    if x.size >= 2 and y.size >= 2:
        logx = np.log(x)
        logy = np.log(y)
        a, b = np.polyfit(logx, logy, 1)
        x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_line = np.exp(b) * x_line**a

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=8, alpha=0.5, label="City → city (≥ 5000 passengers/year)")
        plt.plot(x_line, y_line, linewidth=2, label=f"Log-log trend (slope ≈ {a:.3f})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Average ticket price (USD, log scale)")
        plt.ylabel("Total passengers on route (per year, log scale)")
        plt.title("City-level routes: passengers vs. price (log-log, 1996–2025)")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(OUTPUT_DIR, "routes_cities_log_price_FE.png")
        plt.savefig(outpath, dpi=300)
        plt.show()
        print(f"[CITIES][PLOT] Saved: {outpath}")
    else:
        print("\n[CITIES][Plot] Not enough observations to draw the log–log trend.")


# ============================================================
# BLOC 2 : STATE-LEVEL (origin_state -> city2 -> Year) HDFE & FE
# ============================================================
def run_states_hdfe(state_df: pd.DataFrame):
    _ensure_out_dir()

    print("\n" + "="*72)
    print("[STATES] Elasticity_states_reg_w_FE_from_merged")
    print("="*72)

    df = state_df.copy()

    # Colonnes de base
    col_pax   = first_exact(df, ["total_passengers", "passengers"])
    if col_pax is None:
        col_pax = find_by_keywords(df, ["passenger", "pax"], required=True)

    col_price = first_exact(df, ["avg_price", "Real price", "real_price"])
    if col_price is None:
        col_price = find_by_keywords(df, ["price", "fare", "ticket"], required=True)

    # Année
    col_year = first_exact(df, ["year", "Year"])
    if col_year is None:
        col_year = find_by_keywords(df, ["year", "yr"], required=False)

    # Origine (state)
    col_o = first_exact(df, ["origin_state", "origin", "state_from", "o_state"])
    if col_o is None:
        col_o = find_by_keywords(df, ["orig", "state", "from"], required=False)

    # Destination (city)
    col_d = first_exact(df, ["dest_city", "destination_city", "city_to", "d_city"])
    if col_d is None:
        col_d = first_exact(df, ["city2", "city"], required=False)
    if col_d is None:
        col_d = find_by_keywords(df, ["dest", "to", "city"], required=False)

    # Contrôles potentiels
    col_dist     = first_exact(df, ["distance", "Distance", "dist_miles", "dist_km"], required=False)
    if col_dist is None:
        col_dist = find_by_keywords(df, ["dist"], required=False)
    col_carriers = first_exact(df, ["carriers", "n_carriers", "num_carriers"], required=False)
    if col_carriers is None:
        col_carriers = find_by_keywords(df, ["carrier", "airline", "comp"], required=False)
    col_month    = first_exact(df, ["month", "Month"], required=False)
    if col_month is None:
        col_month = find_by_keywords(df, ["month", "mo"], required=False)
    col_pop_o    = first_exact(df, ["pop_origin", "origin_pop", "pop_o"], required=False)
    if col_pop_o is None:
        col_pop_o = find_by_keywords(df, ["pop", "income", "market"], required=False)
    col_pop_d    = first_exact(df, ["pop_dest", "destination_pop", "pop_d"], required=False)
    if col_pop_d is None and col_d is not None:
        col_pop_d = find_by_keywords(df, ["pop", "income", "market"], required=False)

    # Typage
    df[col_pax]   = to_numeric_safe(df[col_pax])
    df[col_price] = to_numeric_safe(df[col_price])
    if col_dist:     df[col_dist]     = to_numeric_safe(df[col_dist])
    if col_carriers: df[col_carriers] = to_numeric_safe(df[col_carriers])
    if col_pop_o:    df[col_pop_o]    = to_numeric_safe(df[col_pop_o])
    if col_pop_d:    df[col_pop_d]    = to_numeric_safe(df[col_pop_d])

    # Nettoyage minimal
    base_needed = [col_pax, col_price]
    if col_year:
        base_needed.append(col_year)
    df = df.dropna(subset=[c for c in base_needed if c is not None])
    df = df[df[col_pax] >= 5000]
    df = clean_positive(df, [col_pax, col_price])

    # Logs
    df["ln_pax"]   = np.log(df[col_pax])
    df["ln_price"] = np.log(df[col_price])
    if col_dist:  df["ln_dist"]   = log_safe(df[col_dist])
    if col_pop_o: df["ln_pop_o"]  = log_safe(df[col_pop_o])
    if col_pop_d: df["ln_pop_d"]  = log_safe(df[col_pop_d])

    # Route ID + origin/destination standardisées
    have_route = (col_o is not None) and (col_d is not None)
    if have_route:
        df["origin"]      = df[col_o].astype(str)
        df["destination"] = df[col_d].astype(str)
        df["route_id"]    = df["origin"] + " -> " + df["destination"]

    have_year = col_year is not None
    if have_route and have_year:
        year_str = df[col_year].astype(int).astype(str)
        df["origin_year"] = df["origin"]      + "_" + year_str
        df["dest_year"]   = df["destination"] + "_" + year_str

    # Cluster var
    cluster_var = None
    cluster_name = None
    if "route_id" in df.columns:
        cluster_var = df["route_id"]
        cluster_name = "route_id"
    elif col_o is not None:
        cluster_var = df[col_o].astype(str)
        cluster_name = col_o

    if cluster_var is not None:
        print(f"\n[INFO][STATES] Using clustered standard errors (groups = {cluster_name})")
    else:
        print("\n[INFO][STATES] No suitable clustering variable found → using HC1")

    # 1) OLS log–log simple
    X1 = add_constant(df[["ln_price"]])
    if cluster_var is not None:
        ols1 = sm.OLS(df["ln_pax"], X1, missing="drop").fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_var, "use_correction": True}
        )
    else:
        ols1 = sm.OLS(df["ln_pax"], X1, missing="drop").fit(cov_type="HC1")
    print("\n=== [STATES] OLS log–log (raw) ===")
    print(ols1.summary())

    # 2) MLR avec contrôles
    controls = []
    if col_dist:     controls.append("ln_dist")
    if col_carriers: controls.append(col_carriers)
    if col_pop_o:    controls.append("ln_pop_o")
    if col_pop_d:    controls.append("ln_pop_d")

    X_cols = ["ln_price"] + controls
    if col_month and col_month in df.columns:
        d_m = pd.get_dummies(df[col_month].astype("Int64"), prefix="m", drop_first=True)
        X2 = pd.concat([df[X_cols], d_m], axis=1)
        month_dummies = list(d_m.columns)
    else:
        X2 = df[X_cols].copy()
        month_dummies = []

    X2 = add_constant(X2)
    if cluster_var is not None:
        mlr = sm.OLS(df["ln_pax"], X2, missing="drop").fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_var, "use_correction": True}
        )
    else:
        mlr = sm.OLS(df["ln_pax"], X2, missing="drop").fit(cov_type="HC1")

    print("\n=== [STATES] MLR log–log (with observed controls) ===")
    print(f"Controls used: {controls + month_dummies}")
    print(mlr.summary())

    # 3) HDFE 6 FE (o, d, t, od, ot, dt)
    have_hdfe = have_route and have_year
    try:
        if have_hdfe:
            from linearmodels.iv import AbsorbingLS
            print("\n=== [STATES] HDFE log–log (6 FE: o, d, t, od, ot, dt) ===")
            y = df["ln_pax"]
            X_hdfe = df[["ln_price"]]
            absorb = pd.DataFrame({
                "o":  df["origin"].astype("category"),
                "d":  df["destination"].astype("category"),
                "t":  df[col_year].astype("category"),
                "od": df["route_id"].astype("category"),
                "ot": df["origin_year"].astype("category"),
                "dt": df["dest_year"].astype("category"),
            })
            hdfe_mod = AbsorbingLS(y, X_hdfe, absorb=absorb)
            hdfe_res = hdfe_mod.fit(cov_type="robust")
            print(hdfe_res.summary)
        else:
            print("\n[STATES][HDFE] skipped: need origin, destination and year to build 6 FE.")
    except Exception as e:
        print("\n[STATES][HDFE] AbsorbingLS unavailable or other error:")
        print(e)
        print("→ Install:  pip install linearmodels")

    # 4) Panel FE (route & année)
    can_panel = have_route and have_year
    try:
        if can_panel:
            from linearmodels.panel import PanelOLS
            panel = df.set_index(["route_id", col_year]).sort_index()
            fe_X_cols = ["ln_price"]
            if col_dist:     fe_X_cols.append("ln_dist")
            if col_carriers: fe_X_cols.append(col_carriers)
            if col_pop_o:    fe_X_cols.append("ln_pop_o")
            if col_pop_d:    fe_X_cols.append("ln_pop_d")

            Y = panel["ln_pax"]
            Xfe = add_constant(panel[fe_X_cols])
            fe_mod = PanelOLS(Y, Xfe, entity_effects=True, time_effects=True)
            fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)
            print("\n=== [STATES] Panel FE (route & year) — SE clustered by route ===")
            print(f"Variables in X: {fe_X_cols}")
            print(fe_res.summary)
        else:
            print("\n[STATES][Panel FE] skipped: missing columns.")
            if not have_route:
                print("  - route_id unavailable (origin and/or destination missing)")
            if not have_year:
                print("  - year missing")
    except Exception as e:
        print("\n[STATES][Panel FE] linearmodels unavailable or other error:")
        print(e)
        print("→ Install:  pip install linearmodels")

    # 5) Graphique log–log (SAVE TO outputs/)
    x = df[col_price].to_numpy()
    y = df[col_pax].to_numpy()
    if x.size >= 2 and y.size >= 2:
        logx = np.log(x)
        logy = np.log(y)
        a, b = np.polyfit(logx, logy, 1)
        x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_line = np.exp(b) * x_line**a

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=8, alpha=0.5, label="State → city (≥ 5000 passengers/year)")
        plt.plot(x_line, y_line, linewidth=2, label=f"Log-log trend (slope ≈ {a:.3f})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Average ticket price (USD, log scale)")
        plt.ylabel("Total passengers from state to city (per year, log scale)")
        plt.title("State-level routes: passengers vs. price (log-log, 1996–2025)")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(OUTPUT_DIR, "routes_states_log_price_FE.png")
        plt.savefig(outpath, dpi=300)
        plt.show()
        print(f"[STATES][PLOT] Saved: {outpath}")
    else:
        print("\n[STATES][Plot] Pas assez d’observations pour tracer la tendance.")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    merged = load_and_clean_merged(MERGED_CSV)
    state_panel = build_state_city_year_from_merged(merged)

    run_cities_hdfe(merged)
    run_states_hdfe(state_panel)
