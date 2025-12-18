#!/usr/bin/env python3
# Elasticity_cities_states_reg_w_FE_from_merged.py
# -----------------------------------------------------------
# Same computations + same saved figures,
# BUT:
#   - terminal prints: NONE
#   - display: ONLY city->city plot
# -----------------------------------------------------------

import os
import contextlib
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
OUTPUT_DIR = "outputs"


def _ensure_out_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


@contextlib.contextmanager
def silence_output():
    """Mute stdout/stderr (prints + warnings) during a block."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# -----------------------
# Helpers robustes communs
# -----------------------
def first_exact(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Colonnes candidates absentes: {candidates}")
    return None


def find_by_keywords(df, any_tokens, required=False):
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
    df = pd.read_csv(path)

    if "passengers" not in df.columns:
        raise KeyError("Colonne 'passengers' absente du CSV.")
    df["passengers"] = df["passengers"].astype(str).str.replace(",", "", regex=False)
    df["passengers"] = to_numeric_safe(df["passengers"])

    if "Real price" not in df.columns:
        raise KeyError("Colonne 'Real price' absente du CSV.")
    df["Real price"] = to_numeric_safe(df["Real price"])

    for col in ["Year", "city1", "city2"]:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' absente du CSV.")
    df["Year"] = to_numeric_safe(df["Year"])

    df = df.dropna(subset=["passengers", "Real price", "Year", "city1", "city2"])
    df = df[(df["passengers"] > 0) & (df["Real price"] > 0)]
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]
    return df


# -----------------------
# Construction panel états → villes depuis merged (en mémoire)
# -----------------------
def build_state_city_year_from_merged(df_merged: pd.DataFrame) -> pd.DataFrame:
    df = df_merged.copy()
    df["origin_state"] = df["city1"].str.extract(r",\s*([A-Z]{2})")[0]
    df = df.dropna(subset=["Year", "origin_state", "city2", "passengers", "Real price"])

    state_city_year = (
        df.groupby(["Year", "origin_state", "city2"])
          .apply(lambda g: pd.Series({
              "total_passengers": g["passengers"].sum(),
              "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
          }))
          .reset_index()
    )
    return state_city_year


# ============================================================
# BLOC 1 : CITY-LEVEL (city1 -> city2 -> Year) avec HDFE & FE
# ============================================================
def run_cities_hdfe(df_merged: pd.DataFrame, show_plot: bool = True):
    _ensure_out_dir()
    raw = df_merged.copy()

    col_pax_raw   = first_exact(raw, ["passengers"], required=True)
    col_price_raw = first_exact(raw, ["Real price", "real_price", "price", "fare"])
    if col_price_raw is None:
        col_price_raw = find_by_keywords(raw, ["price", "fare"], required=True)

    col_city1 = first_exact(raw, ["city1"]) or find_by_keywords(raw, ["city1","orig","from","source"], required=True)
    col_city2 = first_exact(raw, ["city2"]) or find_by_keywords(raw, ["city2","dest","to","target"], required=True)
    col_year  = first_exact(raw, ["Year","year"]) or find_by_keywords(raw, ["year","yr"], required=True)

    raw[col_pax_raw]   = raw[col_pax_raw].astype(str).str.replace(",", "", regex=False)
    raw[col_pax_raw]   = to_numeric_safe(raw[col_pax_raw])
    raw[col_price_raw] = to_numeric_safe(raw[col_price_raw])

    raw = raw.dropna(subset=[col_pax_raw, col_price_raw, col_city1, col_city2, col_year])
    raw = raw[(raw[col_pax_raw] > 0) & (raw[col_price_raw] > 0)]
    raw = raw[(raw[col_year] >= 1996) & (raw[col_year] <= 2025)]

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

    # Logs & IDs
    grouped["ln_pax"]   = np.log(grouped["total_passengers"])
    grouped["ln_price"] = np.log(grouped["avg_price"])

    grouped["origin"]      = grouped[col_city1].astype(str)
    grouped["destination"] = grouped[col_city2].astype(str)
    grouped["route_id"]    = grouped["origin"] + " -> " + grouped["destination"]
    year_col = col_year
    year_str = grouped[year_col].astype(int).astype(str)
    grouped["origin_year"] = grouped["origin"]      + "_" + year_str
    grouped["dest_year"]   = grouped["destination"] + "_" + year_str

    cluster_var = grouped["route_id"]

    # 1) OLS
    X1 = add_constant(grouped[["ln_price"]])
    sm.OLS(grouped["ln_pax"], X1, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )

    # 2) MLR (controls not in aggregated data, keep structure)
    X2 = add_constant(grouped[["ln_price"]])
    sm.OLS(grouped["ln_pax"], X2, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )

    # 3) HDFE 6FE
    try:
        from linearmodels.iv import AbsorbingLS
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
        AbsorbingLS(y, X_hdfe, absorb=absorb).fit(cov_type="robust")
    except Exception:
        pass

    # 4) Two-way FE panel
    try:
        from linearmodels.panel import PanelOLS
        panel = grouped.set_index(["route_id", year_col]).sort_index()
        Y = panel["ln_pax"]
        Xfe = add_constant(panel[["ln_price"]])
        PanelOLS(Y, Xfe, entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
    except Exception:
        try:
            fe_formula = f"ln_pax ~ ln_price + C(route_id) + C({year_col})"
            smf.ols(fe_formula, data=grouped).fit(
                cov_type="cluster", cov_kwds={"groups": grouped["route_id"]}
            )
        except Exception:
            pass

    # 5) Plot city->city (SAVE + optional SHOW)
    x = grouped["avg_price"].to_numpy()
    y = grouped["total_passengers"].to_numpy()

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

    if show_plot:
        plt.show()
        plt.close()
    else:
        plt.close()


# ============================================================
# BLOC 2 : STATE-LEVEL (origin_state -> city2 -> Year) HDFE & FE
# ============================================================
def run_states_hdfe(state_df: pd.DataFrame):
    _ensure_out_dir()
    df = state_df.copy()

    col_pax   = first_exact(df, ["total_passengers", "passengers"])
    if col_pax is None:
        col_pax = find_by_keywords(df, ["passenger", "pax"], required=True)

    col_price = first_exact(df, ["avg_price", "Real price", "real_price"])
    if col_price is None:
        col_price = find_by_keywords(df, ["price", "fare", "ticket"], required=True)

    col_year = first_exact(df, ["year", "Year"])
    if col_year is None:
        col_year = find_by_keywords(df, ["year", "yr"], required=False)

    col_o = first_exact(df, ["origin_state", "origin", "state_from", "o_state"])
    if col_o is None:
        col_o = find_by_keywords(df, ["orig", "state", "from"], required=False)

    col_d = first_exact(df, ["dest_city", "destination_city", "city_to", "d_city"])
    if col_d is None:
        col_d = first_exact(df, ["city2", "city"], required=False)
    if col_d is None:
        col_d = find_by_keywords(df, ["dest", "to", "city"], required=False)

    df[col_pax]   = to_numeric_safe(df[col_pax])
    df[col_price] = to_numeric_safe(df[col_price])

    base_needed = [col_pax, col_price]
    if col_year:
        base_needed.append(col_year)
    df = df.dropna(subset=[c for c in base_needed if c is not None])
    df = df[df[col_pax] >= 5000]
    df = clean_positive(df, [col_pax, col_price])
    if df.empty:
        return

    df["ln_pax"]   = np.log(df[col_pax])
    df["ln_price"] = np.log(df[col_price])

    have_route = (col_o is not None) and (col_d is not None)
    have_year = col_year is not None
    if have_route:
        df["origin"]      = df[col_o].astype(str)
        df["destination"] = df[col_d].astype(str)
        df["route_id"]    = df["origin"] + " -> " + df["destination"]
        cluster_var = df["route_id"]
    else:
        cluster_var = None

    if have_route and have_year:
        year_str = df[col_year].astype(int).astype(str)
        df["origin_year"] = df["origin"]      + "_" + year_str
        df["dest_year"]   = df["destination"] + "_" + year_str

    # OLS / MLR / HDFE / FE computed but no prints
    X1 = add_constant(df[["ln_price"]])
    if cluster_var is not None:
        sm.OLS(df["ln_pax"], X1, missing="drop").fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_var, "use_correction": True}
        )
    else:
        sm.OLS(df["ln_pax"], X1, missing="drop").fit(cov_type="HC1")

    X2 = add_constant(df[["ln_price"]])
    if cluster_var is not None:
        sm.OLS(df["ln_pax"], X2, missing="drop").fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_var, "use_correction": True}
        )
    else:
        sm.OLS(df["ln_pax"], X2, missing="drop").fit(cov_type="HC1")

    try:
        if have_route and have_year:
            from linearmodels.iv import AbsorbingLS
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
            AbsorbingLS(y, X_hdfe, absorb=absorb).fit(cov_type="robust")
    except Exception:
        pass

    try:
        if have_route and have_year:
            from linearmodels.panel import PanelOLS
            panel = df.set_index(["route_id", col_year]).sort_index()
            Y = panel["ln_pax"]
            Xfe = add_constant(panel[["ln_price"]])
            PanelOLS(Y, Xfe, entity_effects=True, time_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )
    except Exception:
        pass

    # Plot states: SAVE ONLY (NO SHOW)
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
        plt.close()  # ✅ no display


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    _ensure_out_dir()

    # Silence everything except the final city plot window
    with silence_output():
        merged = load_and_clean_merged(MERGED_CSV)
        state_panel = build_state_city_year_from_merged(merged)
        run_states_hdfe(state_panel)

    # City block: computations silent, but plot shows
    with silence_output():
        merged = load_and_clean_merged(MERGED_CSV)
    run_cities_hdfe(merged, show_plot=True)
