#!/usr/bin/env python3
# States_advanced_causal_from_merged.py
# ============================================================
# Advanced causal analysis for state -> city price elasticity
# using ONLY merged_air_travel_data.csv.
#
# FIGURES ONLY VERSION:
#   - No CSV / tables are written to disk.
#   - Only figures are saved into outputs/figures_states_advanced/
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from pathlib import Path

# ============== CONFIG (paths anchored to this file) ==============
BASE = Path(__file__).resolve().parents[1]  # Project_Notebook/ (remonte d’un niveau depuis scr/)
MERGED_CSV = str((BASE / "data" / "merged_air_travel_data.csv").resolve())
CTRL_CSV   = str((BASE / "route_year_controls.csv").resolve())
OUTDIR     = (BASE / "outputs" / "figures_states_advanced")
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_YEAR = 1996
MAX_YEAR = 2025

# ============ HELPERS ==============
def first_exact(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required columns: {candidates}")
    return None

def find_kw(df, tokens, required=False):
    toks = [t.lower() for t in tokens]
    for c in df.columns:
        if any(t in c.lower() for t in toks):
            return c
    if required:
        raise KeyError(f"No column contains any of: {tokens}")
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce") if s is not None else None

def log_safe(s):
    return np.log(s.replace(0, np.nan))

def positive(df, cols):
    for c in cols:
        df = df[df[c].notna() & (df[c] > 0)]
    return df

def info(msg):
    print(f"\n[INFO] {msg}")

# ============ BUILD STATES PANEL FROM MERGED ==============
def build_state_city_year_from_merged(path: str) -> pd.DataFrame:
    """
    Recrée en mémoire l'équivalent de states_departures.csv à partir de merged_air_travel_data.csv.

    Sort un DataFrame avec colonnes:
        Year, origin_state, city2, total_passengers, avg_price
    où:
        - total_passengers = somme des passagers de cet État vers cette ville cette année
        - avg_price = prix moyen pondéré par les passagers (colonne 'Real price')
    """
    info(f"Building state → city → year panel from {path} ...")
    df = pd.read_csv(path)

    # 1) Passengers (avec ou sans virgules)
    if "passengers" not in df.columns:
        raise KeyError("Column 'passengers' not found in merged_air_travel_data.csv")

    df["passengers"] = (
        df["passengers"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["passengers"] = to_num(df["passengers"])

    # 2) Années utiles
    if "Year" not in df.columns:
        raise KeyError("Column 'Year' not found in merged_air_travel_data.csv")

    df["Year"] = to_num(df["Year"])
    df = df[(df["Year"] >= MIN_YEAR) & (df["Year"] <= MAX_YEAR)]

    # 3) Prix réel
    if "Real price" not in df.columns:
        raise KeyError("Column 'Real price' not found in merged_air_travel_data.csv")
    df["Real price"] = to_num(df["Real price"])

    # 4) Extraire l'État depuis city1
    if "city1" not in df.columns:
        raise KeyError("Column 'city1' not found in merged_air_travel_data.csv")
    if "city2" not in df.columns:
        raise KeyError("Column 'city2' not found in merged_air_travel_data.csv")

    df["origin_state"] = df["city1"].str.extract(r",\s*([A-Z]{2})")[0]

    # 5) Drop NA avant agrégation
    df = df.dropna(subset=["Year", "origin_state", "city2", "passengers", "Real price"])

    # 6) Agrégation Year + origin_state + city2
    state_city_year = (
        df
        .groupby(["Year", "origin_state", "city2"])
        .apply(lambda g: pd.Series({
            "total_passengers": g["passengers"].sum(),
            "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
        }))
        .reset_index()
    )

    info(f"State panel built: {len(state_city_year)} rows.")
    return state_city_year


# ============ LOAD (FROM MERGED) ==============
if not os.path.exists(MERGED_CSV):
    raise FileNotFoundError(f"{MERGED_CSV} not found.")

df = build_state_city_year_from_merged(MERGED_CSV)

# Basic columns (robust detection)
col_pax   = first_exact(df, ["total_passengers", "passengers"]) or find_kw(df, ["passenger", "pax"], required=True)
col_price = first_exact(df, ["avg_price", "Real price", "real_price"]) or find_kw(df, ["price","fare","ticket"], required=True)
col_year  = first_exact(df, ["year","Year"]) or find_kw(df, ["year","yr"], required=True)

# State origin and destination city for route_id
col_o = first_exact(df, ["origin_state","origin","state_from","o_state"]) or find_kw(df, ["state","orig","from"], required=False)
col_d = (
    first_exact(df, ["dest_city","destination_city","city_to","d_city"])
    or first_exact(df, ["city2", "city"])
    or find_kw(df, ["dest","to","city"], required=False)
)

# Optional controls that might already be in this panel
col_dist     = first_exact(df, ["distance","Distance","dist_miles","dist_km"], required=False) or find_kw(df, ["dist"], False)
col_carriers = first_exact(df, ["carriers","n_carriers","num_carriers"], required=False) or find_kw(df, ["carrier","airline","comp"], False)
col_month    = first_exact(df, ["month","Month"], required=False)

# Clean types
df[col_pax]   = to_num(df[col_pax])
df[col_price] = to_num(df[col_price])
if col_dist:     df[col_dist]     = to_num(df[col_dist])
if col_carriers: df[col_carriers] = to_num(df[col_carriers])
if col_month and df[col_month].dtype.name == "object":
    df[col_month] = pd.to_numeric(df[col_month], errors="coerce")

# Minimal cleaning
df = df.dropna(subset=[col_pax, col_price, col_year])
df = df[df[col_pax] >= 5000]
df = positive(df, [col_pax, col_price])

# Build route_id if possible
have_route = (col_o is not None) and (col_d is not None)
if have_route:
    df["route_id"] = df[col_o].astype(str) + " -> " + df[col_d].astype(str)
else:
    info("route_id not available (missing origin_state or destination city). FE/DiD may be limited.")

# ============ OPTIONAL MERGE CONTROLS (READ-ONLY) ============
controls_merged = False
if os.path.exists(CTRL_CSV):
    info(f"Merging optional controls from {CTRL_CSV} (read-only) ...")
    ctrl = pd.read_csv(CTRL_CSV)

    key_cols = None
    if have_route and ("route_id" in ctrl.columns) and (col_year in ctrl.columns):
        key_cols = ["route_id", col_year]
        if "route_id" not in df.columns:
            df["route_id"] = df[col_o].astype(str) + " -> " + df[col_d].astype(str)
        df = df.merge(ctrl, on=key_cols, how="left")
        controls_merged = True
    elif (col_o in ctrl.columns) and (col_d in ctrl.columns) and (col_year in ctrl.columns):
        key_cols = [col_o, col_d, col_year]
        df = df.merge(ctrl, on=key_cols, how="left")
        controls_merged = True

    if controls_merged:
        info(f"Controls merged on {key_cols}. Columns now: {len(df.columns)}")
    else:
        info("Could not align control keys; skipping merge.")
else:
    info("No route_year_controls.csv found. Proceeding without external controls.")

# ============ LOG VARIABLES ============
df["ln_pax"]   = np.log(df[col_pax])
df["ln_price"] = np.log(df[col_price])

maybe_controls = []
for cand, newname in [
    ("distance", "ln_dist"),
    ("dist_km", "ln_dist"),
    ("dist_miles", "ln_dist"),
    ("pop_origin", "ln_pop_o"),
    ("origin_pop", "ln_pop_o"),
    ("pop_dest", "ln_pop_d"),
    ("destination_pop", "ln_pop_d"),
]:
    if cand in df.columns:
        df[newname] = log_safe(to_num(df[cand]))
        maybe_controls.append(newname)

if col_carriers and col_carriers in df.columns:
    maybe_controls.append(col_carriers)

for extra in ["has_LCC", "seat_capacity", "airport_fees", "fuel_price", "fuel_x_dist"]:
    if extra in df.columns:
        df[extra] = to_num(df[extra])
        maybe_controls.append(extra)

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_pax","ln_price"])

# ============ CLUSTER VAR ============
cluster_var = None
cluster_name = None
if "route_id" in df.columns:
    cluster_var = df["route_id"]
    cluster_name = "route_id"
elif col_o is not None:
    cluster_var = df[col_o].astype(str)
    cluster_name = col_o

if cluster_var is not None:
    info(f"Using clustered standard errors (groups = {cluster_name})")
else:
    info("No suitable clustering variable found → using HC1 (heteroskedasticity-robust).")

# ============ 1) OLS LOG–LOG ============
info("OLS log–log (cross-section) ...")
X1 = add_constant(df[["ln_price"]])
if cluster_var is not None:
    ols = sm.OLS(df["ln_pax"], X1, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )
else:
    ols = sm.OLS(df["ln_pax"], X1, missing="drop").fit(cov_type="HC1")
print(ols.summary())

# ============ 2) MLR with observed controls ============
info("MLR log–log with observed controls (if any) ...")
X_cols = ["ln_price"] + maybe_controls
X2 = df[X_cols].copy()

if col_month and col_month in df.columns:
    d_m = pd.get_dummies(df[col_month].astype("Int64"), prefix="m", drop_first=True)
    X2 = pd.concat([X2, d_m], axis=1)
    used_controls = maybe_controls + list(d_m.columns)
else:
    used_controls = maybe_controls

X2 = add_constant(X2)
if cluster_var is not None:
    mlr = sm.OLS(df["ln_pax"], X2, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )
else:
    mlr = sm.OLS(df["ln_pax"], X2, missing="drop").fit(cov_type="HC1")
print(f"\nControls used: {used_controls}")
print(mlr.summary())

# ============ 3) Two-way FE (route & year) ============
can_panel = have_route and (col_year is not None)
if can_panel:
    try:
        from linearmodels.panel import PanelOLS
        info("Two-way FE (linearmodels): route & year, clustered by route ...")
        panel = df.set_index(["route_id", col_year]).sort_index()

        fe_X_cols = ["ln_price"]
        for c in ["ln_dist", col_carriers, "ln_pop_o", "ln_pop_d", "has_LCC",
                  "seat_capacity", "airport_fees", "fuel_price", "fuel_x_dist"]:
            if (c in panel.columns) and panel[c].notna().any():
                fe_X_cols.append(c)

        Y = panel["ln_pax"]
        Xfe = add_constant(panel[fe_X_cols])
        fe_mod = PanelOLS(Y, Xfe, entity_effects=True, time_effects=True)
        fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)

        print(f"\nVariables in FE X: {fe_X_cols}")
        print(fe_res.summary)

    except Exception as e:
        info(f"linearmodels not available or failed: {e}\nFalling back to dummy FE with clustered SE.")
        fe_formula = f"ln_pax ~ ln_price + C(route_id) + C({col_year})"
        for c in ["ln_dist", col_carriers, "ln_pop_o", "ln_pop_d", "has_LCC",
                  "seat_capacity", "airport_fees", "fuel_price", "fuel_x_dist"]:
            if c in df.columns and df[c].notna().any():
                fe_formula += f" + {c}"

        fe_fallback = smf.ols(fe_formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["route_id"]}
        )
        print(fe_fallback.summary())
else:
    info("Two-way FE skipped (need route_id and year).")

# ============ 4) DML elasticity (optional; no files written) ============
try:
    from doubleml import DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    info("DML partialling-out (Random Forest learners) ...")
    control_cols = []
    for c in used_controls:
        if c not in control_cols and c != "const":
            control_cols.append(c)

    X = df[control_cols].fillna(0.0).to_numpy() if len(control_cols) else np.zeros((len(df), 0))
    y = df["ln_pax"].to_numpy()
    d = df["ln_price"].to_numpy()

    learner = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    dml = DoubleMLPLR.from_arrays(X=X, y=y, d=d, ml_g=learner, ml_m=learner, n_folds=5)
    dml.fit()
    print(f"\n[DML] coef on ln_price = {float(dml.coef):.3f}  (se = {float(dml.se):.3f})")

except Exception as e:
    info(f"DML skipped (install doubleml & scikit-learn to enable): {e}")

# ============ 5) DiD / Event-study (optional) ============
if have_route and "event_year" in df.columns:
    info("Event-study around event_year (fallback via statsmodels) ...")
    df_es = df.dropna(subset=["route_id", col_year, "event_year", "ln_pax"]).copy()
    df_es["event_time"] = df_es[col_year] - df_es["event_year"]
    df_es = df_es[(df_es["event_time"] >= -5) & (df_es["event_time"] <= 5)]

    ks = sorted([k for k in df_es["event_time"].unique() if k != -1])
    rhs = " + ".join([f"C(event_time, Treatment(reference=-1))[T.{k}]" for k in ks])
    formula = f"ln_pax ~ {rhs} + C(route_id) + C({col_year})"
    if "ln_price" in df_es.columns:
        formula += " + ln_price"

    est = smf.ols(formula, data=df_es).fit(
        cov_type="cluster", cov_kwds={"groups": df_es["route_id"]}
    )
    print(est.summary())

    coefs, ses = [], []
    for k in ks:
        name = f"C(event_time, Treatment(reference=-1))[T.{k}]"
        coefs.append(est.params.get(name, np.nan))
        ses.append(est.bse.get(name, np.nan))

    plt.figure(figsize=(8, 4.5))
    plt.axhline(0, color="gray", lw=1)
    plt.errorbar(ks, coefs, yerr=1.96*np.array(ses), fmt="o-")
    plt.xlabel("Event time (years)")
    plt.ylabel("Effect on ln(passengers) vs. t=-1")
    plt.title("Event-study (state → city)")
    plt.tight_layout()

    out_es = OUTDIR / "event_study_states.png"
    plt.savefig(out_es, dpi=300)
    plt.show()
    print(f"[INFO] Saved figure at: {out_es.resolve()}")

else:
    info("Event-study skipped (need route_id and event_year).")

# ============ 6) IV 2SLS (optional; no files written) ============
iv_ready = have_route and all(c in df.columns for c in ["ln_price", "ln_pax", col_year])
possible_iv = [c for c in ["fuel_x_dist", "airport_fees", "fuel_price"] if c in df.columns]

if iv_ready and possible_iv:
    info(f"IV 2SLS with instruments: {possible_iv}")
    try:
        from linearmodels.iv import IV2SLS

        # FE as dummies (can be heavy, but ok if dataset manageable)
        fe_d = pd.get_dummies(df[["route_id", col_year]].astype(str), drop_first=True)

        Y = df["ln_pax"]
        D = df["ln_price"]
        Z = df[possible_iv].copy()
        X = fe_d

        iv = IV2SLS(Y, X, D, Z).fit(cov_type="clustered", clusters=df["route_id"])
        print(iv.summary)

    except Exception as e:
        info(f"IV skipped (linearmodels missing or error): {e}")
else:
    info("IV 2SLS skipped (need instruments columns like fuel_x_dist / airport_fees).")

# ============ 7) FIGURE: LOG–LOG SCATTER ============
info("Saving log–log scatter with fitted trend ...")
x = df[col_price].to_numpy()
y = df[col_pax].to_numpy()

if x.size >= 2 and y.size >= 2:
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)

    x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_line = np.exp(b) * x_line**a

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=8, alpha=0.5, label="State → city (≥ 5000 passengers/year)")
    plt.plot(x_line, y_line, lw=2, label=f"Log-log trend (slope ≈ {a:.3f})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Average ticket price (USD, log scale)")
    plt.ylabel("Total passengers from state to city (per year, log scale)")
    plt.title("State-level routes: price vs passengers (log–log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outp = OUTDIR / "states_loglog_price.png"
    plt.savefig(outp, dpi=300)
    plt.show()
    print(f"[INFO] Saved figure at: {outp.resolve()}")

else:
    info("Not enough observations for log–log plot.")
