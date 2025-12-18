#!/usr/bin/env python3
# scr/Reverse_Causality_spec_curve.py
# ------------------------------------------------------------
# City -> City recap (exactly 6 specs):
# 1) Naive OLS (no reverse, no FE)                 : ln_pax_t ~ ln_price_t
# 2) OLS + 2FE (no reverse)                        : ln_pax_t ~ ln_price_t + FE(route,year)
# 3) OLS + 6FE (no reverse, HDFE: o,d,t,od,ot,dt)  : ln_pax_t ~ ln_price_t + (o,d,t,od,ot,dt)
# 4) OLS + reverse (price_{t-1})                   : ln_pax_t ~ ln_price_{t-1}
# 5) OLS + 2FE + reverse (route+year)              : ln_pax_t ~ ln_price_{t-1} + FE(route,year)
# 6) OLS + 6FE + reverse (HDFE: o,d,t,od,ot,dt)    : ln_pax_t ~ ln_price_{t-1} + (o,d,t,od,ot,dt)
# ------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from linearmodels.panel import PanelOLS
from linearmodels.iv import AbsorbingLS

# ============================
# PATHS
# ============================
ROOT = Path(__file__).resolve().parents[1]
SCR  = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "merged_air_travel_data.csv"

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures_spec_curve"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# allow importing modules from scr/
sys.path.insert(0, str(SCR))

import elasticity_reg_w_FE as ELAS          # exact cleaning helper (price_t panel)
import Reverse_Causality_w_FE as RC         # exact reverse (lag) panel + FE/HDFE logic


# ============================
# 1) CITY->CITY panel (price_t)
# ============================
def build_city_panel_price_t(csv_path: str, min_pax: int = 5000) -> pd.DataFrame:
    merged = ELAS.load_and_clean_merged(csv_path)

    grouped = (
        merged
        .groupby(["Year", "city1", "city2"], as_index=False, sort=False)
        .apply(
            lambda g: pd.Series({
                "total_passengers": g["passengers"].sum(),
                "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_price", "total_passengers"])
    grouped = grouped[grouped["total_passengers"] >= min_pax].copy()

    if grouped.empty:
        raise ValueError(
            "[ERROR] City→city (price_t) panel is EMPTY after cleaning.\n"
            "Check: DATA_CSV path + columns + min_pax filter."
        )

    grouped["origin"] = grouped["city1"].astype(str)
    grouped["destination"] = grouped["city2"].astype(str)
    grouped["route_id"] = grouped["origin"] + " -> " + grouped["destination"]
    grouped["Year"] = grouped["Year"].astype(int)

    grouped["ln_pax"]   = np.log(grouped["total_passengers"].astype(float))
    grouped["ln_price"] = np.log(grouped["avg_price"].astype(float))

    ystr = grouped["Year"].astype(str)
    grouped["origin_year"] = grouped["origin"] + "_" + ystr
    grouped["dest_year"]   = grouped["destination"] + "_" + ystr

    return grouped


# ============================
# 2) Helpers for coefficient + CI
# ============================
def coef_ci_from_statsmodels(res, varname: str):
    beta = float(res.params[varname])
    lo, hi = res.conf_int().loc[varname].tolist()
    return beta, float(lo), float(hi)

def coef_ci_from_linearmodels(res, varname: str):
    beta = float(res.params[varname])
    try:
        ci = res.conf_int().loc[varname]
        lo, hi = float(ci["lower"]), float(ci["upper"])
        return beta, lo, hi
    except Exception:
        se = float(res.std_errors[varname]) if hasattr(res, "std_errors") else float(res.std_err[varname])
        lo, hi = beta - 1.96 * se, beta + 1.96 * se
        return beta, float(lo), float(hi)


# ============================
# 3) Estimations
# ============================
def estimate_price_t_models(df_t: pd.DataFrame):
    """
    (1) Naive OLS (no reverse, no FE): ln_pax ~ ln_price
    (2) OLS + 2FE (no reverse): PanelOLS with route + year
    (3) OLS + 6FE (no reverse): AbsorbingLS with o,d,t,od,ot,dt
    """
    # (1) Naive OLS
    m1 = smf.ols("ln_pax ~ ln_price", data=df_t).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_t["route_id"], "use_correction": True}
    )
    b1, lo1, hi1 = coef_ci_from_statsmodels(m1, "ln_price")
    n1 = int(m1.nobs)

    # (2) 2FE route + year (no reverse)
    panel = df_t.set_index(["route_id", "Year"]).sort_index()
    Y = panel["ln_pax"]
    X = sm.add_constant(panel[["ln_price"]])
    m2 = PanelOLS(Y, X, entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    b2, lo2, hi2 = coef_ci_from_linearmodels(m2, "ln_price")
    n2 = int(getattr(m2, "nobs", len(df_t)))

    # (3) 6FE (no reverse) via AbsorbingLS
    y = df_t["ln_pax"]
    X = df_t[["ln_price"]]
    absorb = pd.DataFrame({
        "o":  df_t["origin"].astype("category"),
        "d":  df_t["destination"].astype("category"),
        "t":  df_t["Year"].astype("category"),
        "od": df_t["route_id"].astype("category"),
        "ot": df_t["origin_year"].astype("category"),
        "dt": df_t["dest_year"].astype("category"),
    })
    m3 = AbsorbingLS(y, X, absorb=absorb).fit(
        cov_type="clustered", clusters=df_t["route_id"]
    )
    b3, lo3, hi3 = coef_ci_from_linearmodels(m3, "ln_price")
    n3 = int(getattr(m3, "nobs", len(df_t)))

    return (b1, lo1, hi1, n1), (b2, lo2, hi2, n2), (b3, lo3, hi3, n3)


def estimate_reverse_models(csv_path: str):
    """
    (4) OLS + reverse: ln_pax ~ ln_price_lag1
    (5) OLS + 2FE + reverse: route + year
    (6) OLS + 6FE + reverse: o,d,t,od,ot,dt
    Uses EXACT pipeline from Reverse_Causality_w_FE.py.
    """
    df_rc = RC.prepare_city_panel(csv_path)

    if df_rc.empty:
        raise ValueError("[ERROR] Reverse city panel is empty after lag construction.")

    # (4) reverse OLS
    m4 = smf.ols("ln_pax ~ ln_price_lag1", data=df_rc).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_rc["route_id"], "use_correction": True}
    )
    b4, lo4, hi4 = coef_ci_from_statsmodels(m4, "ln_price_lag1")
    n4 = int(m4.nobs)

    # (5) reverse + 2FE (route+year)
    panel = df_rc.set_index(["route_id", "Year"]).sort_index()
    Y = panel["ln_pax"]
    X = sm.add_constant(panel[["ln_price_lag1"]])
    m5 = PanelOLS(Y, X, entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    b5, lo5, hi5 = coef_ci_from_linearmodels(m5, "ln_price_lag1")
    n5 = int(getattr(m5, "nobs", len(df_rc)))

    # (6) reverse + 6FE (HDFE)
    y = df_rc["ln_pax"]
    X = df_rc[["ln_price_lag1"]]
    absorb = pd.DataFrame({
        "o":  df_rc["origin"].astype("category"),
        "d":  df_rc["destination"].astype("category"),
        "t":  df_rc["Year"].astype("category"),
        "od": df_rc["route_id"].astype("category"),
        "ot": df_rc["origin_year"].astype("category"),
        "dt": df_rc["dest_year"].astype("category"),
    })
    m6 = AbsorbingLS(y, X, absorb=absorb).fit(
        cov_type="clustered", clusters=df_rc["route_id"]
    )
    b6, lo6, hi6 = coef_ci_from_linearmodels(m6, "ln_price_lag1")
    n6 = int(getattr(m6, "nobs", len(df_rc)))

    return (b4, lo4, hi4, n4), (b5, lo5, hi5, n5), (b6, lo6, hi6, n6)


# ============================
# 4) Plot
# ============================
def plot_spec_curve(res_df: pd.DataFrame, filename="SpecCurve_City_to_City_6specs.png"):
    outpath = FIG_DIR / filename

    df = res_df.copy()
    df["err_low"] = df["Coefficient"] - df["CI_Lower"]
    df["err_up"]  = df["CI_Upper"] - df["Coefficient"]

    x = np.arange(len(df))

    plt.figure(figsize=(16, 6))
    plt.errorbar(
        x=x,
        y=df["Coefficient"].values,
        yerr=[df["err_low"].values, df["err_up"].values],
        fmt="o",
        capsize=10,
        linewidth=2
    )
    plt.axhline(0, linestyle="--", linewidth=1.5)
    plt.xticks(x, df["Model"].tolist())
    plt.title("City → City: Price Elasticity (log–log)\nSpecification curve (No reverse vs Reverse + FE/HDFE)",
              fontsize=16, fontweight="bold")
    plt.ylabel("Elasticity (log–log coefficient)")
    plt.xlabel("Regression model specification")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(df["Coefficient"].values):
        plt.text(i, v, f"{v:.3f}", fontsize=11, ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show(block=True)
    print(f"[INFO] Saved figure: {outpath}")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    csv_path = str(DATA_CSV)

    # (1)-(3) NO reverse from price_t panel
    df_t = build_city_panel_price_t(csv_path, min_pax=5000)
    (b1, lo1, hi1, n1), (b2, lo2, hi2, n2), (b3, lo3, hi3, n3) = estimate_price_t_models(df_t)

    # (4)-(6) Reverse from Reverse_Causality_w_FE.py
    (b4, lo4, hi4, n4), (b5, lo5, hi5, n5), (b6, lo6, hi6, n6) = estimate_reverse_models(csv_path)

    results = pd.DataFrame([
        {
            "Model": "Naive OLS\n(no reverse)\n(no FE)",
            "Coefficient": b1, "CI_Lower": lo1, "CI_Upper": hi1,
            "N": n1,
            "Source": "elasticity_simple_reg.py (city→city naive OLS)"
        },
        {
            "Model": "OLS + 2FE\n(no reverse)\n(route, year)",
            "Coefficient": b2, "CI_Lower": lo2, "CI_Upper": hi2,
            "N": n2,
            "Source": "elasticity_simple_reg.py (city→city 2FE route+year)"
        },
        {
            "Model": "OLS + 6FE\n(no reverse)\n(o,d,t,od,ot,dt)",
            "Coefficient": b3, "CI_Lower": lo3, "CI_Upper": hi3,
            "N": n3,
            "Source": "elasticity_reg_w_FE.py (city→city HDFE 6FE)"
        },
        {
            "Model": "OLS + reverse\n(price_{t-1})",
            "Coefficient": b4, "CI_Lower": lo4, "CI_Upper": hi4,
            "N": n4,
            "Source": "Reverse_Causality_w_FE.py (reverse OLS)"
        },
        {
            "Model": "OLS + 2FE\n+ reverse\n(route, year)",
            "Coefficient": b5, "CI_Lower": lo5, "CI_Upper": hi5,
            "N": n5,
            "Source": "Reverse_Causality_w_FE.py (reverse + 2FE route+year)"
        },
        {
            "Model": "OLS + 6FE\n+ reverse\n(o,d,t,od,ot,dt)",
            "Coefficient": b6, "CI_Lower": lo6, "CI_Upper": hi6,
            "N": n6,
            "Source": "Reverse_Causality_w_FE.py (reverse + 6FE HDFE)"
        },
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "spec_curve_city_to_city_6specs.csv"
    results.to_csv(out_csv, index=False)
    print(f"[INFO] Saved CSV: {out_csv}")

    plot_spec_curve(results, filename="SpecCurve_City_to_City_6specs.png")
