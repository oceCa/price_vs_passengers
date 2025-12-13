#!/usr/bin/env python3
# Reverse_causality_advanced_price_to_pax.py
# ============================================================
# Reverse causality test: does last year's average ticket price (t-1)
# predict this year's passenger volume (t)?
#
# Why lag the price?
# - Using price_{t-1} reduces simultaneity: current demand shocks in year t
#   cannot mechanically affect last year's price, making interpretation cleaner.
#
# Two specifications (always with lagged price):
#  1) LOG-LOG (elasticity):
#       ln(passengers_odt) = alpha + beta * ln(price_od,t-1) + ...
#     => beta: % change in passengers when last year's price increases by 1%.
#
#  2) LEVELS:
#       passengers_odt = gamma0 + gamma1 * (price_od,t-1 / SCALE_PRICE) + ...
#     => gamma1: absolute change in passengers for SCALE_PRICE units higher price last year.
#
# For each panel (cities, states) we estimate:
#   (A) Baseline OLS (no fixed effects) with route-clustered standard errors
#   (B) High-dimensional FE (6 FE) using AbsorbingLS:
#       - origin FE (o)
#       - destination FE (d)
#       - year FE (t)
#       - origin-destination FE (od)  [route FE]
#       - origin-year FE (ot)
#       - destination-year FE (dt)
#
# Bonus:
#   PanelOLS with route FE + year FE (od + t), clustered SE by route.
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =====================
# Configuration
# =====================
MERGED_CSV = "data/merged_air_travel_data.csv"   # Expected columns include: city1, city2, Year, quarter, passengers, Real price, ...
MIN_PAX = 0            # Minimum annual passengers threshold (0 means "no filter")
SCALE_PRICE = 10.0     # Scaling factor for the LEVELS specification (interpretation per +10 units of lagged price)

# ✅ Save everything under outputs/
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_reverse_causality")  # Where diagnostic figures will be saved


# =====================
# Small utilities
# =====================
def info(msg: str) -> None:
    """Print formatted progress messages (useful for long scripts)."""
    print(f"\n[INFO] {msg}")


def to_numeric(s):
    """Robust numeric conversion: coercing non-parsable values to NaN."""
    return pd.to_numeric(s, errors="coerce")


def _sanitize_label(label: str) -> str:
    """
    Create a filesystem-safe label for figure filenames:
    - lowercases
    - removes punctuation / accents
    - replaces spaces with underscores
    """
    return (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(">", "")
        .replace("<", "")
        .replace(",", "")
        .replace(":", "")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("/", "_")
    )


def _ensure_fig_dir():
    """Create outputs/ and outputs/figures_reverse_causality/ if needed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# 1) CITY PANEL (city1 -> city2 -> year)
# ============================================================
def prepare_city_panel(path: str) -> pd.DataFrame:
    """
    Build an annual panel at the route level (origin city -> destination city):
    - Aggregate quarterly rows into annual totals/averages
    - Construct route_id = "origin -> destination"
    - Create lagged price (t-1) within each route
    - Create log variables for the elasticity regression
    - Create identifiers needed for high-dimensional FE (origin_year, dest_year)
    """
    info(f"Loading quarterly city-level data from {path} ...")
    df = pd.read_csv(path)

    # Ensure numeric types for key variables
    df["passengers"] = to_numeric(df["passengers"])
    df["Real price"] = to_numeric(df["Real price"])

    # Keep only rows with complete essentials
    df = df.dropna(subset=["Year", "city1", "city2", "passengers", "Real price"])

    info("Aggregating to annual level (city1, city2, Year) ...")
    grp_cols = ["Year", "city1", "city2"]
    ann = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              total_passengers=("passengers", "sum"),  # annual passenger volume
              avg_price=("Real price", "mean")         # annual mean price (simple average across quarters)
          )
    )

    # Drop non-positive values (log requires strictly positive)
    ann = ann[(ann["total_passengers"] > 0) & (ann["avg_price"] > 0)]

    # Optional filter to remove very small flows
    if MIN_PAX > 0:
        ann = ann[ann["total_passengers"] >= MIN_PAX]
        info(f"Filtered flows with total_passengers >= {MIN_PAX} -> {len(ann)} rows left")

    # Build core identifiers
    ann["origin"] = ann["city1"].astype(str)
    ann["destination"] = ann["city2"].astype(str)
    ann["route_id"] = ann["origin"] + " -> " + ann["destination"]

    # Lag price within each route (t-1)
    ann = ann.sort_values(["route_id", "Year"])
    ann["price_lag1"] = ann.groupby("route_id")["avg_price"].shift(1)
    ann = ann.dropna(subset=["price_lag1"])  # keep only observations with lag available

    # Regression variables
    ann["ln_pax"] = np.log(ann["total_passengers"])
    ann["ln_price_lag1"] = np.log(ann["price_lag1"])
    ann["price_lag1_scaled"] = ann["price_lag1"] / SCALE_PRICE  # for LEVELS regression interpretation

    # FE identifiers: origin-year and destination-year
    year_str = ann["Year"].astype(int).astype(str)
    ann["origin_year"] = ann["origin"] + "_" + year_str
    ann["dest_year"]   = ann["destination"] + "_" + year_str

    info(f"City panel ready: {len(ann)} observations with lagged price available.")
    return ann


# ============================================================
# 2) STATE PANEL (origin_state -> destination_city -> year)
# ============================================================
def build_state_city_year_from_merged(path: str) -> pd.DataFrame:
    """
    Construct an annual panel where the origin is at the state level and the
    destination is at the city level.

    Steps:
    - Extract origin_state from the string in city1 (expects ", XX" at the end)
    - Aggregate to (Year, origin_state, city2):
        total_passengers = sum(passengers)
        avg_price = passenger-weighted average of price
    """
    info(f"Building state -> city -> year panel from {path} ...")
    df = pd.read_csv(path)

    # Basic schema checks
    if "passengers" not in df.columns:
        raise KeyError("Column 'passengers' not found in merged_air_travel_data.csv")

    # Clean passengers (e.g., "1,234" -> 1234)
    df["passengers"] = (
        df["passengers"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["passengers"] = to_numeric(df["passengers"])

    if "Year" not in df.columns:
        raise KeyError("Column 'Year' not found in merged_air_travel_data.csv")
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]  # optional sample window

    if "Real price" not in df.columns:
        raise KeyError("Column 'Real price' not found in merged_air_travel_data.csv")
    df["Real price"] = to_numeric(df["Real price"])

    if "city1" not in df.columns or "city2" not in df.columns:
        raise KeyError("Columns 'city1' and/or 'city2' not found in merged_air_travel_data.csv")

    # Extract two-letter state code from origin city string
    df["origin_state"] = df["city1"].str.extract(r',\s*([A-Z]{2})')[0]

    # Keep complete cases needed for aggregation
    df = df.dropna(subset=["Year", "origin_state", "city2", "passengers", "Real price"])

    # Aggregate with passenger-weighted price
    state_city_year = (
        df
        .groupby(["Year", "origin_state", "city2"])
        .apply(lambda g: pd.Series({
            "total_passengers": g["passengers"].sum(),
            "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
        }))
        .reset_index()
    )

    info(f"State->city annual aggregation ready: {len(state_city_year)} rows.")
    return state_city_year


def prepare_state_panel(path: str) -> pd.DataFrame:
    """
    Finalize the state-origin panel into a route-year dataset:
    - Build route_id = "origin_state -> destination_city"
    - Create lagged price by route
    - Create log variables + FE identifiers (origin_year, dest_year)
    """
    info(f"Preparing state-origin annual panel from {path} ...")
    df = build_state_city_year_from_merged(path)

    # Ensure numeric
    df["total_passengers"] = to_numeric(df["total_passengers"])
    df["avg_price"] = to_numeric(df["avg_price"])

    # Keep valid and positive values (log needs >0)
    df = df.dropna(subset=["Year", "origin_state", "city2", "total_passengers", "avg_price"])
    df = df[(df["total_passengers"] > 0) & (df["avg_price"] > 0)]

    if MIN_PAX > 0:
        df = df[df["total_passengers"] >= MIN_PAX]
        info(f"Filtered flows with total_passengers >= {MIN_PAX} -> {len(df)} rows left")

    # Identifiers
    df["origin"] = df["origin_state"].astype(str)
    df["destination"] = df["city2"].astype(str)
    df["route_id"] = df["origin"] + " -> " + df["destination"]

    # Lag price within each route
    df = df.sort_values(["route_id", "Year"])
    df["price_lag1"] = df.groupby("route_id")["avg_price"].shift(1)
    df = df.dropna(subset=["price_lag1"])

    # Regression variables
    df["ln_pax"] = np.log(df["total_passengers"])
    df["ln_price_lag1"] = np.log(df["price_lag1"])
    df["price_lag1_scaled"] = df["price_lag1"] / SCALE_PRICE

    # FE identifiers
    year_str = df["Year"].astype(int).astype(str)
    df["origin_year"] = df["origin"] + "_" + year_str
    df["dest_year"]   = df["destination"] + "_" + year_str

    info(f"State panel ready: {len(df)} observations with lagged price available.")
    return df


# ============================================================
# 3) Estimation: Baseline OLS (clustered by route)
# ============================================================
def run_ols_log(df: pd.DataFrame, label: str):
    info(f"Baseline OLS (log-log) for {label}: ln_pax_t ~ ln_price_lag1")

    if "route_id" in df.columns:
        model = smf.ols("ln_pax ~ ln_price_lag1", data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["route_id"], "use_correction": True}
        )
    else:
        model = smf.ols("ln_pax ~ ln_price_lag1", data=df).fit(cov_type="HC1")

    print(model.summary())
    return model


def run_ols_levels(df: pd.DataFrame, label: str):
    info(f"Baseline OLS (levels) for {label}: total_passengers_t ~ price_lag1_scaled")

    if "route_id" in df.columns:
        model = smf.ols("total_passengers ~ price_lag1_scaled", data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["route_id"], "use_correction": True}
        )
    else:
        model = smf.ols("total_passengers ~ price_lag1_scaled", data=df).fit(cov_type="HC1")

    print(model.summary())
    return model


# ============================================================
# 4) Estimation: High-dimensional fixed effects (6 FE) via AbsorbingLS
# ============================================================
def run_hdfe_log(df: pd.DataFrame, label: str):
    try:
        from linearmodels.iv import AbsorbingLS
    except ImportError:
        info(f"linearmodels (AbsorbingLS) not installed -> skipping HDFE log-log for {label}.")
        return None

    info(f"HDFE (log-log, 6 FE) for {label}")

    y = df["ln_pax"]
    X = df[["ln_price_lag1"]]

    absorb = pd.DataFrame({
        "o":  df["origin"].astype("category"),
        "d":  df["destination"].astype("category"),
        "t":  df["Year"].astype("category"),
        "od": df["route_id"].astype("category"),
        "ot": df["origin_year"].astype("category"),
        "dt": df["dest_year"].astype("category"),
    })

    if "route_id" in df.columns:
        clusters = df["route_id"]
        res = AbsorbingLS(y, X, absorb=absorb).fit(cov_type="clustered", clusters=clusters)
    else:
        res = AbsorbingLS(y, X, absorb=absorb).fit(cov_type="robust")

    print(res.summary)
    return res


def run_hdfe_levels(df: pd.DataFrame, label: str):
    try:
        from linearmodels.iv import AbsorbingLS
    except ImportError:
        info(f"linearmodels (AbsorbingLS) not installed -> skipping HDFE levels for {label}.")
        return None

    info(f"HDFE (levels, 6 FE) for {label}")

    y = df["total_passengers"]
    X = df[["price_lag1_scaled"]]

    absorb = pd.DataFrame({
        "o":  df["origin"].astype("category"),
        "d":  df["destination"].astype("category"),
        "t":  df["Year"].astype("category"),
        "od": df["route_id"].astype("category"),
        "ot": df["origin_year"].astype("category"),
        "dt": df["dest_year"].astype("category"),
    })

    if "route_id" in df.columns:
        clusters = df["route_id"]
        res = AbsorbingLS(y, X, absorb=absorb).fit(cov_type="clustered", clusters=clusters)
    else:
        res = AbsorbingLS(y, X, absorb=absorb).fit(cov_type="robust")

    print(res.summary)
    return res


# ============================================================
# 5) Estimation: Two-way FE panel (route + year) via PanelOLS
# ============================================================
def run_fe_panel(df: pd.DataFrame, label: str):
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        info(f"linearmodels.panel not installed -> skipping PanelOLS FE for {label}.")
        return None

    info(f"Panel FE (route & year) for {label} (log-log) ...")

    panel = df.set_index(["route_id", "Year"]).sort_index()

    Y = panel["ln_pax"]
    X = sm.add_constant(panel[["ln_price_lag1"]])

    fe_mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)

    print(fe_res.summary)
    return fe_res


# ============================================================
# 6) Plots: quick diagnostics and intuition
# ============================================================
def plot_loglog_relationship(df: pd.DataFrame, label: str):
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    x = df["ln_price_lag1"].values
    y = df["ln_pax"].values

    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.2, s=5)
    if not np.isnan(slope):
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, linewidth=2)

    ax.set_xlabel(r"$\ln(\text{price}_{t-1})$")
    ax.set_ylabel(r"$\ln(\text{passengers}_{t})$")
    ax.set_title(f"Reverse causality (log-log)\n{label}")

    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, f"loglog_pax_vs_pricelag_{safe}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    info(f"Saved figure: {outpath}")


def plot_levels_relationship(df: pd.DataFrame, label: str):
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    x = df["price_lag1_scaled"].values
    y = df["total_passengers"].values

    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.2, s=5)
    if not np.isnan(slope):
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, linewidth=2)

    ax.set_xlabel(r"$\text{price}_{t-1} / " + str(int(SCALE_PRICE)) + r"$")
    ax.set_ylabel(r"$\text{passengers}_{t}$")
    ax.set_title(f"Reverse causality (levels)\n{label}")

    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, f"levels_pax_vs_pricelag_{safe}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    info(f"Saved figure: {outpath}")


def plot_example_route_timeseries(df: pd.DataFrame, label: str):
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    if df.empty:
        return

    counts = df["route_id"].value_counts()
    top_route = counts.index[0]

    sub = df[df["route_id"] == top_route].sort_values("Year")
    if sub["Year"].nunique() < 3:
        return

    years = sub["Year"].values
    pax = sub["total_passengers"].values
    price = sub["avg_price"].values

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(years, pax, marker="o")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Passengers", rotation=90)
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(years, price, marker="s", linestyle="--")
    ax2.set_ylabel("Average price")

    fig.suptitle(f"Example route time series: {top_route}\n{label}")
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, f"timeseries_example_route_{safe}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    info(f"Saved figure: {outpath}")


def plot_all_diagnostics(df: pd.DataFrame, label: str):
    if df.empty:
        return
    plot_loglog_relationship(df, label)
    plot_levels_relationship(df, label)
    plot_example_route_timeseries(df, label)


# ============================================================
# 7) Main wrappers: run all models + plots for a given panel
# ============================================================
def run_all_for_panel(df: pd.DataFrame, label: str):
    if len(df) == 0:
        info(f"Panel {label} is empty after preparation (no lagged price available).")
        return

    run_ols_log(df, label)
    run_ols_levels(df, label)
    run_fe_panel(df, label)
    run_hdfe_log(df, label)
    run_hdfe_levels(df, label)
    plot_all_diagnostics(df, label)


def main():
    _ensure_fig_dir()

    if os.path.exists(MERGED_CSV):
        city_panel = prepare_city_panel(MERGED_CSV)
        run_all_for_panel(city_panel, "cities (city1 -> city2, annual)")
    else:
        info(f"File {MERGED_CSV} not found, skipping city panel.")

    if os.path.exists(MERGED_CSV):
        state_panel = prepare_state_panel(MERGED_CSV)
        run_all_for_panel(state_panel, "states (origin_state -> city2, annual)")
    else:
        info(f"File {MERGED_CSV} not found, skipping state panel.")


if __name__ == "__main__":
    main()
