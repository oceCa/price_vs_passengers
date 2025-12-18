#!/usr/bin/env python3
# Reverse_Causality_w_FE.py
# ============================================================
# Reverse causality test: does last year's average ticket price (t-1)
# predict this year's passenger volume (t)?
#
# Panels:
#   1) Cities:  city1 -> city2 -> year
#   2) States:  origin_state -> city2 -> year
#
# For each panel we estimate (and still SAVE outputs as before):
#   (A) Baseline OLS (log-log) with route-clustered SE
#   (B) Baseline OLS (levels) with route-clustered SE
#   (C) Two-way FE panel (route + year) via PanelOLS (if available)
#   (D) High-dimensional FE (6 FE) via AbsorbingLS (if available):
#       o, d, t, od, ot, dt
#   (E) Diagnostic figures (saved)
#
# DISPLAY ONLY (your request):
#   - Show ONLY city plots (no states plots)
#   - Show ONLY the "main" city plot (log-log pax vs price_{t-1})
#   - Print NOTHING in terminal (stdout suppressed), but errors still show
# ============================================================

import os
import contextlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =====================
# Configuration
# =====================
MERGED_CSV = "data/merged_air_travel_data.csv"

MIN_PAX = 0            # Minimum annual passengers threshold (0 = no filter)
SCALE_PRICE = 10.0     # Scaling for LEVELS regression: +1 unit = +SCALE_PRICE USD in lagged price

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_reverse_causality")

# ---- DISPLAY CONTROL (only affects plt.show + terminal prints) ----
SHOW_CITY_PLOTS = True          # show city plots
SHOW_STATE_PLOTS = False        # do NOT show state plots
SHOW_ONLY_CITY_MAIN = True      # show only the main city log-log plot (still save all figures)
QUIET_TERMINAL = True           # suppress stdout prints (errors still visible)


# =====================
# Utilities
# =====================
def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _sanitize_label(label: str) -> str:
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
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


@contextlib.contextmanager
def _quiet_stdout(enabled: bool = True):
    """
    Hide stdout prints (keeps stderr so tracebacks still show).
    This matches: "nothing in terminal" while preserving errors.
    """
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# ============================================================
# 1) CITY PANEL (city1 -> city2 -> year)
# ============================================================
def prepare_city_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Key columns expected
    for c in ["Year", "city1", "city2", "passengers", "Real price"]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {path}")

    df["passengers"] = to_numeric(df["passengers"])
    df["Real price"] = to_numeric(df["Real price"])
    df["Year"] = to_numeric(df["Year"])

    df = df.dropna(subset=["Year", "city1", "city2", "passengers", "Real price"])
    df = df[(df["passengers"] > 0) & (df["Real price"] > 0)]
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]

    # Annual aggregation (as in your original script: simple mean across quarters)
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

    ann["origin"] = ann["city1"].astype(str)
    ann["destination"] = ann["city2"].astype(str)
    ann["route_id"] = ann["origin"] + " -> " + ann["destination"]

    # Lag price within route
    ann = ann.sort_values(["route_id", "Year"])
    ann["price_lag1"] = ann.groupby("route_id")["avg_price"].shift(1)
    ann = ann.dropna(subset=["price_lag1"])

    # Regression variables
    ann["ln_pax"] = np.log(ann["total_passengers"])
    ann["ln_price_lag1"] = np.log(ann["price_lag1"])
    ann["price_lag1_scaled"] = ann["price_lag1"] / SCALE_PRICE

    # FE identifiers
    year_str = ann["Year"].astype(int).astype(str)
    ann["origin_year"] = ann["origin"] + "_" + year_str
    ann["dest_year"] = ann["destination"] + "_" + year_str

    return ann


# ============================================================
# 2) STATE PANEL (origin_state -> city2 -> year)
# ============================================================
def build_state_city_year_from_merged(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in ["Year", "city1", "city2", "passengers", "Real price"]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {path}")

    df["passengers"] = (
        df["passengers"].astype(str).str.replace(",", "", regex=False)
    )
    df["passengers"] = to_numeric(df["passengers"])
    df["Real price"] = to_numeric(df["Real price"])
    df["Year"] = to_numeric(df["Year"])

    df = df.dropna(subset=["Year", "city1", "city2", "passengers", "Real price"])
    df = df[(df["passengers"] > 0) & (df["Real price"] > 0)]
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]

    df["origin_state"] = df["city1"].str.extract(r",\s*([A-Z]{2})")[0]
    df = df.dropna(subset=["origin_state"])

    def _agg_weighted(g):
        pax = g["passengers"].sum()
        return pd.Series({
            "total_passengers": pax,
            "avg_price": (g["Real price"] * g["passengers"]).sum() / pax if pax > 0 else np.nan
        })

    # keep compatibility across pandas versions
    try:
        out = (
            df.groupby(["Year", "origin_state", "city2"])
              .apply(_agg_weighted, include_groups=False)
              .reset_index()
        )
    except TypeError:
        out = (
            df.groupby(["Year", "origin_state", "city2"])
              .apply(_agg_weighted)
              .reset_index()
        )

    return out


def prepare_state_panel(path: str) -> pd.DataFrame:
    df = build_state_city_year_from_merged(path)

    df["total_passengers"] = to_numeric(df["total_passengers"])
    df["avg_price"] = to_numeric(df["avg_price"])
    df["Year"] = to_numeric(df["Year"])

    df = df.dropna(subset=["Year", "origin_state", "city2", "total_passengers", "avg_price"])
    df = df[(df["total_passengers"] > 0) & (df["avg_price"] > 0)]

    if MIN_PAX > 0:
        df = df[df["total_passengers"] >= MIN_PAX]

    df["origin"] = df["origin_state"].astype(str)
    df["destination"] = df["city2"].astype(str)
    df["route_id"] = df["origin"] + " -> " + df["destination"]

    df = df.sort_values(["route_id", "Year"])
    df["price_lag1"] = df.groupby("route_id")["avg_price"].shift(1)
    df = df.dropna(subset=["price_lag1"])

    df["ln_pax"] = np.log(df["total_passengers"])
    df["ln_price_lag1"] = np.log(df["price_lag1"])
    df["price_lag1_scaled"] = df["price_lag1"] / SCALE_PRICE

    year_str = df["Year"].astype(int).astype(str)
    df["origin_year"] = df["origin"] + "_" + year_str
    df["dest_year"] = df["destination"] + "_" + year_str

    return df


# ============================================================
# 3) Estimation blocks (no printing here; stdout will be muted anyway)
# ============================================================
def run_ols_log(df: pd.DataFrame):
    # ln_pax_t ~ ln_price_{t-1}
    return smf.ols("ln_pax ~ ln_price_lag1", data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["route_id"], "use_correction": True}
    )


def run_ols_levels(df: pd.DataFrame):
    # pax_t ~ price_{t-1}/scale
    return smf.ols("total_passengers ~ price_lag1_scaled", data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["route_id"], "use_correction": True}
    )


def run_fe_panel(df: pd.DataFrame):
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return None

    panel = df.set_index(["route_id", "Year"]).sort_index()
    Y = panel["ln_pax"]
    X = sm.add_constant(panel[["ln_price_lag1"]])

    mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res


def run_hdfe_log(df: pd.DataFrame):
    try:
        from linearmodels.iv import AbsorbingLS
    except ImportError:
        return None

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

    clusters = df["route_id"]
    return AbsorbingLS(y, X, absorb=absorb).fit(cov_type="clustered", clusters=clusters)


def run_hdfe_levels(df: pd.DataFrame):
    try:
        from linearmodels.iv import AbsorbingLS
    except ImportError:
        return None

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

    clusters = df["route_id"]
    return AbsorbingLS(y, X, absorb=absorb).fit(cov_type="clustered", clusters=clusters)


# ============================================================
# 4) Plots (SAVED always; SHOW only if show=True)
# ============================================================
def plot_loglog_relationship(df: pd.DataFrame, label: str, show: bool = False):
    """
    Main plot: passengers_t vs price_{t-1}, log-log axes.
    Labels ON the plot (legend): points vs trend line + slope.
    """
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    price = df["price_lag1"].to_numpy(dtype=float)
    pax = df["total_passengers"].to_numpy(dtype=float)

    m = np.isfinite(price) & np.isfinite(pax) & (price > 0) & (pax > 0)
    price, pax = price[m], pax[m]
    if price.size < 2:
        return

    # log-log slope for the trend line
    b, a = np.polyfit(np.log(price), np.log(pax), 1)
    x_line = np.logspace(np.log10(price.min()), np.log10(price.max()), 200)
    y_line = np.exp(a) * x_line**b

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    point_label = "City → city (price lagged by 1y)" if "cities" in label.lower() else "State → city (price lagged by 1y)"
    if MIN_PAX > 0:
        point_label += f" (≥ {MIN_PAX} pax/year)"

    ax.scatter(price, pax, s=10, alpha=0.45, label=point_label)
    ax.plot(x_line, y_line, linewidth=2.2, label=f"Log-log trend (slope ≈ {b:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Keep it clean (like your “other outcomes”)
    ax.set_title("Passengers vs. lagged price (1996–2025)")
    ax.set_xlabel("Average ticket price, t-1 (USD)")
    ax.set_ylabel("Total passengers, t (per year)")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, f"loglog_pax_vs_pricelag_{safe}.png")
    fig.savefig(outpath, dpi=300)

    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_levels_relationship(df: pd.DataFrame, label: str, show: bool = False):
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    x = df["price_lag1_scaled"].to_numpy(dtype=float)
    y = df["total_passengers"].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return

    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.scatter(x, y, s=10, alpha=0.25, label="Routes")
    ax.plot(grid, intercept + slope * grid, linewidth=2.2,
            label=f"Trend line (slope ≈ {slope:.1f})")

    ax.set_title("Passengers vs. lagged price (levels)")
    ax.set_xlabel(f"price_(t-1) / {int(SCALE_PRICE)}")
    ax.set_ylabel("passengers_t")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, f"levels_pax_vs_pricelag_{safe}.png")
    fig.savefig(outpath, dpi=300)

    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_example_route_timeseries(df: pd.DataFrame, label: str, show: bool = False):
    _ensure_fig_dir()
    safe = _sanitize_label(label)

    if df.empty:
        return
    top_route = df["route_id"].value_counts().index[0]
    sub = df[df["route_id"] == top_route].sort_values("Year")
    if sub["Year"].nunique() < 3:
        return

    years = sub["Year"].to_numpy()
    pax = sub["total_passengers"].to_numpy()
    price = sub["avg_price"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=160)
    ax1.plot(years, pax, marker="o", linewidth=2.0)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Passengers")

    ax2 = ax1.twinx()
    ax2.plot(years, price, marker="s", linestyle="--", linewidth=2.0)
    ax2.set_ylabel("Average price")

    fig.suptitle(f"Example route time series: {top_route}")
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, f"timeseries_example_route_{safe}.png")
    fig.savefig(outpath, dpi=300)

    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_all_diagnostics(df: pd.DataFrame, label: str, show_plots: bool = False, show_only_main: bool = False):
    if df.empty:
        return

    # Always SAVE all diagnostics (unchanged), but control SHOW separately
    plot_loglog_relationship(df, label, show=show_plots)

    if not show_only_main:
        plot_levels_relationship(df, label, show=show_plots)
        plot_example_route_timeseries(df, label, show=show_plots)
    else:
        # still save these two (unchanged) but do not show
        plot_levels_relationship(df, label, show=False)
        plot_example_route_timeseries(df, label, show=False)


# ============================================================
# 5) Runner for a panel (models run + figures saved)
# ============================================================
def run_all_for_panel(df: pd.DataFrame, label: str, show_plots: bool = False, show_only_main: bool = False):
    if df is None or len(df) == 0:
        return

    # Run models (no terminal output; you can still use returned objects if needed)
    _ = run_ols_log(df)
    _ = run_ols_levels(df)
    _ = run_fe_panel(df)
    _ = run_hdfe_log(df)
    _ = run_hdfe_levels(df)

    # Figures (always saved); shown only depending on flags
    plot_all_diagnostics(df, label, show_plots=show_plots, show_only_main=show_only_main)


# ============================================================
# MAIN
# ============================================================
def main():
    _ensure_fig_dir()

    # Suppress ALL stdout prints (but still show errors on stderr)
    with _quiet_stdout(QUIET_TERMINAL):

        # ---- CITY panel: SHOW (only city), but only the main plot window ----
        if os.path.exists(MERGED_CSV):
            city_panel = prepare_city_panel(MERGED_CSV)
            run_all_for_panel(
                city_panel,
                "cities (city1 -> city2, annual)",
                show_plots=SHOW_CITY_PLOTS,
                show_only_main=SHOW_ONLY_CITY_MAIN
            )

        # ---- STATE panel: DO NOT SHOW, but still run + SAVE all outputs ----
        if os.path.exists(MERGED_CSV):
            state_panel = prepare_state_panel(MERGED_CSV)
            run_all_for_panel(
                state_panel,
                "states (origin_state -> city2, annual)",
                show_plots=SHOW_STATE_PLOTS,
                show_only_main=False
            )


if __name__ == "__main__":
    main()
