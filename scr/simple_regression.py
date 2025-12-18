#!/usr/bin/env python3
# simple_regression.py
# -----------------------------------------------------------
# Un seul script pour :
#  - générer un DAG (Price -> Passengers)
#  - faire une régression simple niveau states -> cities
#  - faire une régression simple niveau routes (city1 -> city2)
# en utilisant uniquement merged_air_travel_data.csv
#
# OUTPUT POLICY:
#  - Terminal: no prints
#  - Display: ONLY the city->city plot (plt.show only there)
#  - Files: DAG + states plot + cities plot are still saved
# -----------------------------------------------------------

import os
import shutil
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant


# ✅ Fix path + centralize outputs
MERGED_CSV = "data/merged_air_travel_data.csv"

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_simple_reg")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


@contextlib.contextmanager
def silence_stdout():
    """
    Mute prints coming from our code (and some libraries) for a block.
    Keeps errors/exceptions visible.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# =========================
# DAG (Price -> Passengers)
# =========================
def make_dag_price_demand():
    """
    Render a DAG with graphviz if available (and 'dot' exists),
    else fall back to a simple networkx plot.
    Saved into outputs/figures_simple_reg/
    """
    ensure_dirs()

    try:
        import graphviz

        if shutil.which("dot") is None:
            raise RuntimeError("Graphviz 'dot' executable not found")

        g = graphviz.Digraph("dag_price_demand", format="png")
        nodes = ["Price", "Passengers", "Distance", "Competition",
                 "MarketSize", "Season", "FuelCost", "AirportFees"]
        for n in nodes:
            g.node(n)
        for z in ["Distance", "Competition", "MarketSize",
                  "Season", "FuelCost", "AirportFees"]:
            g.edge(z, "Price")
            g.edge(z, "Passengers")
        g.edge("Price", "Passengers")

        outbase = os.path.join(FIG_DIR, "dag_price_demand")
        g.render(filename=outbase, cleanup=True)
        return

    except Exception:
        # fallback networkx
        try:
            import networkx as nx

            G = nx.DiGraph()
            Z = ["Distance", "Competition", "MarketSize",
                 "Season", "FuelCost", "AirportFees"]
            G.add_nodes_from(Z + ["Price", "Passengers"])
            for z in Z:
                G.add_edge(z, "Price")
                G.add_edge(z, "Passengers")
            G.add_edge("Price", "Passengers")

            plt.figure(figsize=(7, 5))
            pos = nx.spring_layout(G, seed=1)
            nx.draw_networkx(G, pos, arrows=True, with_labels=True, node_size=1200)
            plt.axis("off")
            plt.tight_layout()

            outpng = os.path.join(FIG_DIR, "dag_price_demand_fallback.png")
            plt.savefig(outpng, dpi=200)
            plt.close()

        except Exception:
            # if even fallback fails, just skip silently
            return


# =========================
# Helper: load & clean merged_air_travel_data.csv
# =========================
def load_and_clean_merged(path=MERGED_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable dans le dossier courant.")

    df = pd.read_csv(path)

    if "passengers" not in df.columns:
        raise KeyError("Colonne 'passengers' absente du CSV.")
    df["passengers"] = df["passengers"].astype(str).str.replace(",", "", regex=False)
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")

    if "Real price" not in df.columns:
        raise KeyError("Colonne 'Real price' absente du CSV.")
    df["Real price"] = pd.to_numeric(df["Real price"], errors="coerce")

    for col in ["city1", "city2", "Year"]:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' absente du CSV.")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df = df.dropna(subset=["passengers", "Real price", "city1", "city2", "Year"])
    df = df[(df["passengers"] > 0) & (df["Real price"] > 0)]
    df = df[(df["Year"] >= 1996) & (df["Year"] <= 2025)]
    return df


# =========================
# Helper: build state-level panel
# =========================
def build_state_city_year_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


# =========================
# Plot 1: State-level simple regression (SAVED ONLY, NO SHOW)
# =========================
def plot_states_simple(df_state_city_year: pd.DataFrame):
    ensure_dirs()
    df = df_state_city_year.copy()

    df["total_passengers"] = pd.to_numeric(df["total_passengers"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["total_passengers", "avg_price"])

    # Keep your original rule
    df = df[df["total_passengers"] >= 5000]
    if df.empty:
        return  # silently skip

    df["route_id"] = df["origin_state"].astype(str) + " -> " + df["city2"].astype(str)

    X = add_constant(df["avg_price"])
    y = df["total_passengers"]

    model = sm.OLS(y, X, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": df["route_id"], "use_correction": True}
    )

    a = model.params.get("avg_price", np.nan)
    b = model.params.get("const", np.nan)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["avg_price"], df["total_passengers"], s=8, alpha=0.5,
                label="State → city (≥ 5000 passengers/year)")

    x = df["avg_price"].values
    if np.isfinite(a) and np.isfinite(b) and x.size >= 2:
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = a * x_line + b
        plt.plot(x_line, y_line, linewidth=2,
                 label=f"OLS trend (clustered by route): passengers = {a:.4e} * price + {b:.2f}")

    plt.xlabel("Average ticket price (USD, passenger-weighted)")
    plt.ylabel("Total passengers from state to city (per year)")
    plt.title("State-level routes: passengers vs. price (1996–2025)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpng = os.path.join(FIG_DIR, "routes_states_price_x.png")
    plt.savefig(outpng, dpi=300)
    plt.close()  # ✅ IMPORTANT: no plt.show()


# =========================
# Plot 2: City-level simple regression (SAVED + SHOW)
# =========================
def plot_cities_simple(df_merged: pd.DataFrame):
    ensure_dirs()
    df = df_merged.copy()

    grouped = (
        df.groupby(["Year", "city1", "city2"], as_index=False)
          .apply(lambda g: pd.Series({
              "total_passengers": g["passengers"].sum(),
              "avg_price": (g["Real price"] * g["passengers"]).sum() / g["passengers"].sum()
          }))
          .reset_index(drop=True)
    )

    grouped["total_passengers"] = pd.to_numeric(grouped["total_passengers"], errors="coerce")
    grouped["avg_price"] = pd.to_numeric(grouped["avg_price"], errors="coerce")
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_price", "total_passengers"])

    # Keep your original rule (>=5000 per year obs)
    grouped = grouped[grouped["total_passengers"] >= 5000]
    if grouped.empty:
        raise ValueError("Aucune route ≥ 5000 passagers pour 1996–2025 après nettoyage.")

    grouped["route_id"] = grouped["city1"].astype(str) + " -> " + grouped["city2"].astype(str)

    X = add_constant(grouped["avg_price"])
    y = grouped["total_passengers"]

    model = sm.OLS(y, X, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": grouped["route_id"], "use_correction": True}
    )

    a = model.params.get("avg_price", np.nan)
    b = model.params.get("const", np.nan)

    plt.figure(figsize=(10, 6))
    plt.scatter(grouped["avg_price"], grouped["total_passengers"], s=8, alpha=0.5,
                label="Routes (≥ 5000 passengers/year)")

    x = grouped["avg_price"].to_numpy()
    if np.isfinite(a) and np.isfinite(b) and x.size >= 2:
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = a * x_line + b
        plt.plot(x_line, y_line, linewidth=2,
                 label=f"OLS trend (clustered by route): passengers = {a:.4f} * price + {b:.2f}")

    plt.xlabel("Average ticket price (USD, passenger-weighted)")
    plt.ylabel("Total passengers on route (per year)")
    plt.title("Air routes 1996–2025: passengers vs. price (≥ 5000 passengers/year)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpng = os.path.join(FIG_DIR, "routes_cities_price_x.png")
    plt.savefig(outpng, dpi=300)

    plt.show()   # ✅ ONLY thing that pops up
    plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dirs()

    # Run everything but keep terminal silent
    with silence_stdout():
        make_dag_price_demand()
        df_merged = load_and_clean_merged(MERGED_CSV)
        df_state_city_year = build_state_city_year_from_df(df_merged)
        plot_states_simple(df_state_city_year)

    # Only city plot is shown, no extra terminal output
    df_merged = load_and_clean_merged(MERGED_CSV)
    plot_cities_simple(df_merged)
