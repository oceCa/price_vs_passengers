#!/usr/bin/env python3
# simple_regression.py
# -----------------------------------------------------------
# Un seul script pour :
#  - générer un DAG (Price -> Passengers)
#  - faire une régression simple niveau states -> cities
#  - faire une régression simple niveau routes (city1 -> city2)
# en utilisant uniquement merged_air_travel_data.csv
# -----------------------------------------------------------

import os
import shutil
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

        # ✅ save to outputs folder (graphviz adds .png)
        outbase = os.path.join(FIG_DIR, "dag_price_demand")
        outpath = g.render(filename=outbase, cleanup=True)
        print(f"[DAG] saved → {outpath}")
        return

    except Exception as err:
        print(f"[DAG] graphviz path failed ({err}). Falling back to networkx…")
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

            # ✅ save to outputs folder
            outpng = os.path.join(FIG_DIR, "dag_price_demand_fallback.png")
            plt.savefig(outpng, dpi=200)
            plt.close()
            print(f"[DAG] saved → {outpng}")

        except Exception as err2:
            print(f"[DAG] fallback also failed: {err2}")


# =========================
# Helper: load & clean merged_air_travel_data.csv
# =========================
def load_and_clean_merged(path=MERGED_CSV) -> pd.DataFrame:
    """
    Charge merged_air_travel_data.csv, nettoie les colonnes clés
    et restreint la fenêtre temporelle 1996–2025.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable dans le dossier courant.")

    print(f"[DATA] Loading {path} ...")
    df = pd.read_csv(path)

    if "passengers" not in df.columns:
        raise KeyError("Colonne 'passengers' absente du CSV.")
    df["passengers"] = (
        df["passengers"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
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

    print(f"[DATA] Cleaned merged_air_travel_data: {len(df)} rows.")
    return df


# =========================
# Helper: build state-level panel
# =========================
def build_state_city_year_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort un DataFrame avec colonnes:
        Year, origin_state, city2, total_passengers, avg_price
    """
    if "city1" not in df.columns:
        raise KeyError("Column 'city1' not found in DataFrame.")
    if "city2" not in df.columns:
        raise KeyError("Column 'city2' not found in DataFrame.")

    df = df.copy()
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

    print(f"[DATA] Aggregated panel (state → city → year) ready: {len(state_city_year)} observations.")
    return state_city_year


# =========================
# Plot 1: State-level simple regression
# =========================
def plot_states_simple(df_state_city_year: pd.DataFrame):
    ensure_dirs()
    df = df_state_city_year.copy()

    df["total_passengers"] = pd.to_numeric(df["total_passengers"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["total_passengers", "avg_price"])

    df = df[df["total_passengers"] >= 5000]
    if df.empty:
        raise ValueError("Aucune observation state→city ≥ 5000 passagers après nettoyage.")

    df["route_id"] = df["origin_state"].astype(str) + " -> " + df["city2"].astype(str)
    cluster_var = df["route_id"]
    print("\n[STATES] Running OLS (levels) with clustered SE (groups = route_id)")

    X = add_constant(df["avg_price"])
    y = df["total_passengers"]

    model = sm.OLS(y, X, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )

    print("\n=== [STATES] OLS levels (total_passengers ~ avg_price) ===")
    print(model.summary())

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
    plt.show()
    print(f"[PLOT] saved → {outpng}")


# =========================
# Plot 2: City-level simple regression
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
    grouped = grouped[grouped["total_passengers"] >= 5000]
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_price", "total_passengers"])

    if grouped.empty:
        raise ValueError("Aucune route ≥ 5000 passagers pour 1996–2025 après nettoyage.")

    grouped["route_id"] = grouped["city1"].astype(str) + " -> " + grouped["city2"].astype(str)
    cluster_var = grouped["route_id"]
    print("\n[CITIES] Running OLS (levels) with clustered SE (groups = route_id)")

    X = add_constant(grouped["avg_price"])
    y = grouped["total_passengers"]

    model = sm.OLS(y, X, missing="drop").fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var, "use_correction": True}
    )

    print("\n=== [CITIES] OLS levels (total_passengers ~ avg_price) ===")
    print(model.summary())

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
    plt.show()
    print(f"[PLOT] saved → {outpng}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dirs()

    print("[DAG] generating…")
    make_dag_price_demand()
    print("[DAG] done.")

    df_merged = load_and_clean_merged(MERGED_CSV)

    df_state_city_year = build_state_city_year_from_df(df_merged)
    plot_states_simple(df_state_city_year)

    plot_cities_simple(df_merged)
