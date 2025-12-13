import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ============================
# CONFIG
# ============================

DATA_CSV = "data/merged_air_travel_data.csv"

# ✅ dossier de sortie
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_spec_curve")  # optionnel, pour ranger les figures


# ============================
# HELPERS
# ============================

def first_exact(df, candidates, required=False):
    """Cherche la première colonne existante parmi une liste."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Colonnes candidates absentes: {candidates}")
    return None


def ensure_dirs():
    """Crée outputs/ (et outputs/figures_spec_curve/) si nécessaire."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


# ============================
# LOADER ROBUSTE
# ============================

def load_city_panel_annual(path=DATA_CSV):
    """
    Charge le fichier merged_air_travel_data.csv et 
    construit un panel annuel (origin, destination, Year)
    avec ln_pax, ln_price_lag1, etc.
    """
    df = pd.read_csv(path)

    # identification automatique des colonnes
    year_col   = first_exact(df, ["Year", "YEAR", "year"], required=True)
    origin_col = first_exact(df, ["city1", "origin", "Origin"], required=True)
    dest_col   = first_exact(df, ["city2", "destination", "Destination"], required=True)

    pax_col = first_exact(
        df,
        ["passengers", "Passengers", "passenger_count", "PAX"],
        required=True
    )

    price_col = first_exact(
        df,
        ["Real price", "real_price", "avg_price", "fare", "Fare", "price"],
        required=True
    )

    print("\n[INFO] Colonnes détectées :")
    print(f"  Year         -> {year_col}")
    print(f"  origin       -> {origin_col}")
    print(f"  destination  -> {dest_col}")
    print(f"  passengers   -> {pax_col}")
    print(f"  price        -> {price_col}")

    # standardisation des id
    df["Year"] = df[year_col].astype(int)
    df["origin"] = df[origin_col].astype(str)
    df["destination"] = df[dest_col].astype(str)

    # conversion en numérique des variables de niveau
    df[pax_col] = pd.to_numeric(df[pax_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # on enlève les lignes sans info de prix ou de pax
    df = df.dropna(subset=[pax_col, price_col]).copy()

    # agrégation annuelle
    annual = (
        df.groupby(["origin", "destination", "Year"], as_index=False)
          .agg(
              total_passengers=(pax_col, "sum"),
              mean_price=(price_col, "mean")
          )
    )

    # logs (on est maintenant sûr que ce sont des floats)
    annual["ln_pax"] = np.log(annual["total_passengers"].astype(float) + 1e-8)
    annual["ln_price"] = np.log(annual["mean_price"].astype(float) + 1e-8)

    # lag du prix
    annual = annual.sort_values(["origin", "destination", "Year"])
    annual["ln_price_lag1"] = (
        annual.groupby(["origin", "destination"])["ln_price"].shift(1)
    )

    # on enlève les premières années sans lag
    annual = annual.dropna(subset=["ln_price_lag1"]).copy()

    print(f"[INFO] Panel annuel prêt : {len(annual)} observations.\n")
    return annual


# ============================
# SPECIFICATIONS À TESTER
# ============================

def build_specifications():
    """
    Chaque spécification ajoute un bloc de fixed effects pour illustrer
    ce que capturent les FE dans ton DAG : crises, météo, attractivité, etc.
    """
    specs = []

    specs.append({
        "name": "OLS simple",
        "formula": "ln_pax ~ ln_price_lag1"
    })

    specs.append({
        "name": "+ FE route (o,d)",
        "formula": "ln_pax ~ ln_price_lag1 + C(origin) + C(destination)"
    })

    specs.append({
        "name": "+ FE année",
        "formula": "ln_pax ~ ln_price_lag1 + C(origin) + C(destination) + C(Year)"
    })

    specs.append({
        "name": "+ FE origin×year",
        "formula": "ln_pax ~ ln_price_lag1 + C(origin)*C(Year) + C(destination)"
    })

    specs.append({
        "name": "+ FE destination×year",
        "formula": "ln_pax ~ ln_price_lag1 + C(destination)*C(Year) + C(origin)"
    })

    return specs


# ============================
# ESTIMATION DE CHAQUE MODÈLE
# ============================

def estimate_specifications(df, specs):
    results = []

    for i, spec in enumerate(specs, start=1):
        print(f"[INFO] Estimation {i}/{len(specs)} : {spec['name']}")
        model = smf.ols(spec["formula"], data=df).fit(cov_type="HC1")

        beta = model.params["ln_price_lag1"]
        ci_low, ci_high = model.conf_int().loc["ln_price_lag1"]

        results.append({
            "name": spec["name"],
            "beta": beta,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

    return pd.DataFrame(results)


# ============================
# PLOT ET SAUVEGARDE
# ============================

def plot_spec_curve(results, output_path=None):
    """
    Graphe type "erreur bars" montrant l'effet de chaque bloc de FE
    sur l'élasticité prix(t-1) -> passagers(t).
    """
    ensure_dirs()

    # ✅ chemin par défaut dans outputs/figures_spec_curve/
    if output_path is None:
        output_path = os.path.join(FIG_DIR, "spec_curve_elasticity.png")
    else:
        # si on te passe juste un nom de fichier, on le met dans FIG_DIR
        if not os.path.dirname(output_path):
            output_path = os.path.join(FIG_DIR, output_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(results))
    betas = results["beta"].values
    lo = results["ci_low"].values
    hi = results["ci_high"].values

    for i in range(len(results)):
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color="black", linewidth=2)
        ax.scatter(x[i], betas[i], s=60, zorder=3)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(results["name"], rotation=30, ha="right")
    ax.set_ylabel("Elasticity (log-log)")
    ax.set_title("Effect of Fixed-Effects Blocks on Lagged Price Elasticity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Plot sauvegardé : {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    ensure_dirs()

    df = load_city_panel_annual()
    specs = build_specifications()
    results = estimate_specifications(df, specs)

    # (optionnel) sauvegarder le tableau de résultats aussi
    results_csv = os.path.join(OUT_DIR, "spec_curve_results.csv")
    results.to_csv(results_csv, index=False)
    print(f"[INFO] Results saved: {results_csv}")

    plot_spec_curve(results, output_path="spec_curve_elasticity.png")
