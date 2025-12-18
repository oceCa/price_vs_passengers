#!/usr/bin/env python3
# endogeneity_from_merged.py
# ============================================================
# Analyse d'endogénéité du prix pour la demande de transport aérien,
# inspirée de Mumbower (2014) et Miller & Alberini (2016).
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ---------- CONFIG GÉNÉRALE ----------

MERGED_CSV = "data/merged_air_travel_data.csv"   # base unique
MIN_PAX = 0                                      # seuil optionnel pour filtrer les flux faibles (0 = pas de filtre)

#  Everything goes under outputs/
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures_endogeneity")

#  Show figures when running (Notebook / VSCode / Terminal)
SHOW_FIGS = True

# Pour les IV : noms de colonnes candidates (si elles existent dans ton CSV)
INSTRUMENT_CANDIDATES = [
    "jetfuel_price", "kerosene_price", "oil_price", "fuel_price",
    "competitors", "num_carriers", "lowcost_share", "lowcost_dummy"
]

CONTROL_CANDIDATES = [
    "distance_km", "distance", "log_distance",
    "hub_dummy", "is_hub",
    "tourism_dummy", "income_origin", "income_destination"
]


# ---------- HELPERS ----------

def info(msg: str) -> None:
    print(f"\n[INFO] {msg}")


def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def ensure_dirs():
    """Create outputs/ and outputs/figures_endogeneity/ if needed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def first_exact(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sanitize_for_filename(s: str) -> str:
    return (
        str(s)
        .lower()
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


def maybe_show(fig, title: str | None = None):
    """
    Show figure in Notebook output (and also in normal python runs).
    Controlled via SHOW_FIGS.
    """
    if not SHOW_FIGS:
        return
    if title:
        try:
            fig.suptitle(title)
        except Exception:
            pass
    plt.show()


# ---------- PRÉPARATION DES DONNÉES : PANEL VILLES ----------

def prepare_city_panel(path: str) -> pd.DataFrame:
    """
    Charge merged_air_travel_data.csv, agrège à l'année au niveau route (origin -> destination),
    et construit un panel avec :
      - total_passengers, avg_price
      - log_passengers, log_price
      - route_id, Year
    """
    info(f"Chargement du panel measuring cities depuis {path} ...")
    df = pd.read_csv(path)

    origin_col = first_exact(df, ["city1", "origin", "origin_city", "citymarketid_1"])
    dest_col   = first_exact(df, ["city2", "destination", "dest_city", "citymarketid_2"])
    year_col   = first_exact(df, ["Year", "year"])

    if origin_col is None or dest_col is None or year_col is None:
        raise KeyError("Colonnes city/origin/destination/Year introuvables dans merged_air_travel_data.csv")

    pax_col   = first_exact(df, ["passengers", "total_passengers"])
    price_col = first_exact(df, ["Real price", "price", "avg_price"])
    if pax_col is None or price_col is None:
        raise KeyError("Colonnes passagers/prix introuvables dans merged_air_travel_data.csv")

    df[pax_col] = to_numeric(df[pax_col])
    df[price_col] = to_numeric(df[price_col])
    df = df.dropna(subset=[year_col, origin_col, dest_col, pax_col, price_col])

    info("Agrégation annuelle (origin, destination, Year) ...")
    grp_cols = [year_col, origin_col, dest_col]
    ann = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              total_passengers=(pax_col, "sum"),
              avg_price=(price_col, "mean")
          )
    )

    ann = ann[(ann["total_passengers"] > 0) & (ann["avg_price"] > 0)]

    if MIN_PAX > 0:
        ann = ann[ann["total_passengers"] >= MIN_PAX]
        info(f"Filtre flux avec total_passengers >= {MIN_PAX} -> {len(ann)} lignes restantes")

    ann = ann.rename(columns={year_col: "Year", origin_col: "origin", dest_col: "destination"})
    ann["route_id"] = ann["origin"].astype(str) + " -> " + ann["destination"].astype(str)

    ann["log_passengers"] = np.log(ann["total_passengers"])
    ann["log_price"] = np.log(ann["avg_price"])

    info(f"Panel villes prêt : {len(ann)} observations.")
    return ann


# ---------- CONSTRUCTION PANEL ÉTATS -> VILLES DEPUIS MERGED ----------

def build_state_city_year_from_merged(path: str) -> pd.DataFrame:
    """
    Recrée en mémoire l'équivalent de states_departures.csv à partir de merged_air_travel_data.csv.
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
    """
    Construit le panel états -> villes à partir de merged_air_travel_data.csv
    """
    info(f"Chargement du panel états (reconstruit depuis {path}) ...")
    df = build_state_city_year_from_merged(path)

    year_col   = first_exact(df, ["Year", "year"])
    origin_col = first_exact(df, ["origin_state", "origin", "state", "state_origin"])
    dest_col   = first_exact(df, ["city2", "destination", "city_dest"])
    pax_col    = first_exact(df, ["total_passengers", "passengers"])
    price_col  = first_exact(df, ["avg_price", "price"])

    if year_col is None or origin_col is None or dest_col is None or pax_col is None or price_col is None:
        raise KeyError("Colonnes Year/origin_state/city2/total_passengers/avg_price manquantes dans panel états.")

    df[pax_col] = to_numeric(df[pax_col])
    df[price_col] = to_numeric(df[price_col])
    df = df.dropna(subset=[year_col, origin_col, dest_col, pax_col, price_col])
    df = df[(df[pax_col] > 0) & (df[price_col] > 0)]

    if MIN_PAX > 0:
        df = df[df[pax_col] >= MIN_PAX]
        info(f"Filtre flux avec total_passengers >= {MIN_PAX} -> {len(df)} lignes restantes")

    df = df.rename(columns={
        year_col: "Year",
        origin_col: "origin_state",
        dest_col: "destination",
        pax_col: "total_passengers",
        price_col: "avg_price"
    })

    df["route_id"] = df["origin_state"].astype(str) + " -> " + df["destination"].astype(str)
    df["log_passengers"] = np.log(df["total_passengers"])
    df["log_price"] = np.log(df["avg_price"])

    info(f"Panel états prêt : {len(df)} observations.")
    return df


# ---------- DESCRIPTIVES & PLOTS À LA MILLER & ALBERINI ----------

def descriptives_states(df: pd.DataFrame):
    """
    Descriptives par état-destination et par état seul.
    Sauvegarde en CSV dans OUT_DIR.
    """
    info("Descriptives par état-destination ...")
    desc_state = (
        df.groupby(["origin_state", "destination"])
          [["total_passengers", "avg_price"]]
          .describe(percentiles=[0.25, 0.5, 0.75])
    )
    desc_state.to_csv(os.path.join(OUT_DIR, "descriptives_states_origin_destination.csv"))

    info("Descriptives par état (agrégées sur toutes les destinations) ...")
    desc_state_only = (
        df.groupby("origin_state")
          [["total_passengers", "avg_price"]]
          .agg(["mean", "median", "std", "min", "max"])
    )
    desc_state_only.to_csv(os.path.join(OUT_DIR, "descriptives_states_only.csv"))


def plot_mean_price_vs_pax_by_state(df: pd.DataFrame):
    """
    Scatter de la moyenne du prix vs moyenne des passagers par état.
    Sauvegarde une figure PNG ET l'affiche (Notebook friendly).
    """
    ensure_dirs()
    info("Plot mean price vs mean passengers par état ...")

    state_mean = (
        df.groupby("origin_state")
          .agg(
              avg_pax=("total_passengers", "mean"),
              avg_price=("avg_price", "mean")
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(state_mean["avg_pax"], state_mean["avg_price"])

    for _, row in state_mean.iterrows():
        ax.text(row["avg_pax"], row["avg_price"], row["origin_state"], fontsize=6)

    ax.set_xlabel("Mean passengers per origin_state")
    ax.set_ylabel("Mean price per origin_state")
    ax.set_title("Mean price vs mean passengers (by origin_state)")
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "mean_price_vs_pax_by_state.png")
    fig.savefig(outpath, dpi=300)
    info(f"Figure sauvegardée : {outpath}")

    # ✅ show in notebook / interactive runs
    maybe_show(fig)

    # close after showing (avoid memory leaks)
    plt.close(fig)


# ---------- ELASTICITÉS À DIFFÉRENTS NIVEAUX ----------

def elasticity_simple(df: pd.DataFrame, label: str, y="log_passengers", x="log_price"):
    info(f"Élasticité simple pour {label}: {y} ~ {x}")
    if "route_id" in df.columns:
        model = smf.ols(f"{y} ~ {x}", data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["route_id"], "use_correction": True}
        )
    else:
        model = smf.ols(f"{y} ~ {x}", data=df).fit(cov_type="HC1")
    print(model.summary())
    return model


def elasticity_national_from_states(df_states: pd.DataFrame):
    info("Élasticité au niveau national (agrégation par année à partir des états) ...")

    nat = (
        df_states.groupby("Year", as_index=False)
                 .agg(
                     total_passengers=("total_passengers", "sum"),
                     avg_price=("avg_price", "mean")
                 )
    )
    nat["log_passengers"] = np.log(nat["total_passengers"])
    nat["log_price"] = np.log(nat["avg_price"])

    model = smf.ols("log_passengers ~ log_price", data=nat).fit(cov_type="HC1")
    print(model.summary())

    nat.to_csv(os.path.join(OUT_DIR, "national_series_from_states.csv"), index=False)
    return model, nat


# ---------- CROSS-SECTION vs POOLED vs FE ----------

def cross_section_pooled_fe(df_states: pd.DataFrame):
    results = {}

    last_year = df_states["Year"].max()
    df_last = df_states[df_states["Year"] == last_year].copy()
    info(f"Cross-section sur la dernière année : {last_year}")

    if "route_id" in df_last.columns:
        mod_cs = smf.ols("log_passengers ~ log_price", data=df_last).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_last["route_id"], "use_correction": True}
        )
    else:
        mod_cs = smf.ols("log_passengers ~ log_price", data=df_last).fit(cov_type="HC1")
    print(mod_cs.summary())
    results["beta_cross_section"] = mod_cs.params.get("log_price", np.nan)

    info("Pooled panel (toutes années, sans FE) ...")
    if "route_id" in df_states.columns:
        mod_pool = smf.ols("log_passengers ~ log_price", data=df_states).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_states["route_id"], "use_correction": True}
        )
    else:
        mod_pool = smf.ols("log_passengers ~ log_price", data=df_states).fit(cov_type="HC1")
    print(mod_pool.summary())
    results["beta_pooled"] = mod_pool.params.get("log_price", np.nan)

    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        info("linearmodels.panel non installé -> FE route+year sauté.")
        results["beta_fe_route_year"] = np.nan
        return results

    info("Panel FE (route_id + Year) ...")
    panel = df_states.set_index(["route_id", "Year"]).sort_index()
    Y = panel["log_passengers"]
    X = sm.add_constant(panel[["log_price"]])

    fe_mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)
    print(fe_res.summary)

    results["beta_fe_route_year"] = fe_res.params.get("log_price", np.nan)

    pd.DataFrame([results]).to_csv(
        os.path.join(OUT_DIR, "cross_section_pooled_fe_states.csv"), index=False
    )

    return results


# ---------- HÉTÉROGÉNÉITÉ PAR DISTANCE (SI DISPONIBLE) ----------

def heterogeneity_by_distance(df: pd.DataFrame, label: str):
    dist_col = first_exact(df, ["distance_km", "distance"])
    if dist_col is None:
        info(f"Aucune colonne distance trouvée pour {label} -> hétérogénéité par distance sautée.")
        return None

    info(f"Hétérogénéité par distance pour {label}, en utilisant {dist_col} ...")
    df = df.copy()
    df["long_haul"] = (df[dist_col] > 1500).astype(int)

    if "route_id" in df.columns:
        mod_int = smf.ols("log_passengers ~ log_price * long_haul", data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["route_id"], "use_correction": True}
        )
    else:
        mod_int = smf.ols("log_passengers ~ log_price * long_haul", data=df).fit(cov_type="HC1")
    print(mod_int.summary())

    short = df[df["long_haul"] == 0]
    long = df[df["long_haul"] == 1]

    if "route_id" in short.columns and len(short) > 0:
        mod_short = smf.ols("log_passengers ~ log_price", data=short).fit(
            cov_type="cluster",
            cov_kwds={"groups": short["route_id"], "use_correction": True}
        )
    else:
        mod_short = smf.ols("log_passengers ~ log_price", data=short).fit(cov_type="HC1")

    if "route_id" in long.columns and len(long) > 0:
        mod_long = smf.ols("log_passengers ~ log_price", data=long).fit(
            cov_type="cluster",
            cov_kwds={"groups": long["route_id"], "use_correction": True}
        )
    else:
        mod_long = smf.ols("log_passengers ~ log_price", data=long).fit(cov_type="HC1")

    info(f"{label} - Short-haul: beta = {mod_short.params.get('log_price', np.nan):.3f}")
    info(f"{label} - Long-haul : beta = {mod_long.params.get('log_price', np.nan):.3f}")

    out = {
        "beta_int_log_price": mod_int.params.get("log_price", np.nan),
        "beta_int_interaction": mod_int.params.get("log_price:long_haul", np.nan),
        "beta_short": mod_short.params.get("log_price", np.nan),
        "beta_long": mod_long.params.get("log_price", np.nan)
    }

    pd.DataFrame([out]).to_csv(
        os.path.join(OUT_DIR, f"heterogeneity_distance_{sanitize_for_filename(label)}.csv"),
        index=False
    )

    return out


# ---------- TABLEAU DE ROBUSTESSE (MULTIPLES SPÉCS) ----------

def robustness_table_states(df_states: pd.DataFrame):
    results = []

    if "route_id" in df_states.columns:
        mA = smf.ols("log_passengers ~ log_price", data=df_states).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_states["route_id"], "use_correction": True}
        )
    else:
        mA = smf.ols("log_passengers ~ log_price", data=df_states).fit(cov_type="HC1")

    results.append({
        "spec": "A_simple",
        "beta": mA.params.get("log_price", np.nan),
        "se": mA.bse.get("log_price", np.nan),
        "method": "OLS",
        "notes": ""
    })

    controls = [c for c in CONTROL_CANDIDATES if c in df_states.columns]
    formula_B = "log_passengers ~ log_price"
    if controls:
        formula_B += " + " + " + ".join(controls)
    info(f"Spécification B (controls={controls}) ...")

    if "route_id" in df_states.columns:
        mB = smf.ols(formula_B, data=df_states).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_states["route_id"], "use_correction": True}
        )
    else:
        mB = smf.ols(formula_B, data=df_states).fit(cov_type="HC1")

    results.append({
        "spec": "B_controls",
        "beta": mB.params.get("log_price", np.nan),
        "se": mB.bse.get("log_price", np.nan),
        "method": "OLS",
        "notes": f"controls={controls}"
    })

    try:
        from linearmodels.panel import PanelOLS
        has_panel = True
    except ImportError:
        info("linearmodels.panel non installé -> specs FE sautées.")
        has_panel = False

    if has_panel:
        panel = df_states.set_index(["route_id", "Year"]).sort_index()
        Y = panel["log_passengers"]

        cols_x = ["log_price"] + [c for c in controls if c in panel.columns]
        X = sm.add_constant(panel[cols_x])

        info("Spec C : FE route_id seulement ...")
        mC = PanelOLS(Y, X, entity_effects=True, time_effects=False).fit(
            cov_type="clustered", cluster_entity=True
        )
        results.append({
            "spec": "C_FE_route",
            "beta": mC.params.get("log_price", np.nan),
            "se": mC.std_errors.get("log_price", np.nan),
            "method": "PanelOLS",
            "notes": "entity_effects=True, time_effects=False"
        })

        info("Spec D : FE route_id + année ...")
        mD = PanelOLS(Y, X, entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        results.append({
            "spec": "D_FE_route_year",
            "beta": mD.params.get("log_price", np.nan),
            "se": mD.std_errors.get("log_price", np.nan),
            "method": "PanelOLS",
            "notes": "entity_effects=True, time_effects=True"
        })

    instr = [c for c in INSTRUMENT_CANDIDATES if c in df_states.columns]
    if instr:
        try:
            from linearmodels.iv import IV2SLS
            info(f"Spec E : IV2SLS avec instruments {instr} ...")

            exog_part = "1"
            if controls:
                exog_part += " + " + " + ".join(controls)
            instr_part = " + ".join(instr)
            formula_iv = f"log_passengers ~ {exog_part} + [log_price ~ {instr_part}]"

            iv_mod = IV2SLS.from_formula(formula_iv, data=df_states)

            if "route_id" in df_states.columns:
                mE = iv_mod.fit(cov_type="clustered", clusters=df_states["route_id"])
            else:
                mE = iv_mod.fit(cov_type="robust")

            print(mE.summary)
            results.append({
                "spec": "E_IV",
                "beta": mE.params.get("log_price", np.nan),
                "se": mE.std_errors.get("log_price", np.nan),
                "method": "IV2SLS",
                "notes": f"instruments={instr}, controls={controls}"
            })
        except ImportError:
            info("linearmodels.iv non installé -> spec IV sautée.")
    else:
        info("Aucun instrument trouvé dans les colonnes -> spec IV sautée.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "robustness_elasticities_states.csv"), index=False)
    info("Tableau de robustesse sauvegardé : robustness_elasticities_states.csv")
    return res_df


# ---------- SIMULATION D'UNE TAXE (SCÉNARIO DE POLITIQUE) ----------

def simulate_tax_effect(df_states: pd.DataFrame, elasticity: float, price_increase_pct: float = 0.10):
    info("Simulation d'une taxe / hausse de prix ...")
    if np.isnan(elasticity):
        info("Élasticité fournie = NaN -> simulation annulée.")
        return None

    last_year = df_states["Year"].max()
    df_last = df_states[df_states["Year"] == last_year]

    baseline_pax = df_last["total_passengers"].sum()
    dQ_over_Q = elasticity * price_increase_pct
    new_pax = baseline_pax * (1 + dQ_over_Q)

    info(f"Année: {last_year}")
    info(f"Baseline passengers: {baseline_pax:,.0f}")
    info(f"Elasticity: {elasticity:.3f}, ΔP/P = {price_increase_pct:.2f}")
    info(f"Predicted ΔQ/Q = {dQ_over_Q:.3f}")
    info(f"New passengers (approx.): {new_pax:,.0f}")

    out = {
        "year": last_year,
        "baseline_passengers": baseline_pax,
        "elasticity": elasticity,
        "price_increase_pct": price_increase_pct,
        "delta_Q_over_Q": dQ_over_Q,
        "new_passengers": new_pax
    }
    pd.DataFrame([out]).to_csv(os.path.join(OUT_DIR, "simulation_tax_effect.csv"), index=False)
    return out


# ---------- MAIN ----------

def main():
    ensure_dirs()

    if os.path.exists(MERGED_CSV):
        city_panel = prepare_city_panel(MERGED_CSV)
        elasticity_simple(city_panel, "routes (villes)")
        heterogeneity_by_distance(city_panel, "routes (villes)")
    else:
        info(f"Fichier {MERGED_CSV} introuvable, panel villes sauté.")

    if os.path.exists(MERGED_CSV):
        state_panel = prepare_state_panel(MERGED_CSV)

        descriptives_states(state_panel)
        plot_mean_price_vs_pax_by_state(state_panel)

        elasticity_simple(state_panel, "états -> destinations")

        nat_model, nat_df = elasticity_national_from_states(state_panel)

        fe_results = cross_section_pooled_fe(state_panel)

        heterogeneity_by_distance(state_panel, "états -> destinations")

        robustness_table_states(state_panel)

        beta_fe = fe_results.get("beta_fe_route_year", np.nan)
        if np.isnan(beta_fe):
            beta_nat = nat_model.params.get("log_price", np.nan)
            simulate_tax_effect(state_panel, elasticity=beta_nat, price_increase_pct=0.10)
        else:
            simulate_tax_effect(state_panel, elasticity=beta_fe, price_increase_pct=0.10)

    else:
        info(f"Fichier {MERGED_CSV} introuvable, panel états sauté.")


if __name__ == "__main__":
    main()
