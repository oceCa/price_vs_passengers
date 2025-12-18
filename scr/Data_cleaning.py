import pandas as pd
from pathlib import Path

def main(input_dir="."):
    input_dir = Path(input_dir)

    # Charger le CSV principal
    cities_path = input_dir / "data/row_data/Cities_All.csv"
    df = pd.read_csv(cities_path)


    # Garder seulement les colonnes voulues
    cols_to_keep = [
        "Year",
        "quarter",
        "citymarketid_1",
        "citymarketid_2",
        "city1",
        "city2",
        "nsmiles",
        "passengers",
        "fare",
        "carrier_lg",
        "large_ms",
        "carrier_low",
        "lf_ms",
        "fare_low",
    ]
    df = df[cols_to_keep]

    # 4) Charger l’inflation
    inflation_path = input_dir / "data/row_data/inflation_data.csv"
    inflation_df = pd.read_csv(inflation_path)

    # petit helper pour retrouver les bons noms
    def find_col(df_, target_name: str) -> str:
        target = target_name.strip().lower()
        for c in df_.columns:
            if str(c).strip().lower() == target:
                return c
        raise KeyError(f"Column '{target_name}' not found in inflation_data.csv")

    year_col = find_col(inflation_df, "Year")
    amount_col = find_col(inflation_df, "amount")

    inflation_df = inflation_df[[year_col, amount_col]].rename(
        columns={year_col: "Year", amount_col: "amount"}
    )

    # 5) Typage
    inflation_df["Year"] = pd.to_numeric(inflation_df["Year"], errors="coerce").astype("Int64")
    inflation_df["amount"] = pd.to_numeric(inflation_df["amount"], errors="coerce")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # ⚠️ 6) Nettoyer la colonne fare (enlever le $)
    # Exemple typique: "$220.10" -> 220.10
    df["fare"] = (
        df["fare"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)  # enlève $ et ,
        .str.strip()
        .replace("", pd.NA)
        .astype(float)
    )

    # 7) Merge avec l’inflation
    df = df.merge(inflation_df, on="Year", how="left")

    # 8) Calcul du prix réel
    # si amount = "combien vaut 1$ de 1996 en 2025", on fait fare / amount
    df["Real price"] = df["fare"] / df["amount"]

    # 9) Sauvegarde
    output_path = input_dir / "data/merged_air_travel_data.csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main(".")
