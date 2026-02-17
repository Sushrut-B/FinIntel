import pandas as pd
from pathlib import Path

def transform_npci():
    csv_path = Path("data/raw/npci/upi_product_stats.csv")
    silver_path = Path("data/silver/npci_stats_silver.parquet")

    df = pd.read_csv(csv_path)

    # Parse Month column (format: Jul-25)
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")

    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter

    df = df.dropna(subset=["Month"])

    df.to_parquet(silver_path, index=False)
    print(f"✅ NPCI Silver data saved to {silver_path}")
    print(df.head())

if __name__ == "__main__":
    transform_npci()
