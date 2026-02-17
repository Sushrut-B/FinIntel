import pandas as pd
from pathlib import Path

def ingest_npci():
    # Paths
    raw_path = Path("data/raw/npci/upi_product_stats.csv")
    bronze_path = Path("data/bronze/npci_stats.parquet")
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw CSV
    df = pd.read_csv(raw_path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure Month column is datetime
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

    # Convert numeric columns (remove commas if needed)
    if "Volume (in Mn)" in df.columns:
        df["Volume"] = (
            df["Volume (in Mn)"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    if "Value (in Cr.)" in df.columns:
        df["Value"] = (
            df["Value (in Cr.)"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    # Keep only clean cols
    df = df[["Month", "Volume", "Value"]].dropna().sort_values("Month")

    # Save parquet to bronze
    df.to_parquet(bronze_path, index=False)

    print(f"✅ NPCI Bronze parquet created at: {bronze_path}")

if __name__ == "__main__":
    ingest_npci()
