import pandas as pd

df = pd.read_csv("data/raw/npci/upi_product_stats.csv")
print("Sample 'Month' values from raw CSV:")
print(df["Month"].dropna().unique()[:20])  # Show 20 unique non-null values


import pandas as pd
from pathlib import Path

def transform_phonepe():
    bronze_path = Path("data/bronze/phonepe_txn.parquet")
    silver_path = Path("data/silver/phonepe_txn_silver.parquet")

    print("✅ Loading PhonePe Bronze Data...")
    phonepe = pd.read_parquet(bronze_path)

    if "paymentInstruments" in phonepe.columns:
        phonepe = phonepe.explode("paymentInstruments").reset_index(drop=True)
        phonepe = pd.concat(
            [phonepe.drop(columns=["paymentInstruments"]),
             phonepe["paymentInstruments"].apply(pd.Series)],
            axis=1
        )
    
    phonepe.columns = phonepe.columns.str.strip().str.replace(" ", "_").str.lower()

    if ("month" not in phonepe.columns) and ("year" in phonepe.columns and "quarter" in phonepe.columns):
        quarter_to_month = {1: "01", 2: "04", 3: "07", 4: "10"}
        phonepe["month"] = (
            phonepe["year"].astype(str) + "-" + phonepe["quarter"].astype(int).map(quarter_to_month)
        )
        phonepe["month"] = pd.to_datetime(phonepe["month"], format="%Y-%m")

    order = ["month", "year", "quarter", "name", "amount", "count", "type"]
    cols = [c for c in order if c in phonepe.columns] + [c for c in phonepe.columns if c not in order]
    phonepe = phonepe[cols]

    print("✅ PhonePe Transform complete. Preview:")
    print(phonepe.head())

    phonepe.to_parquet(silver_path, index=False)
    print(f"✅ PhonePe Silver data saved at: {silver_path}")

def transform_upi():
    raw_csv_path = Path("data/raw/npci/upi_product_stats.csv")
    bronze_path = Path("data/bronze/npci_stats.parquet")
    silver_path = Path("data/silver/npci_stats_silver.parquet")

    print("✅ Loading NPCI raw CSV Data...")
    upi_raw = pd.read_csv(raw_csv_path)

    # Parse Month safely from float YYYYMM with possible missing values
    upi_raw["Month"] = pd.to_datetime(
        upi_raw["Month"].apply(lambda x: str(int(x)) if pd.notna(x) else None),
        format="%Y%m",
        errors="coerce"
    )

    missing_count = upi_raw["Month"].isna().sum()
    print(f"⚠️ Dropping {missing_count} rows with missing or invalid Month values")

    # Drop rows with missing Month after parse
    upi_raw = upi_raw.dropna(subset=["Month"])

    upi_raw["Year"] = upi_raw["Month"].dt.year
    upi_raw["Quarter"] = upi_raw["Month"].dt.quarter

    upi = upi_raw[["Month", "Year", "Quarter", "Volume", "Value"]]

    print("✅ NPCI Transform complete. Preview:")
    print(upi.head())

    upi.to_parquet(bronze_path, index=False)
    upi.to_parquet(silver_path, index=False)
    print(f"✅ NPCI Bronze and Silver data saved at: {bronze_path} and {silver_path}")

if __name__ == "__main__":
    transform_phonepe()
    transform_upi()


