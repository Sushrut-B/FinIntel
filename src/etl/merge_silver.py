import pandas as pd
from pathlib import Path

def merge_silver():
    phonepe_path = Path("data/silver/phonepe_txn_silver.parquet")
    npci_path = Path("data/silver/npci_stats_silver.parquet")
    gold_path = Path("data/gold/upi_macro_gold.parquet")

    # Create data/gold directory if it doesn't exist
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    phonepe = pd.read_parquet(phonepe_path)
    npci = pd.read_parquet(npci_path)

    phonepe_agg = phonepe.groupby("month").agg({
        "amount": "sum",
        "count": "sum"
    }).reset_index()

    npci = npci.rename(columns={"Month": "month"})

    merged = pd.merge(npci, phonepe_agg, how="inner", on="month")

    merged.to_parquet(gold_path, index=False)
    print(f"✅ Gold layer dataset saved at: {gold_path}")
    print(merged.head())

if __name__ == "__main__":
    merge_silver()
