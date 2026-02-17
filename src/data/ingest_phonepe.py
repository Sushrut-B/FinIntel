import os
import json
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/phonepe_pulse")
BRONZE_PATH = Path("data/bronze")

def load_phonepe_transactions():
    """Walk through PhonePe Pulse raw JSON and flatten into a DataFrame."""
    rows = []
    for root, _, files in os.walk(RAW_PATH):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f), "r") as infile:
                    data = json.load(infile)
                    if "data" not in data or "transactionData" not in data["data"]:
                        continue
                    for entry in data["data"]["transactionData"]:
                        rows.append({
                            "state": data.get("state", "NA"),
                            "year": data.get("year", "NA"),
                            "quarter": data.get("quarter", "NA"),
                            "name": entry["name"],
                            "count": entry["paymentInstruments"][0]["count"],
                            "amount": entry["paymentInstruments"][0]["amount"]
                        })
    return pd.DataFrame(rows)

def main():
    df = load_phonepe_transactions()
    BRONZE_PATH.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_PATH / "phonepe_txn.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(df)} rows → {out_path}")

if __name__ == "__main__":
    main()
