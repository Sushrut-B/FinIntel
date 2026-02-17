import os
import json
import pandas as pd
from pathlib import Path

def ingest_phonepe():
    # Paths
    raw_base = Path("data/raw/phonepe_pulse/data/aggregated/transaction/country/india/state")
    bronze_path = Path("data/bronze/phonepe_transactions.parquet")
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for state in os.listdir(raw_base):
        state_path = raw_base / state
        if not state_path.is_dir():
            continue

        for year in os.listdir(state_path):
            year_path = state_path / year
            for file in os.listdir(year_path):
                if not file.endswith(".json"):
                    continue

                quarter = int(file.replace(".json", ""))
                with open(year_path / file, "r") as f:
                    data = json.load(f)

                # Extract transaction data
                for txn in data.get("data", {}).get("transactionData", []) or []:
                    records.append({
                        "state": state,
                        "year": int(year),
                        "quarter": quarter,
                        "txn_type": txn["name"],
                        "txn_count": txn["paymentInstruments"][0]["count"],
                        "txn_amount": txn["paymentInstruments"][0]["amount"]
                    })

    df = pd.DataFrame(records)

    # Save parquet
    df.to_parquet(bronze_path, index=False)
    print(f"✅ PhonePe Bronze parquet created at: {bronze_path} with {len(df)} rows")

if __name__ == "__main__":
    ingest_phonepe()
