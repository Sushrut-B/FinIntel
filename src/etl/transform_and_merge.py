import os
import json
import pandas as pd
from pathlib import Path

def transform_and_merge():
    print("Starting Unified ETL and Merging Pipeline...")

    # Paths
    raw_npci_path = Path("data/raw/npci/upi_product_stats.csv")
    raw_phonepe_base = Path("data/raw/phonepe_pulse/data/aggregated/transaction/country/india/state")
    gold_path = Path("data/gold/upi_macro_gold.parquet")
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. Load and Transform NPCI Monthly Data
    # ----------------------------------------------------
    print("Loading and cleaning NPCI stats...")
    if not raw_npci_path.exists():
        raise FileNotFoundError(f"NPCI raw file not found at {raw_npci_path}")
    
    npci_df = pd.read_csv(raw_npci_path)
    npci_df.columns = [c.strip() for c in npci_df.columns]

    # Parse Month (format: Jul-25)
    npci_df["Month"] = pd.to_datetime(npci_df["Month"], format="%b-%y", errors="coerce")
    npci_df = npci_df.dropna(subset=["Month"])

    # Clean commas in numbers
    if "Volume_in_Mn" in npci_df.columns:
        npci_df["Volume"] = npci_df["Volume_in_Mn"].astype(str).str.replace(",", "", regex=False).astype(float)
    elif "Volume (in Mn)" in npci_df.columns:
        npci_df["Volume"] = npci_df["Volume (in Mn)"].astype(str).str.replace(",", "", regex=False).astype(float)
    else:
        npci_df["Volume"] = npci_df["Volume"].astype(str).str.replace(",", "", regex=False).astype(float)

    if "Value_in_Cr" in npci_df.columns:
        npci_df["Value"] = npci_df["Value_in_Cr"].astype(str).str.replace(",", "", regex=False).astype(float)
    elif "Value (in Cr.)" in npci_df.columns:
        npci_df["Value"] = npci_df["Value (in Cr.)"].astype(str).str.replace(",", "", regex=False).astype(float)
    else:
        npci_df["Value"] = npci_df["Value"].astype(str).str.replace(",", "", regex=False).astype(float)

    # Re-normalize to original names: Month, Year, Quarter, Volume, Value
    npci_df["Year"] = npci_df["Month"].dt.year
    npci_df["Quarter"] = npci_df["Month"].dt.quarter
    npci_df = npci_df[["Month", "Year", "Quarter", "Volume", "Value"]].rename(columns={"Month": "month"})
    npci_df = npci_df.sort_values("month").reset_index(drop=True)
    print(f"Cleaned NPCI data: {len(npci_df)} monthly records.")

    # ----------------------------------------------------
    # 2. Load and Flatten PhonePe Quarterly Data
    # ----------------------------------------------------
    print("Loading and flattening PhonePe Pulse JSON files...")
    if not raw_phonepe_base.exists():
        raise FileNotFoundError(f"PhonePe base path not found at {raw_phonepe_base}")

    phonepe_records = []
    for state in os.listdir(raw_phonepe_base):
        state_path = raw_phonepe_base / state
        if not state_path.is_dir():
            continue
        for year in os.listdir(state_path):
            year_path = state_path / year
            if not year_path.is_dir():
                continue
            for file in os.listdir(year_path):
                if not file.endswith(".json"):
                    continue
                quarter = int(file.replace(".json", ""))
                with open(year_path / file, "r") as f:
                    data = json.load(f)
                for txn in data.get("data", {}).get("transactionData", []) or []:
                    phonepe_records.append({
                        "year": int(year),
                        "quarter": quarter,
                        "amount": txn["paymentInstruments"][0]["amount"],
                        "count": txn["paymentInstruments"][0]["count"]
                    })
    
    phonepe_df = pd.DataFrame(phonepe_records)
    # Aggregate country-wide quarterly sums
    phonepe_agg = phonepe_df.groupby(["year", "quarter"]).agg({
        "amount": "sum",
        "count": "sum"
    }).reset_index()

    # Map year and quarter to quarter start month for joining
    quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
    phonepe_agg["q_start_month"] = pd.to_datetime(
        phonepe_agg["year"].astype(str) + "-" + phonepe_agg["quarter"].map(quarter_to_month).astype(str) + "-01"
    )
    print(f"Cleaned PhonePe data: {len(phonepe_agg)} quarterly records.")

    # ----------------------------------------------------
    # 3. Merge & Upsample PhonePe Proportional to NPCI
    # ----------------------------------------------------
    print("Merging and upsampling PhonePe quarterly to NPCI monthly granularity...")
    
    # Map each monthly NPCI row to its corresponding quarter start month
    npci_df["q_start_month"] = pd.to_datetime(
        npci_df["Year"].astype(str) + "-" + npci_df["Quarter"].map(quarter_to_month).astype(str) + "-01"
    )

    # Left join to preserve all monthly NPCI rows
    merged = pd.merge(
        npci_df,
        phonepe_agg[["q_start_month", "amount", "count"]].rename(columns={"amount": "amount_q", "count": "count_q"}),
        on="q_start_month",
        how="left"
    )

    # Calculate sum of NPCI Volume for each quarter group to distribute PhonePe metrics proportionally
    quarter_vol_sums = merged.groupby("q_start_month")["Volume"].transform("sum")

    # Distribute the quarterly amount/count proportionally to monthly NPCI Volume ratio
    merged["amount"] = merged["amount_q"] * (merged["Volume"] / quarter_vol_sums)
    merged["count"] = merged["count_q"] * (merged["Volume"] / quarter_vol_sums)

    # ----------------------------------------------------
    # 4. Integrate Macroeconomic Data (FRED CPI & Exchange Rate)
    # ----------------------------------------------------
    print("Loading and cleaning macroeconomic indicators from FRED...")
    raw_cpi_path = Path("data/raw/macro/india_cpi.csv")
    raw_ex_rate_path = Path("data/raw/macro/inr_usd_exchange_rate.csv")
    
    # Load and clean CPI
    if raw_cpi_path.exists():
        cpi_df = pd.read_csv(raw_cpi_path)
        cpi_df.columns = [c.strip() for c in cpi_df.columns]
        cpi_df["month"] = pd.to_datetime(cpi_df["observation_date"])
        cpi_df["cpi"] = pd.to_numeric(cpi_df["INDCPIALLMINMEI"], errors="coerce")
        cpi_df = cpi_df[["month", "cpi"]].dropna()
    else:
        print("Warning: Raw CPI data not found. Creating placeholder.")
        cpi_df = pd.DataFrame(columns=["month", "cpi"])

    # Load and clean Exchange Rate
    if raw_ex_rate_path.exists():
        ex_df = pd.read_csv(raw_ex_rate_path)
        ex_df.columns = [c.strip() for c in ex_df.columns]
        ex_df["month"] = pd.to_datetime(ex_df["observation_date"])
        ex_df["exchange_rate"] = pd.to_numeric(ex_df["EXINUS"], errors="coerce")
        ex_df = ex_df[["month", "exchange_rate"]].dropna()
    else:
        print("Warning: Raw Exchange Rate data not found. Creating placeholder.")
        ex_df = pd.DataFrame(columns=["month", "exchange_rate"])

    # Clean up intermediate join columns for the base UPI dataset
    merged = merged[["month", "Year", "Quarter", "Volume", "Value", "amount", "count"]]

    # Join Macro CPI
    merged = pd.merge(merged, cpi_df, on="month", how="left")
    # Join Macro Exchange Rate
    merged = pd.merge(merged, ex_df, on="month", how="left")

    # Fallback/fill values if any missing (including macro features)
    merged["amount"] = merged["amount"].ffill().bfill()
    merged["count"] = merged["count"].ffill().bfill()
    merged["cpi"] = merged["cpi"].ffill().bfill()
    merged["exchange_rate"] = merged["exchange_rate"].ffill().bfill()
    
    # Save Parquet
    merged.to_parquet(gold_path, index=False)
    print(f"Gold layer monthly dataset with macro features created successfully at: {gold_path} with {len(merged)} records.")
    print(merged.head())

if __name__ == "__main__":
    transform_and_merge()
