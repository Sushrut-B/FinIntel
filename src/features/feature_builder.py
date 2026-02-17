import pandas as pd
from pathlib import Path
from src.features.holiday_calendar import add_calendar_features

def build_features(input_parquet="data/bronze/npci_stats.parquet",
                   output_parquet="data/silver/npci_features.parquet"):
    df = pd.read_parquet(input_parquet)
    df = df.rename(columns={"Month": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # Add calendar features
    df = add_calendar_features(df, date_col="date")

    # Monthly lags
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}m"] = df["Volume"].shift(lag)

    # Rolling stats
    for win in [3, 6, 12]:
        df[f"roll{win}_mean"] = df["Volume"].rolling(win).mean()
        df[f"roll{win}_std"]  = df["Volume"].rolling(win).std()

    # 🚨 Don’t drop all NA rows — only drop if target itself missing
    df = df.reset_index(drop=True)

    # Save
    Path("data/silver").mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Features saved → {output_parquet} with shape {df.shape}")

if __name__ == "__main__":
    build_features()
