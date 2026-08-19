import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from pathlib import Path
from src.features.holiday_calendar import add_calendar_features

def build_features(input_parquet="data/gold/upi_macro_gold.parquet",
                   output_parquet="data/gold/upi_macro_features.parquet"):
    df = pd.read_parquet(input_parquet)
    df = df.sort_values("month").reset_index(drop=True)

    # Add calendar features (month_idx, quarter_idx, holiday_count, is_holiday_month)
    df = add_calendar_features(df, date_col="month")

    # Target variable lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_amount_{lag}m"] = df["amount"].shift(lag)
        df[f"lag_volume_{lag}m"] = df["Volume"].shift(lag)

    # Target variable rolling statistics
    for win in [3, 6, 12]:
        df[f"roll_amount_{win}m_mean"] = df["amount"].rolling(win).mean()
        df[f"roll_amount_{win}m_std"]  = df["amount"].rolling(win).std()
        df[f"roll_volume_{win}m_mean"] = df["Volume"].rolling(win).mean()
        df[f"roll_volume_{win}m_std"]  = df["Volume"].rolling(win).std()

    # Fill NaNs from lags/rolling windows to keep all historical rows for fitting
    # Using forward fill and backward fill is a standard practice in real-time dashboards
    df = df.ffill().bfill()

    # Save features Parquet
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Features built successfully -> {output_parquet} with shape {df.shape}")

if __name__ == "__main__":
    build_features()
