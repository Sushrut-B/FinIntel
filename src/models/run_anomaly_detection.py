import pandas as pd
import numpy as np

# Configurable threshold multiplier (lowered to 2 for more sensitivity)
THRESHOLD_MULTIPLIER = 1.5

# Load actuals and forecast data
gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
gold["ds"] = pd.to_datetime(gold["ds"]).dt.to_period("M").dt.to_timestamp("M")

preds = pd.read_csv("data/gold/upi_forecast_backtest.csv")
preds["ds"] = pd.to_datetime(preds["ds"]).dt.to_period("M").dt.to_timestamp("M")

# Merge actuals and predictions
eval_df = pd.merge(gold, preds, on="ds", how="inner")

# Calculate residuals and detect anomalies
eval_df["residual"] = eval_df["y"] - eval_df["NBEATSx"]  # Adjust column if forecast column name differs
threshold = THRESHOLD_MULTIPLIER * eval_df["residual"].std()

anomalies = eval_df[eval_df["residual"].abs() > threshold]

# Save anomalies for dashboard
anomalies[["ds", "y", "NBEATSx", "residual"]].to_csv("data/gold/upi_anomalies.csv", index=False)

print(f"Anomaly detection complete. {len(anomalies)} anomalies saved.")
