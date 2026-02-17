import pandas as pd

def save_anomalies():
    gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
    forecast = pd.read_csv("data/gold/upi_forecast_backtest.csv")
    gold["ds"] = pd.to_datetime(gold["ds"])
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    df = pd.merge(gold, forecast, on="ds", how="inner")
    df["residual"] = df["y"] - df["NBEATSx"]
    threshold = df["residual"].std() * 1.5  # Adjust threshold as needed
    df["anomaly"] = df["residual"].abs() > threshold
    anomalies = df[df["anomaly"]]
    anomalies.to_csv("data/gold/upi_anomalies.csv", index=False)
    print(f"Saved {len(anomalies)} anomalies")

if __name__ == "__main__":
    save_anomalies()
