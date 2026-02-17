import pandas as pd

def detect_anomalies():
    # Load actual data with datetime
    gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
    gold["ds"] = pd.to_datetime(gold["ds"])

    # Load backtest forecast
    forecast = pd.read_csv("data/gold/upi_forecast_backtest.csv")
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # Merge actuals and forecast on date
    df = pd.merge(gold, forecast, on="ds", how="inner")

    # Calculate residual between actual and forecast
    df["residual"] = df["y"] - df["NBEATSx"]

    # Define anomaly threshold as 3 standard deviations
    threshold = df["residual"].std() * 3

    # Flag anomalies where residual absolute value exceeds threshold
    df["anomaly"] = df["residual"].abs() > threshold

    # Display anomalies
    anomalies = df[df["anomaly"]]
    print("Detected anomalies:")
    print(anomalies[["ds", "y", "NBEATSx", "residual"]])

if __name__ == "__main__":
    detect_anomalies()
