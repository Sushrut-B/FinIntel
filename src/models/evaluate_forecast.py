import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def evaluate():
    gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
    forecast = pd.read_csv("data/gold/upi_forecast.csv")
    gold["ds"] = pd.to_datetime(gold["ds"], errors="coerce")
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")

    print("Gold min/max ds:", gold["ds"].min(), gold["ds"].max())
    print("Forecast min/max ds:", forecast["ds"].min(), forecast["ds"].max())

    forecast_hist = forecast[forecast["ds"].isin(gold["ds"])]

    merged = pd.merge(gold, forecast_hist, how="inner", on="ds")

    if merged.empty:
        print("No overlapping dates in actuals and forecast to evaluate!")
        return

    mae = mean_absolute_error(merged["y"], merged["NBEATSx"])
    mape = mean_absolute_percentage_error(merged["y"], merged["NBEATSx"])

    print(f"MAE: {mae:,.0f}")
    print(f"MAPE: {mape * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
