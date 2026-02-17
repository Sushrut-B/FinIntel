import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    actuals = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
    nbeats_forecast = pd.read_csv("data/gold/upi_forecast_backtest.csv")
    tft_forecast = pd.read_csv("data/gold/upi_forecast_backtest_tft.csv")

    actuals["ds"] = pd.to_datetime(actuals["ds"])
    nbeats_forecast["ds"] = pd.to_datetime(nbeats_forecast["ds"])
    tft_forecast["ds"] = pd.to_datetime(tft_forecast["ds"])

    return actuals, nbeats_forecast, tft_forecast

def plot_forecasts():
    actuals, nbeats_forecast, tft_forecast = load_data()

    plt.figure(figsize=(12, 6))
    plt.plot(actuals["ds"], actuals["y"], label="Actuals", marker="o")
    plt.plot(nbeats_forecast["ds"], nbeats_forecast["NBEATSx"], label="N-BEATSx Forecast", marker="x")
    plt.plot(tft_forecast["ds"], tft_forecast["TFT"], label="TFT Forecast", marker="^")

    plt.xlabel("Date")
    plt.ylabel("UPI Volume")
    plt.title("UPI Volume: Actual vs Forecast Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_forecasts()
