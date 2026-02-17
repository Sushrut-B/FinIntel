import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

def forecast_full():
    gold_path = "data/gold/upi_macro_gold.parquet"
    df = pd.read_parquet(gold_path)

    df = df.rename(columns={"month": "ds", "amount": "y"}).sort_values("ds")
    df["unique_id"] = "upi"

    # Initialize NeuralForecast model
    nf = NeuralForecast(models=[NBEATSx(input_size=12, h=6)], freq="M")

    # Fit on full dataset
    nf.fit(df[["unique_id", "ds", "y"]])

    # Predict next 6 months beyond last data date
    forecast = nf.predict()
    print("Future Forecast:")
    print(forecast)

    # Save forecast results
    forecast.to_csv("data/gold/upi_forecast_future.csv", index=False)
    print("✅ Future forecast saved to data/gold/upi_forecast_future.csv")

if __name__ == "__main__":
    forecast_full()
