import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

def forecast_upi():
    # Load the gold dataset
    gold_path = "data/gold/upi_macro_gold.parquet"
    df = pd.read_parquet(gold_path)

    # Prepare data: rename, sort, and add unique_id
    df_ = df.rename(columns={"month": "ds", "amount": "y"}).sort_values("ds")
    df_["unique_id"] = "upi"

    # Initialize model (freq="M" = monthly)
    nf = NeuralForecast(
        models=[NBEATSx(input_size=12, h=6)],
        freq="M"
    )

    # Fit model with required columns
    nf.fit(df_[["unique_id", "ds", "y"]])

    # Predict next 6 months
    forecast = nf.predict()
    print(forecast)

    # Save forecast to CSV
    forecast.to_csv("data/gold/upi_forecast.csv", index=False)
    print("✅ Forecast saved to data/gold/upi_forecast.csv")

if __name__ == "__main__":
    forecast_upi()

