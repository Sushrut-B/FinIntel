import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def forecast_backtest():
    gold_path = "data/gold/upi_macro_gold.parquet"
    df = pd.read_parquet(gold_path)

    df = df.rename(columns={"month": "ds", "amount": "y"}).sort_values("ds")
    df["unique_id"] = "upi"

    # Split date for train/test: last 6 months as test
    split_date = df["ds"].max() - pd.DateOffset(months=6)
    train_df = df[df["ds"] <= split_date].copy()
    test_df = df[df["ds"] > split_date].copy()

    # Fit the model on train data
    nf = NeuralForecast(models=[NBEATSx(input_size=12, h=6)], freq="M")
    nf.fit(train_df[["unique_id", "ds", "y"]])

    # Forecast next 6 months (test period)
    forecast = nf.predict()
    print("Backtest forecast:")
    print(forecast)

    # Align both ds columns to first day of month for matching keys
    forecast["ds"] = pd.to_datetime(forecast["ds"]).dt.to_period("M").dt.to_timestamp()
    test_df["ds"] = pd.to_datetime(test_df["ds"]).dt.to_period("M").dt.to_timestamp()

    print("Test dates:")
    print(test_df["ds"].to_list())
    print("Forecast dates:")
    print(forecast["ds"].to_list())

    eval_df = pd.merge(test_df, forecast, how="inner", on="ds")

    if eval_df.empty:
        print("No overlapping dates to evaluate!")
        return

    mae = mean_absolute_error(eval_df["y"], eval_df["NBEATSx"])
    mape = mean_absolute_percentage_error(eval_df["y"], eval_df["NBEATSx"])

    print(f"Backtest MAE: {mae:.0f}")
    print(f"Backtest MAPE: {mape*100:.2f}%")

    # Save forecast for further use
    forecast.to_csv("data/gold/upi_forecast_backtest.csv", index=False)
    print("✅ Backtest forecast saved to data/gold/upi_forecast_backtest.csv")

if __name__ == "__main__":
    forecast_backtest()
