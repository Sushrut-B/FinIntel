import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_data(path):
    df = pd.read_parquet(path)
    df = df.rename(columns={"month": "ds", "amount": "y"})
    df["unique_id"] = "upi"
    df = df.sort_values("ds")
    return df

def train_backtest(df):
    train = df[df["ds"] < "2024-07-01"]
    test = df[df["ds"] >= "2024-07-01"]

    # Align test dates to month ends
    test["ds"] = test["ds"].dt.to_period("M").dt.to_timestamp("M")

    model = NeuralForecast(models=[TFT(input_size=12, h=6)], freq="M")
    model.fit(train[["unique_id", "ds", "y"]])

    preds = model.predict()

    preds_test = preds[preds["ds"].isin(test["ds"])]

    matched_dates = preds_test["ds"].unique()
    test_filtered = test[test["ds"].isin(matched_dates)]

    mae = mean_absolute_error(test_filtered["y"], preds_test["TFT"])
    mape = mean_absolute_percentage_error(test_filtered["y"], preds_test["TFT"]) * 100

    print(f"TFT Backtest MAE: {mae}")
    print(f"TFT Backtest MAPE: {mape:.2f}%")

    preds.to_csv("data/gold/upi_forecast_backtest_tft.csv", index=False)
    print("TFT backtest forecast saved to data/gold/upi_forecast_backtest_tft.csv")

if __name__ == "__main__":
    df = load_data("data/gold/upi_macro_gold.parquet")
    train_backtest(df)
    