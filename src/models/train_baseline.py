import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_data(path):
    df = pd.read_parquet(path)
    df = df.rename(columns={"month": "ds", "amount": "y"})
    df["unique_id"] = "upi"
    df = df.sort_values("ds")
    return df

def train_backtest(df):
    # Split data into train and test
    train = df[df["ds"] < "2024-07-01"]
    test = df[df["ds"] >= "2024-07-01"]

    # Align test dates to month ends to match prediction freq
    test["ds"] = test["ds"].dt.to_period("M").dt.to_timestamp("M")

    model = NeuralForecast(models=[NBEATSx(input_size=12, h=6)], freq="M")
    model.fit(train[["unique_id", "ds", "y"]])

    preds = model.predict()

    print("Test ds:", test["ds"].tolist())
    print("Predicted ds:", preds["ds"].tolist())

    preds_test = preds[preds["ds"].isin(test["ds"])]

    print(f"Matched preds_test.shape: {preds_test.shape}, test.shape: {test.shape}")

    if preds_test.empty:
        print("No overlapping prediction dates with test set. Check frequency, horizon, and date alignment.")
        return

    # Filter test to matching dates only
    matched_dates = preds_test["ds"].unique()
    test_filtered = test[test["ds"].isin(matched_dates)]

    mae = mean_absolute_error(test_filtered["y"], preds_test["NBEATSx"])
    mape = mean_absolute_percentage_error(test_filtered["y"], preds_test["NBEATSx"]) * 100

    print(f"Backtest MAE: {mae}")
    print(f"Backtest MAPE: {mape:.2f}%")

    preds.to_csv("data/gold/upi_forecast_backtest.csv", index=False)
    print("Backtest forecast saved to data/gold/upi_forecast_backtest.csv")

if __name__ == "__main__":
    df = load_data("data/gold/upi_macro_gold.parquet")
    train_backtest(df)
