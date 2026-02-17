import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)))

def rolling_origin_backtest(df, target="Volume", horizon=3, n_splits=4):
    """Simple rolling-origin backtest with naive forecast (lag1 baseline)."""
    metrics = []
    n = len(df)

    for i in range(n_splits):
        split_point = n - (n_splits - i)*horizon
        train, test = df.iloc[:split_point], df.iloc[split_point:split_point+horizon]

        # Naive forecast = last observed value repeated
        forecast = np.repeat(train[target].iloc[-1], horizon)

        mape = mean_absolute_error(test[target], forecast) / test[target].mean() * 100
        rmse = np.sqrt(mean_squared_error(test[target], forecast))
        smape_score = smape(test[target].values, forecast)

        metrics.append({
            "split": i+1,
            "mape": mape,
            "rmse": rmse,
            "smape": smape_score
        })

    return pd.DataFrame(metrics)

if __name__ == "__main__":
    df = pd.read_parquet("data/silver/npci_features.parquet")
    results = rolling_origin_backtest(df, target="Volume", horizon=3, n_splits=4)
    print("✅ Backtest results:")
    print(results)
