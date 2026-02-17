# src/ml/retrain_models.py
from src.utils.alerting import send_anomaly_alert

import pandas as pd
import joblib
from datetime import datetime

def retrain():
    print(f"Model retraining started at {datetime.now()}")

    # Load your latest training data
    try:
        data = pd.read_parquet("data/gold/upi_macro_gold.parquet")
        # --- Removed artificial anomaly injection ---
        # data.loc[0, 'amount'] = data.loc[0, 'amount'] * 10
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Example: simple linear regression training (replace with your actual model logic)
    from sklearn.linear_model import LinearRegression

    try:
        # Prepare features and target (example)
        # Assume 'month' as datetime feature and 'amount' as target
        data['month_num'] = data['month'].dt.month + 12 * (data['month'].dt.year - data['month'].dt.year.min())
        X = data[['month_num']]
        y = data['amount']

        model = LinearRegression()
        model.fit(X, y)

        # Save the trained model
        joblib.dump(model, "models/linear_regression_model.pkl")
        print("Model retrained and saved successfully.")

        # Generate updated forecasts and save CSV
        data['forecast'] = model.predict(X)
        forecast_df = data[['month', 'forecast']].rename(columns={'month': 'ds'})
        forecast_df.to_csv("data/gold/upi_forecast_backtest.csv", index=False)
        print("Updated forecasts saved.")

        # Detect anomalies based on residuals
        eval_df = data.copy()
        eval_df['residual'] = eval_df['amount'] - eval_df['forecast']
        threshold = 3 * eval_df['residual'].std()
        anomalies_detected = eval_df[eval_df['residual'].abs() > threshold]

        # Optionally save anomalies for reference
        if not anomalies_detected.empty:
            anomalies_detected[['month', 'amount', 'forecast', 'residual']].rename(columns={'month': 'ds', 'amount': 'y'}).to_csv("data/gold/upi_anomalies.csv", index=False)
            print(f"Anomalies detected: {len(anomalies_detected)} and saved.")
        else:
            print("No significant anomalies detected.")
            # If you want, remove existing anomalies file if none detected
            import os
            anomaly_file = "data/gold/upi_anomalies.csv"
            if os.path.exists(anomaly_file):
                os.remove(anomaly_file)
                print("Existing anomalies file removed.")

        # Send anomaly alert email if anomalies detected
        if not anomalies_detected.empty:
            send_anomaly_alert(anomalies_detected.rename(columns={'month': 'ds', 'amount': 'y', 'forecast': 'forecast', 'residual': 'residual'}),
                              to_email="bankalgisushrut@gmail.com")

    except Exception as e:
        print(f"Error during model retraining or forecast generation: {e}")

    print(f"Model retraining finished at {datetime.now()}")

if __name__ == "__main__":
    retrain()
