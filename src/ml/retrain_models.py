# src/ml/retrain_models.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.alerting import send_anomaly_alert

import pandas as pd
import joblib
from datetime import datetime

def retrain():
    print(f"Model retraining started at {datetime.now()}")

    # Load your latest training data containing engineered features
    try:
        data = pd.read_parquet("data/gold/upi_macro_features.parquet")
    except Exception as e:
        print(f"Error loading training features data: {e}")
        return

    # Train an optimized Gradient Boosting Regressor (ML based)
    from sklearn.ensemble import GradientBoostingRegressor

    try:
        # Select active engineered features
        feature_cols = [
            'month_idx', 'quarter_idx', 'holiday_count', 'is_holiday_month',
            'lag_amount_1m', 'lag_amount_2m', 'lag_amount_3m',
            'lag_volume_1m', 'lag_volume_2m', 'lag_volume_3m',
            'roll_amount_3m_mean', 'roll_amount_3m_std',
            'roll_volume_3m_mean', 'roll_volume_3m_std'
        ]
        
        X = data[feature_cols]
        y = data['amount']

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X, y)

        # Save the trained model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/gradient_boosting_model.pkl")
        # Keep linear_regression_model.pkl as fallback link
        joblib.dump(model, "models/linear_regression_model.pkl")
        print("Gradient Boosting Regressor retrained and saved successfully.")

        # Generate forecasts
        data['forecast'] = model.predict(X)
        forecast_df = data[['month', 'forecast']].rename(columns={'month': 'ds'})
        forecast_df.to_csv("data/gold/upi_forecast_backtest_linreg.csv", index=False)
        print("Updated forecasts saved to upi_forecast_backtest_linreg.csv")

        # Detect anomalies using the unified service
        from src.services.anomaly_detector import detect_and_classify_anomalies

        actuals_df = data.rename(columns={'month': 'ds', 'amount': 'y'})
        forecast_df = data.rename(columns={'month': 'ds'})

        anomalies_detected = detect_and_classify_anomalies(
            actuals_df=actuals_df,
            forecast_df=forecast_df,
            forecast_col="forecast",
            threshold_multiplier=1.5
        )

        # Optionally save anomalies for reference
        if not anomalies_detected.empty:
            anomalies_detected.to_csv("data/gold/upi_anomalies.csv", index=False)
            print(f"Anomalies detected: {len(anomalies_detected)} and saved using unified service.")
        else:
            print("No significant anomalies detected.")
            anomaly_file = "data/gold/upi_anomalies.csv"
            if os.path.exists(anomaly_file):
                os.remove(anomaly_file)
                print("Existing anomalies file removed.")

        # Send anomaly alert email if anomalies detected
        if not anomalies_detected.empty:
            send_anomaly_alert(anomalies_detected, to_email="bankalgisushrut@gmail.com")

    except Exception as e:
        print(f"Error during model retraining or forecast generation: {e}")

    print(f"Model retraining finished at {datetime.now()}")

if __name__ == "__main__":
    retrain()
