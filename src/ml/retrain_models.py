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
        # Select active engineered features including macro indicators
        feature_cols = [
            'month_idx', 'quarter_idx', 'holiday_count', 'is_holiday_month',
            'lag_amount_1m', 'lag_amount_2m', 'lag_amount_3m',
            'lag_volume_1m', 'lag_volume_2m', 'lag_volume_3m',
            'roll_amount_3m_mean', 'roll_amount_3m_std',
            'roll_volume_3m_mean', 'roll_volume_3m_std',
            'cpi', 'exchange_rate',
            'lag_cpi_1m', 'lag_cpi_2m',
            'lag_exchange_rate_1m', 'lag_exchange_rate_2m',
            'roll_cpi_3m_mean', 'roll_exchange_rate_3m_mean'
        ]
        
        X = data[feature_cols]
        y = data['amount']

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X, y)

        # Generate forecasts
        data['forecast'] = model.predict(X)

        # Evaluate performance metrics
        import numpy as np
        mae = float(np.mean(np.abs(y - data['forecast'])))
        mape = float(np.mean(np.abs((y - data['forecast']) / y)) * 100)
        print(f"Candidate model evaluated. MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        # Query active model from registry
        from src.database.db_manager import get_db_session, ModelRegistry, AnomalyRecord
        
        active_model = None
        with get_db_session() as session:
            active_model = session.query(ModelRegistry).filter(
                ModelRegistry.model_name == "GradientBoosting",
                ModelRegistry.status == "Active"
            ).order_by(ModelRegistry.id.desc()).first()

        # Promotion Gate Decision
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"gb_v_{timestamp}"
        
        is_promoted = False
        if active_model is None:
            print("[Promotion Gate] No active model registered. Promoting candidate model by default.")
            is_promoted = True
        elif mae < active_model.mae:
            print(f"[Promotion Gate] Candidate model outperforms active model (New MAE: {mae:.2f} < Active MAE: {active_model.mae:.2f}). Promoting model.")
            is_promoted = True
        else:
            print(f"[Promotion Gate] Candidate model rejected. (New MAE: {mae:.2f} >= Active MAE: {active_model.mae:.2f}). Retaining active model.")
            is_promoted = False

        os.makedirs("models", exist_ok=True)
        
        if is_promoted:
            # 1. Save versioned binary
            versioned_path = f"models/gradient_boosting_{version_str}.pkl"
            joblib.dump(model, versioned_path)
            
            # 2. Save operational binaries
            joblib.dump(model, "models/gradient_boosting_model.pkl")
            joblib.dump(model, "models/linear_regression_model.pkl")
            print(f"Gradient Boosting Regressor saved to {versioned_path} and operational path.")

            # 3. Save operational forecasts
            forecast_df = data[['month', 'forecast']].rename(columns={'month': 'ds'})
            forecast_df.to_csv("data/gold/upi_forecast_backtest_linreg.csv", index=False)
            print("Updated forecasts saved to upi_forecast_backtest_linreg.csv")

            # 4. Detect anomalies using the unified service
            from src.services.anomaly_detector import detect_and_classify_anomalies
            actuals_df = data.rename(columns={'month': 'ds', 'amount': 'y'})
            forecast_df = data.rename(columns={'month': 'ds'})

            anomalies_detected = detect_and_classify_anomalies(
                actuals_df=actuals_df,
                forecast_df=forecast_df,
                forecast_col="forecast",
                threshold_multiplier=1.5
            )

            if not anomalies_detected.empty:
                anomalies_detected.to_csv("data/gold/upi_anomalies.csv", index=False)
                print(f"Anomalies detected: {len(anomalies_detected)} and saved locally.")
            else:
                print("No significant anomalies detected.")
                anomaly_file = "data/gold/upi_anomalies.csv"
                if os.path.exists(anomaly_file):
                    os.remove(anomaly_file)
                    print("Existing anomalies CSV removed.")

            # 5. Register in DB & Sync Anomalies
            with get_db_session() as session:
                # Deprecate older models
                session.query(ModelRegistry).filter(
                    ModelRegistry.model_name == "GradientBoosting",
                    ModelRegistry.status == "Active"
                ).update({"status": "Deprecated"})
                
                # Add new model record
                registry_record = ModelRegistry(
                    model_name="GradientBoosting",
                    version=version_str,
                    train_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    mae=mae,
                    mape=mape,
                    file_path=versioned_path,
                    status="Active"
                )
                session.add(registry_record)
                
                # Sync Anomaly records in SQLite
                session.query(AnomalyRecord).delete()
                if not anomalies_detected.empty:
                    for _, row in anomalies_detected.iterrows():
                        rec = AnomalyRecord(
                            ds=str(row["ds"]),
                            y=float(row["y"]),
                            forecast=float(row["forecast"]),
                            residual=float(row["residual"]),
                            type=str(row["type"])
                        )
                        session.add(rec)
                session.commit()
                print(f"Registered model {version_str} and synced anomalies to SQLite.")

            # 6. Send anomaly alerts
            if not anomalies_detected.empty:
                send_anomaly_alert(anomalies_detected, to_email="bankalgisushrut@gmail.com")
        else:
            # Save rejected model binary for debugging/logging
            rejected_path = f"models/gradient_boosting_rejected_{version_str}.pkl"
            joblib.dump(model, rejected_path)
            print(f"Rejected model binary saved to {rejected_path} for archive.")
            
            with get_db_session() as session:
                registry_record = ModelRegistry(
                    model_name="GradientBoosting",
                    version=version_str,
                    train_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    mae=mae,
                    mape=mape,
                    file_path=rejected_path,
                    status="Rejected"
                )
                session.add(registry_record)
                session.commit()
                print(f"Registered rejected model run {version_str} in database.")

    except Exception as e:
        print(f"Error during model retraining or forecast generation: {e}")

    print(f"Model retraining finished at {datetime.now()}")

if __name__ == "__main__":
    retrain()
