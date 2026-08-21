import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from src.services.anomaly_detector import detect_and_classify_anomalies

# Configurable threshold multiplier
THRESHOLD_MULTIPLIER = 1.5

def main():
    # Load actuals and forecast data
    try:
        gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
    except Exception as e:
        print(f"Error loading actuals data: {e}")
        return

    try:
        preds = pd.read_csv("data/gold/upi_forecast_backtest_nbeats.csv")
    except Exception as e:
        print(f"Error loading forecast data: {e}")
        return

    # Use the unified service
    anomalies = detect_and_classify_anomalies(
        actuals_df=gold,
        forecast_df=preds,
        forecast_col="NBEATSx",
        threshold_multiplier=THRESHOLD_MULTIPLIER
    )

    # Save anomalies to CSV
    anomalies.to_csv("data/gold/upi_anomalies.csv", index=False)
    print(f"Anomaly detection complete. {len(anomalies)} anomalies saved to CSV.")

    # Sync anomalies to SQLite
    from src.database.db_manager import get_db_session, AnomalyRecord
    try:
        with get_db_session() as session:
            session.query(AnomalyRecord).delete()
            if not anomalies.empty:
                for _, row in anomalies.iterrows():
                    rec = AnomalyRecord(
                        ds=str(row["ds"]),
                        y=float(row["y"]),
                        forecast=float(row["forecast"]),
                        residual=float(row["residual"]),
                        type=str(row["type"])
                    )
                    session.add(rec)
            session.commit()
            print(f"Synced {len(anomalies)} anomalies to SQLite database.")
    except Exception as e:
        print(f"Error syncing anomalies to SQLite: {e}")

if __name__ == "__main__":
    main()
