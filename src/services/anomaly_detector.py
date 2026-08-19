import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def detect_and_classify_anomalies(actuals_df, forecast_df, forecast_col, threshold_multiplier=1.5):
    """
    Detects anomalies by comparing actuals and forecasts, computes residuals,
    fits an STL decomposition, and classifies the anomalies deterministically.
    
    Args:
        actuals_df (pd.DataFrame): Dataframe containing 'ds' (datetime) and 'y' (target).
        forecast_df (pd.DataFrame): Dataframe containing 'ds' (datetime) and the forecast column.
        forecast_col (str): Name of the column in forecast_df containing the predictions.
        threshold_multiplier (float): Multiplier for standard deviation of residuals.
        
    Returns:
        pd.DataFrame: DataFrame containing detected anomalies with columns:
                      ['ds', 'y', 'forecast', 'residual', 'type']
    """
    # 1. Clean, align dates, and subset only required columns to prevent column collision
    actuals = actuals_df[["ds", "y"]].copy()
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    forecasts = forecast_df[["ds", forecast_col]].copy()
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])
    
    # 2. Join datasets
    eval_df = pd.merge(actuals, forecasts, on="ds", how="inner")
    if eval_df.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast", "residual", "type"])
        
    eval_df = eval_df.rename(columns={forecast_col: "forecast"})
    eval_df["residual"] = eval_df["y"] - eval_df["forecast"]
    
    # 3. Calculate residual standard deviation and threshold
    std_residual = eval_df["residual"].std()
    threshold = threshold_multiplier * std_residual
    
    # 4. Filter anomalies
    anomalies = eval_df[eval_df["residual"].abs() > threshold].copy()
    if anomalies.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast", "residual", "type"])
        
    # 5. Fit STL Decomposition for seasonal outlier detection
    try:
        stl_df = eval_df[['ds', 'y']].sort_values('ds').set_index('ds')
        period = 12 if len(stl_df) >= 24 else 4
        stl = STL(stl_df['y'], period=period, robust=True).fit()
        seasonal = stl.seasonal.rename('seasonal')
        
        # Merge seasonal component back into anomalies
        anomalies = anomalies.merge(seasonal.reset_index(), on='ds', how='left')
        seasonal_std = seasonal.std()
    except Exception as e:
        print(f"Warning: STL decomposition failed in anomaly detector service: {e}")
        anomalies['seasonal'] = 0.0
        seasonal_std = 1.0
        
    # 6. Rule-based classification
    anomaly_types = []
    for _, row in anomalies.iterrows():
        res_val = row['residual']
        seas_val = row.get('seasonal', 0.0)
        
        # Seasonal Outlier: residual deviates opposite to strong seasonal trend
        if abs(seas_val) > 0.5 * seasonal_std and (res_val * seas_val < 0):
            anomaly_types.append("Seasonal Outlier")
        elif res_val > 0:
            anomaly_types.append("Spike")
        else:
            anomaly_types.append("Drop")
            
    anomalies['type'] = anomaly_types
    
    # Clean up output columns
    return anomalies[["ds", "y", "forecast", "residual", "type"]]
