import pandas as pd
from prophet import Prophet

# 1. Load your historical data
df = pd.read_csv("upi_historical_data.csv") # ensure columns ['ds','y']

# 2. Fit the model
m = Prophet(yearly_seasonality=True)
m.fit(df)

# 3. Make future dates and predictions (e.g., 60 months/5 years ahead)
future = m.make_future_dataframe(periods=60, freq='MS')
forecast = m.predict(future)

# 4. Filter only the future predictions (after the last actual date)
last_actual = df["ds"].max()
future_only = forecast[forecast["ds"] > last_actual][["ds", "yhat"]]
future_only.rename(columns={"yhat": "forecast"}, inplace=True)

# 5. Save for dashboard
future_only.to_csv("future_forecast.csv", index=False)
print("Successfully saved future_forecast.csv")
