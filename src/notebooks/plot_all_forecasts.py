import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load actual data
gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})

# Load backtest forecast
backtest = pd.read_csv("data/gold/upi_forecast_backtest.csv")
backtest["ds"] = pd.to_datetime(backtest["ds"])

# Load future forecast
future = pd.read_csv("data/gold/upi_forecast_future.csv")
future["ds"] = pd.to_datetime(future["ds"])

plt.figure(figsize=(14,6))

# Plot actuals
plt.plot(gold["ds"], gold["y"], label="Historical Actuals", marker='o')

# Plot backtest forecast (last 6 months)
plt.plot(backtest["ds"], backtest["NBEATSx"], label="Backtest Forecast", marker='x')

# Plot future forecast (next 6 months)
plt.plot(future["ds"], future["NBEATSx"], label="Future Forecast", marker='x', linestyle='--')

plt.title("UPI Volume: Historical, Backtest Forecast & Future Forecast")
plt.xlabel("Date")
plt.ylabel("UPI Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig("output/upi_forecast_plot.png")
print("✅ Plot saved to output/upi_forecast_plot.png")

# Comment out interactive plot display to avoid hanging or manual interrupts
# plt.show()
