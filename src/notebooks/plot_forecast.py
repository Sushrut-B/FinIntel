import pandas as pd
import matplotlib.pyplot as plt

# Load actual gold data
gold = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})

# Load forecast saved previously
forecast = pd.read_csv("data/gold/upi_forecast.csv")

plt.figure(figsize=(10,5))
plt.plot(gold["ds"], gold["y"], label="Actual UPI Volume", marker='o')
plt.plot(forecast["ds"], forecast["NBEATSx"], label="Forecast", marker='x')
plt.title("UPI Volume: Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()
