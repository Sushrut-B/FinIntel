import pandas as pd
import requests
from io import StringIO
import os

# Step 1: Download NPCI UPI statistics page
url = "https://www.npci.org.in/what-we-do/upi/product-statistics"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()

# Step 2: Parse the HTML table cleanly (requires lxml)
tables = pd.read_html(StringIO(resp.text))
df = tables[0]  # First table is the main stats

# Step 3: Clean up columns
df.columns = [c.strip() for c in df.columns]

# Step 4: Save to CSV
os.makedirs("data/raw/npci", exist_ok=True)
df.to_csv("data/raw/npci/upi_product_stats.csv", index=False)

print("✅ Downloaded and saved fresh NPCI UPI CSV as data/raw/npci/upi_product_stats.csv")
print(df.head())
