import pandas as pd
import requests
from io import StringIO
import os

# Step 1: Try to download and parse fresh NPCI UPI statistics page
url = "https://www.npci.org.in/what-we-do/upi/product-statistics"
local_csv = "data/raw/npci/upi_product_stats.csv"
success = False

try:
    print(f"Attempting to download latest NPCI stats from: {url}")
    # Add common headers to mimic real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    # Parse HTML table cleanly
    tables = pd.read_html(StringIO(resp.text))
    if len(tables) > 0:
        df = tables[0]
        # Clean columns
        df.columns = [c.strip() for c in df.columns]
        
        # Save to CSV
        os.makedirs(os.path.dirname(local_csv), exist_ok=True)
        df.to_csv(local_csv, index=False)
        print(f"Successfully scraped and saved fresh NPCI UPI CSV to {local_csv}")
        print(df.head())
        success = True
    else:
        print("Warning: No HTML tables found on the NPCI product statistics page.")
except Exception as e:
    print(f"Warning: Failed to fetch live NPCI stats: {e}")

if not success:
    if os.path.exists(local_csv):
        print(f"Using existing local raw data from fallback: {local_csv}")
        df = pd.read_csv(local_csv)
        print(df.head())
    else:
        print(f"Error: Scrape failed and no local raw data exists at {local_csv}")
        raise RuntimeError("No NPCI data available.")
