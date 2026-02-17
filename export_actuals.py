import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"
USERNAME = "your_username"  # Replace with your API login username
PASSWORD = "your_password"  # Replace with your API login password

def export_actuals():
    auth = (USERNAME, PASSWORD)
    response = requests.get(f"{API_BASE_URL}/actuals", auth=auth)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv("upi_historical_data.csv", index=False)
    print("Saved upi_historical_data.csv")

if __name__ == "__main__":
    export_actuals()
