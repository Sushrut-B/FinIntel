import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"
USERNAME = "sushrut"  # Replace with your API login username
PASSWORD = "sushrutpass"  # Replace with your API login password

def export_actuals():
    # Fetch JWT Token
    login_response = requests.post(
        f"{API_BASE_URL}/login", 
        json={"username": USERNAME, "password": PASSWORD},
        timeout=10
    )
    login_response.raise_for_status()
    token = login_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/actuals", headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv("upi_historical_data.csv", index=False)
    print("Saved upi_historical_data.csv using JWT credentials.")

if __name__ == "__main__":
    export_actuals()
