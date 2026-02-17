import subprocess
import datetime
import time

def retrain_models():
    print(f"Retraining started at {datetime.datetime.now()}")
    # Run your retraining script or commands here; adjust path as needed
    result = subprocess.run(["python", "src/ml/retrain_models.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Retraining completed successfully.")
    else:
        print("Retraining failed:")
        print(result.stderr)

if __name__ == "__main__":
    # Example: retrain once every 24 hours
    while True:
        retrain_models()
        time.sleep(24 * 3600)  # Sleep for 24 hours
