import subprocess
import datetime
import time

import sys

def retrain_models():
    print(f"Retraining and data updates started at {datetime.datetime.now()}")
    # Run master pipelines orchestrator
    result = subprocess.run([sys.executable, "run_all_pipelines.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Retraining and data pipeline completed successfully.")
        print(result.stdout)
    else:
        print("Retraining and data pipeline failed:")
        print(result.stderr)

if __name__ == "__main__":
    # Example: retrain once every 24 hours
    while True:
        retrain_models()
        time.sleep(24 * 3600)  # Sleep for 24 hours
