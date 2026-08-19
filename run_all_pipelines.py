import subprocess
import sys
import os

def run_script(path):
    print(f"\n--- Running: {path} ---")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success: {path}")
        print(result.stdout)
    else:
        print(f"Failed: {path}")
        print(result.stderr)
        raise RuntimeError(f"Pipeline step failed: {path}")

def main():
    print("Starting Master Data and Retraining Pipeline...")
    try:
        # Step 1: Dynamic NPCI stats ingestion
        run_script("src/data/ingest_npci.py")
        
        # Step 2: Unified transform & merge (upsampling PhonePe data)
        run_script("src/etl/transform_and_merge.py")
        
        # Step 3: Feature builder (dynamic holidays, lags, rolling stats)
        run_script("src/features/feature_builder.py")
        
        # Step 4: ML Retraining & STL Anomaly detection
        run_script("src/ml/retrain_models.py")
        
        print("Master Pipeline Executed Successfully!")
    except Exception as e:
        print(f"Master Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
