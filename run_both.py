import sys
import subprocess
import time


def main():
    # Start FastAPI server
    fastapi_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
    ])

    # Wait a few seconds for FastAPI to start
    time.sleep(5)

    # Start Streamlit dashboard
    streamlit_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"
    ])

    # Start scheduler for automated retraining with alerts
    scheduler_proc = subprocess.Popen([
        sys.executable, "src/utils/schedule_retrain.py"
    ])

    try:
        # Wait for all processes
        fastapi_proc.wait()
        streamlit_proc.wait()
        scheduler_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down servers and scheduler...")
        fastapi_proc.terminate()
        streamlit_proc.terminate()
        scheduler_proc.terminate()


if __name__ == "__main__":
    main()
