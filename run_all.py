import subprocess
import time
import signal
import sys

def run_command(cmd, wait=True):
    proc = subprocess.Popen(cmd, shell=True)
    if wait:
        proc.wait()
    return proc

def main():
    # Start FastAPI server
    fastapi_proc = run_command("uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")

    # Wait for FastAPI to start fully
    time.sleep(7)

    # Export actuals
    print("Running export_actuals.py ...")
    run_command("python export_actuals.py")

    # Generate forecast
    print("Running generate_future_forecast.py ...")
    run_command("python generate_future_forecast.py")

    # Start Streamlit dashboard
    streamlit_proc = run_command("streamlit run src/dashboard/app.py", wait=False)

    # Start scheduler
    scheduler_proc = run_command("python src/utils/schedule.py", wait=False)

    def shutdown(*args):
        print("Shutting down all processes ...")
        fastapi_proc.terminate()
        streamlit_proc.terminate()
        scheduler_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        fastapi_proc.wait()
        streamlit_proc.wait()
        scheduler_proc.wait()
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
