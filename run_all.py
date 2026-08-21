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
    # Start FastAPI server with multiple workers for high availability and throughput
    import os
    workers_count = max(2, min(os.cpu_count() or 2, 8))
    print(f"Launching FastAPI server with {workers_count} uvicorn processes...")
    fastapi_proc = run_command(f'"{sys.executable}" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers {workers_count}', wait=False)

    # Wait for FastAPI to start fully
    time.sleep(7)

    # Export actuals
    print("Running export_actuals.py ...")
    run_command(f'"{sys.executable}" export_actuals.py')

    # Generate forecast
    print("Running generate_future_forecast.py ...")
    run_command(f'"{sys.executable}" generate_future_forecast.py')

    # Start Streamlit dashboard
    streamlit_proc = run_command(f'"{sys.executable}" -m streamlit run src/dashboard/app.py', wait=False)

    # Start scheduler
    scheduler_proc = run_command(f'"{sys.executable}" src/utils/schedule_retrain.py', wait=False)

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
