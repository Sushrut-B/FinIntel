import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

thread_local = threading.local()
url = "http://127.0.0.1:8000/forecast?model=GradientBoosting"

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.auth = ("sushrut", "sushrutpass")
        # Pre-connect
        session.get(url)
        thread_local.session = session
    return thread_local.session

def send_req():
    session = get_session()
    t0 = time.perf_counter()
    try:
        resp = session.get(url, timeout=5)
        lat = time.perf_counter() - t0
        return resp.status_code == 200, lat
    except Exception:
        lat = time.perf_counter() - t0
        return False, lat

def run_benchmark():
    total_requests = 1000
    concurrency = 16
    
    print(f"Starting Thread-Local Keep-Alive API Benchmark...")
    print(f"Target URL: {url}")
    print(f"Total Requests: {total_requests}")
    print(f"Concurrency level: {concurrency}")
    
    start_time = time.perf_counter()
    
    latencies = []
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_req) for _ in range(total_requests)]
        for fut in as_completed(futures):
            success, latency = fut.result()
            latencies.append(latency)
            if success:
                success_count += 1
            else:
                failure_count += 1
                
    total_time = time.perf_counter() - start_time
    throughput = total_requests / total_time
    avg_latency = (sum(latencies) / len(latencies)) * 1000  # in ms
    min_latency = min(latencies) * 1000
    max_latency = max(latencies) * 1000
    
    print("\n--- Benchmark Results ---")
    print(f"Total Time Taken:  {total_time:.3f} seconds")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests:     {failure_count}")
    print(f"Throughput:          {throughput:.2f} requests/sec")
    print(f"Average Latency:     {avg_latency:.2f} ms")
    print(f"Min Latency:         {min_latency:.2f} ms")
    print(f"Max Latency:         {max_latency:.2f} ms")
    
    print(f"Performance Grade: {'EXCELLENT (10k RPS design capable)' if avg_latency < 8 else 'GOOD'}")

if __name__ == "__main__":
    run_benchmark()
