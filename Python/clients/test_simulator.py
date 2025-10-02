"""
test_simulator.py

Simple client that generates a synthetic 24-hour ECG-like signal and sends half-second windows to the API.
Modify API URL and user_id as needed.
"""
import numpy as np
import requests
import time

def run_simulation(api_url="http://127.0.0.1:8000/predict",
                   user_id=1,
                   hours=24,
                   fs=360,
                   win_len_sec=0.5,
                   batch_size=1000,
                   device_info="null_test_sim"):
    win_len = int(win_len_sec * fs)
    N = hours * 3600 * fs
    t = np.linspace(0, hours * 3600, N, endpoint=False)
    heartbeat_freq = 2  # 2 Hz-like base
    signal = 0.6 * np.sin(2 * np.pi * heartbeat_freq * t)
    signal += 0.05 * np.random.randn(N)

    num_windows = len(signal) // win_len
    windows = signal[:num_windows * win_len].reshape(num_windows, win_len)

    for i in range(0, num_windows, batch_size):
        batch = windows[i:i+batch_size]
        for w in batch:
            payload = {
                "user_id": user_id,
                "features": w.tolist(),
                "model_type": "arrhythmia",
                "device_info": device_info
            }
            try:
                r = requests.post(api_url, json=payload, timeout=5)
            except Exception as e:
                print(f"[simulator] Request error: {e}")
                time.sleep(0.1)
                continue
            if r.status_code != 200:
                print("[simulator] Error:", r.status_code, r.text)
            else:
                res = r.json()
                print(f"[simulator] Prediction ID {res.get('prediction_id')}, label: {res.get('predicted_label')}")
        # small pause to avoid hammering
        time.sleep(0.05)

if __name__ == "__main__":
    run_simulation()
