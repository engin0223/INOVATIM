import numpy as np
import json
from datetime import datetime

def generate_mock_smartwatch_data(fs=360, win_len_sec=1):
    """
    Generates a mock ECG + HR + SpO2 JSON payload like a smartwatch.
    fs: sampling frequency (Hz)
    win_len_sec: length of each window in seconds
    """
    win_len = int(fs * win_len_sec)

    # Simulated ECG waveform (sinusoidal + noise)
    t = np.linspace(0, win_len_sec, win_len, endpoint=False)
    heartbeat_freq = 1.2  # Hz, ~72 bpm
    ecg_signal = 0.5 * np.sin(2 * np.pi * heartbeat_freq * t) + 0.05 * np.random.randn(win_len)

    # Simulated HR and SpO2
    heart_rate = np.random.randint(60, 100)
    spo2 = np.random.randint(95, 100)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "user_id": 1,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "features": ecg_signal.tolist(),  # this matches your /predict 'features'
        "model_type": "arrhythmia",
        "device_info": "mock_smartwatch"
    }

    return payload

# Generate example
mock_data = generate_mock_smartwatch_data()
print(json.dumps(mock_data, indent=2))
