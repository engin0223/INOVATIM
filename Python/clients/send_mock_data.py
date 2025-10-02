import requests
from create_mock_data import generate_mock_smartwatch_data  # your function above

api_url = "http://127.0.0.1:8000/predict"  # replace with your server URL

# Generate mock data
payload = generate_mock_smartwatch_data()

# Send to the server
try:
    response = requests.post(api_url, json=payload, timeout=5)
    if response.status_code == 200:
        print("Server response:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print("Request failed:", e)
