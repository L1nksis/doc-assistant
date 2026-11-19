import requests

BASE_URL = "http://127.0.0.1:8000"

r_health = requests.get(f"{BASE_URL}/health")
print("Health:", r_health.status_code, r_health.json())

r = requests.post(
    f"{BASE_URL}/ask",
    json={"question": "How can I cancel my order?", "k": 3}
)
print("Ask:", r.status_code, r.json())

r_err = requests.post(
    f"{BASE_URL}/ask",
    json={"question": "   ", "k": 3}
)
print("Empty check:", r_err.status_code, r_err.json())
