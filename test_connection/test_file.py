import sys

import requests

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:7860"

data = {"name": "Huy", "scores": [9, 8.5, 10], "ok": True}

r = requests.post(f"{BASE}/json", json=data, timeout=60)
print("POST ->", r.status_code, r.text)
key = r.json()['id']

r = requests.get(f"{BASE}/json/{key}", timeout=60)
print("GET  ->", r.status_code, r.json())
print("Giống nhau:", r.json() == data)