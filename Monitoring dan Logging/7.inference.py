import requests
import time
import random

url = "http://127.0.0.1:5000/predict"

# Data dummy untuk looping kirim data
while True:
    data_pasien = {
        "age": random.randint(29, 77),
        "sex": random.choice([0, 1]),
        "cp": random.randint(0, 3),
        "trestbps": random.randint(100, 180),
        "chol": random.randint(126, 564),
        "fbs": random.choice([0, 1]),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(71, 202),
        "exang": random.choice([0, 1]),
        "oldpeak": round(random.uniform(0.0, 6.2), 1),
        "slope": random.randint(0, 2),
        "ca": random.randint(0, 4),
        "thal": random.randint(0, 3)
    }
    
    try:
        response = requests.post(url, json=data_pasien)
        print(f"Status: {response.status_code} | Response: {response.json()['status']}")
    except Exception as e:
        print(f"Gagal terhubung ke API: {e}")
        
    time.sleep(2) # Tembak setiap 2 detik
