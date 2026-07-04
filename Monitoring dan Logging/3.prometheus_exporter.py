import time
import random
import os
import json
from prometheus_client import start_http_server, Counter, Gauge

# 1. Definisikan metrik yang akan dipantau
REQUEST_COUNT = Counter('api_requests_total', 'Total jumlah request ke API Prediksi Penyakit Jantung', ['method', 'endpoint', 'http_status'])
MODEL_ACCURACY = Gauge('model_accuracy_v1', 'Akurasi model aktif yang di-deploy')

if __name__ == '__main__':
    # Jalankan server exporter di port 8001
    start_http_server(8001)
    print("Prometheus Exporter berjalan di port 8001...")
    
    # Jalur file log metrik dinamis
    METRICS_FILE = "metrics.json"
    
    # Simulasi pengumpulan data metrik transaksi API secara kontinu
    while True:
        # FIX PERBAIKAN: Membaca nilai akurasi secara dinamis dari metrics.json
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r") as f:
                    data = json.load(f)
                    current_accuracy = data.get("accuracy", 0.85)
                    MODEL_ACCURACY.set(current_accuracy)
            except Exception as e:
                print(f"Gagal membaca metrics.json: {e}")
        else:
            # Nilai default sebelum ada request masuk
            MODEL_ACCURACY.set(0.85)
            
        # Menyimulasikan variasi status code yang masuk ke endpoint /predict
        status_code = random.choice(['200', '200', '200', '422', '500'])
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=status_code).inc()
        
        time.sleep(random.uniform(1, 4))
