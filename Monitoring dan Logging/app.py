import os
import skops.io as sio
import pandas as pd
import json  # TAMBAHAN: Untuk menulis metrics.json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="Heart Disease Classification API", 
    description="API untuk memprediksi risiko penyakit jantung (Tingkat Basic)",
    version="1.0"
)

# 2. Tentukan Jalur File Model
MODEL_PATH = "model.skops"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File model tidak ditemukan di: {os.path.abspath(MODEL_PATH)}")

try:
    untrusted_types = sio.get_untrusted_types(file=MODEL_PATH)
    model = sio.load(MODEL_PATH, trusted=untrusted_types)
except Exception as e:
    raise RuntimeError(f"Gagal memuat file model.skops: {str(e)}")

# 3. Definisikan Skema Input Data Pasien menggunakan Pydantic
class PatientData(BaseModel):
    age: int = Field(..., description="Usia pasien", example=52)
    sex: int = Field(..., description="Jenis kelamin (1 = pria, 0 = wanita)", example=1)
    cp: int = Field(..., description="Tipe nyeri dada (0-3)", example=0)
    trestbps: int = Field(..., description="Tekanan darah istirahat", example=125)
    chol: int = Field(..., description="Kolesterol serum", example=212)
    fbs: int = Field(..., description="Gula darah puasa > 120 mg/dl (1 = true, 0 = false)", example=0)
    restecg: int = Field(..., description="Hasil elektrokardiografi istirahat (0-2)", example=1)
    thalach: int = Field(..., description="Detak jantung maksimum yang dicapai", example=168)
    exang: int = Field(..., description="Angina akibat olahraga (1 = yes, 0 = no)", example=0)
    oldpeak: float = Field(..., description="Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat", example=1.0)
    slope: int = Field(..., description="Kemiringan segmen ST latihan puncak (0-2)", example=2)
    ca: int = Field(..., description="Jumlah pembuluh darah utama (0-4) yang diwarnai dengan fluoroskopi", example=2)
    thal: int = Field(..., description="Hasil tes thalasemia (0-3)", example=3)

# 4. Endpoint Utama (Home)
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "API Klasifikasi Penyakit Jantung siap digunakan!"
    }

# 5. Endpoint untuk Prediksi
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Konversi input data Pydantic menjadi DataFrame Pandas
        input_df = pd.DataFrame([data.model_dump()])
        
        # Lakukan prediksi kelas (0 atau 1)
        prediction = model.predict(input_df)
        
        # Hitung probabilitas prediksi jika model mendukungnya
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[:, 1][0]
        else:
            probability = 1.0 if prediction[0] == 1 else 0.0
            
        # -------------------------------------------------------------
        # 🛠️ TAMBAHAN: SIMULASI AKURASI DINAMIS UNTUK PROMETHEUS 
        # -------------------------------------------------------------
        # Kita buat akurasi bergerak dinamis di kisaran probabilitas model saat ini
        # Ini akan membuat grafik akurasi di Grafana naik-turun secara riil!
        dynamic_accuracy = float(round(0.80 + (probability * 0.15), 2))
        
        # Tulis nilai ini ke file metrics.json di folder yang sama
        with open("metrics.json", "w") as f:
            json.dump({"accuracy": dynamic_accuracy}, f)
        # -------------------------------------------------------------
        
        # Tentukan status berdasarkan hasil prediksi
        result_status = "Jantung Bermasalah (Risiko Tinggi)" if prediction[0] == 1 else "Jantung Sehat (Risiko Rendah)"
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "status": result_status,
            "simulated_accuracy": dynamic_accuracy
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan pada server: {str(e)}"
        )
