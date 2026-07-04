import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay
)

def train_model():
    # 1. Set nama eksperimen di MLflow - GUNAKAN ENVIRONMENT VARIABLE JIKA ADA
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Heart_Disease_Classification")
    mlflow.set_experiment(experiment_name)
    
    # 2. Mulai run MLflow
    with mlflow.start_run(run_name="Run_Classification"):
        print(f"Mulai melatih model dengan eksperimen: {experiment_name}...")
        
        # Path data preprocessing sesuai struktur folder lokal Anda saat ini
        data_dir = "heart_disease_preprocessing"
        
        # 3. Memuat data siap latih
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
        
        # 4. Inisialisasi Model & Hyperparameters
        n_estimators = 100
        max_depth = 5
        random_state = 42
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        
        # 5. Logging Parameters ke MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 6. Melatih Model
        model.fit(X_train, y_train)
        
        # 7. Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 8. Logging Metrics ke MLflow (Manual Logging)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # --- MEMBUAT & MENYIMPAN 2 ARTEFAK TAMBAHAN ---
        # Artefak 1: Classification Report (Teks)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
            
        # Artefak 2: Confusion Matrix (Gambar)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # --- LOGGING ARTEFAK KE MLFLOW/DAGSHUB ---
        mlflow.log_artifact(report_path) # Unggah file teks ke DagsHub
        mlflow.log_artifact(cm_path)     # Unggah file gambar ke DagsHub
        
        # 9. Logging Model Utama sebagai Artefak
        mlflow.sklearn.log_model(model, "heart_disease_model")
        
        # Hapus file lokal sementara agar folder kerja Anda tetap bersih
        if os.path.exists(report_path): 
            os.remove(report_path)
        if os.path.exists(cm_path): 
            os.remove(cm_path)
        
        print("✓ Model beserta 2 artefak tambahan berhasil dilatih dan diunggah!")
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    train_model()
