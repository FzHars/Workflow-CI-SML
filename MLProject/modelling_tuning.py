import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_with_tuning():
    mlflow.set_experiment("Heart_Disease_Tuning")
    
    data_dir = "../preprocessing/heart_disease_preprocessing"
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # Kita coba beberapa kombinasi hyperparameter (Tuning)
    grid_search = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 7}
    ]
    
    for i, params in enumerate(grid_search):
        with mlflow.start_run(run_name=f"Run_Tuning_{i+1}"):
            print(f"Menjalankan Tuning ke-{i+1}: {params}")
            
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            
            # Log Parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            # Log Model
            mlflow.sklearn.log_model(model, f"model_run_{i+1}")
            print(f"✓ Run {i+1} Selesai. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_with_tuning()
