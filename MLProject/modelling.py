import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import json
import platform
import shutil

mlflow.set_tracking_uri("https://dagshub.com/TimothyS710/Membangun_Sistem_ML.mlflow")
mlflow.set_experiment("Submission_Final_Workflow")

mlflow.sklearn.autolog(log_models=False)

file_name = 'credit_risk_preprocessing.csv'
folder_path = os.path.join('Membangun_model', file_name)

print("Searching for dataset...")

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
elif os.path.exists(folder_path):
    df = pd.read_csv(folder_path)
else:
    print(f"Error: Dataset {file_name} not found.")
    exit()

if 'approved' in df.columns:
    target_col = 'approved'
elif 'loan_status' in df.columns:
    target_col = 'loan_status'
else:
    target_col = df.columns[-1]

y = df[target_col]
X = df.drop(target_col, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [50, 100]

print("Starting training...")

for n in estimators:
    with mlflow.start_run(run_name=f"Advanced_Model_RF_{n}"):
        print(f"Training with {n} trees...")
        
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {acc}")
        
        mlflow.log_metric("accuracy", acc)

        monitoring_data = {
            "model_name": "RandomForest_CreditRisk",
            "model_version": f"v_trees_{n}",
            "accuracy": acc,
            "python_version": platform.python_version(),
            "input_features": list(X.columns),
            "target_column": target_col
        }
        
        with open("metric_info.json", "w") as f:
            json.dump(monitoring_data, f, indent=4)
        
        mlflow.log_artifact("metric_info.json")
        
        # 1. Upload ke DagsHub (Tetap dilakukan agar Reviewer bisa lihat)
        print("Uploading to DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditRiskModel_Final"
        )
        
        # 2. SIMPAN LOKAL (Jalur Penyelamat untuk Docker)
        print("Saving locally for Docker...")
        local_path = "model_output"
        if os.path.exists(local_path):
            shutil.rmtree(local_path) # Hapus folder lama biar bersih
        
        mlflow.sklearn.save_model(model, local_path)
        print("✅ Model saved locally to 'model_output'")

print("Done.")