import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub
import os
import json
import platform
import shutil

#Konfigurasi Login DagsHub (PENTING AGAR TIDAK ERROR 403) ---
dagshub.init(repo_owner='TimothyS710', repo_name='Membangun_Sistem_ML', mlflow=True)

# Konfigurasi MLflow
mlflow.set_tracking_uri("https://dagshub.com/TimothyS710/Membangun_Sistem_ML.mlflow")
mlflow.set_experiment("Submission_Final_Workflow")


mlflow.sklearn.autolog()

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'credit_risk_preprocessing.csv')

print("Searching for dataset...")

#Load Dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset ditemukan di: {dataset_path}")
else:
    fallback_path = 'credit_risk_preprocessing.csv'
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        print("✅ Dataset ditemukan di folder aktif!")
    else:
        print(f"❌ ERROR: Dataset tidak ditemukan di {dataset_path}")
        exit()

# Target Kolom
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

#Loop Training
for n in estimators:
    with mlflow.start_run(run_name=f"Advanced_Model_RF_{n}"):
        print(f"Training with {n} trees...")
        
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {acc}")

        monitoring_data = {
            "model_name": "RandomForest_CreditRisk",
            "model_version": f"v_trees_{n}",
            "accuracy": acc,
            "python_version": platform.python_version(),
            "input_features": list(X.columns),
            "target_column": target_col
        }
        
        json_path = "metric_info.json"
        with open(json_path, "w") as f:
            json.dump(monitoring_data, f, indent=4)
        
        mlflow.log_artifact(json_path)
        
        print("Saving locally for Docker...")
        local_path = os.path.join(script_dir, "model_output")
        
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        mlflow.sklearn.save_model(model, local_path)
        print(f"✅ Model saved locally to '{local_path}'")

print("Done.")