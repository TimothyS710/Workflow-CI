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

#  KONFIGURASI MLFLOW 
mlflow.set_tracking_uri("https://dagshub.com/TimothyS710/Membangun_Sistem_ML.mlflow")
mlflow.set_experiment("Submission_Final_Workflow")

mlflow.sklearn.autolog()

#  SETUP PATH DATASET 
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'credit_risk_preprocessing.csv')

# Cek dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    fallback_path = 'credit_risk_preprocessing.csv'
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
    else:
        print("Dataset not found!")
        exit()

#  PREPROCESSING 
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

#  TRAINING LOOP 
for n in estimators:
    with mlflow.start_run(run_name=f"Advanced_Model_RF_{n}"):
        
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train) 
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {acc}")

        # JSON Info
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
        
        
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            workspace_dir = os.environ.get('GITHUB_WORKSPACE')
            local_path = os.path.join(workspace_dir, "Workflow-CI/MLProject/model_output")
            print(f"🔧 Mode CI/CD: Menyimpan model ke {local_path}")
        else:
            
            local_path = os.path.join(script_dir, "model_output")
            print(f"💻 Mode Lokal: Menyimpan model ke {local_path}")

        # Bersihkan folder lama & Simpan
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        mlflow.sklearn.save_model(model, local_path)
        
        
        mlflow.log_artifacts(local_path, artifact_path="model")