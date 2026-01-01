import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import json
import platform

#konek kedagshub
mlflow.set_tracking_uri("https://dagshub.com/KinoVelverika/Membangun_Sistem_ML.mlflow")
mlflow.set_experiment("Submission_Final_Workflow")

#autologging
mlflow.sklearn.autolog(log_models=False)

#preprocessing data
file_name = 'credit_risk_preprocessing.csv'
folder_path = os.path.join('Membangun_model', file_name)

print("Mencari file dataset...")

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
    print(f"✅ File ditemukan di: {file_name}")
elif os.path.exists(folder_path):
    df = pd.read_csv(folder_path)
    print(f"✅ File ditemukan di: {folder_path}")
else:
    print(f"❌ ERROR: File '{file_name}' tidak ditemukan dimanapun.")
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

#TRAINING LOOP
estimators = [50, 100]

print("\nMulai training...")

for n in estimators:
    with mlflow.start_run(run_name=f"Advanced_Model_RF_{n}"):
        print(f"🔄 Training dengan {n} pohon...")
        
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"    ✅ Selesai! Akurasi: {acc}")
        
        mlflow.log_metric("accuracy", acc)

        # FILE MONITORING JSON (Tetap pertahankan logic Anda ini)
        print("    📊 Membuat monitoring json...")
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
        
        print("    📦 Membungkus artifacts model (Auto log_model)...")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  
            registered_model_name="CreditRiskModel_Final"
        )
        
        print("    ✅ Model berhasil disimpan ke DagsHub dengan format Docker-Ready.")

print("\n🎉 Selesai! Cek DagsHub.")