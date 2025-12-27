import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import pickle

#autentikasi dagshub
os.environ["MLFLOW_TRACKING_USERNAME"] = "KinoVelverika"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9dba75b0b97434ebba08b1bf99401f4cac3820cf"

#konek kedagshub
mlflow.set_tracking_uri("https://dagshub.com/KinoVelverika/Membangun_Sistem_ML.mlflow")
mlflow.set_experiment("Submission_Final_Workflow")

#autologging
mlflow.sklearn.autolog()

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
    print(f"Posisi terminal saat ini: {os.getcwd()}")
    exit()

# Memilih Fitur & Target
print(f"📋 Kolom tersedia: {list(df.columns)}")

if 'approved' in df.columns:
    target_col = 'approved'
elif 'loan_status' in df.columns:
    target_col = 'loan_status'
else:
    target_col = df.columns[-1] # Ambil kolom terakhir sebagai tebakan
    print(f"⚠️ Warning: Target tidak dikenali. Menggunakan kolom terakhir.")

print(f"🎯 Target kolom yang digunakan: {target_col}")

# Split Data
try:
    y = df[target_col]
    X = df.drop(target_col, axis=1)
except KeyError as e:
    print(f"❌ ERROR KEY: {e}. Pastikan nama kolom target benar-benar ada.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TRAINING LOOP
estimators = [50, 100]
last_model = None

print("Mulai training dengan Autolog...")

for n in estimators:
    with mlflow.start_run(run_name=f"Model_RF_{n}"):
        print(f"Training dengan {n} pohon...")
        
        # Melatih Model
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"    Selesai! Akurasi: {acc}")

        last_model = model

print("Semua proses training selesai!")

#simpan model
if last_model:
    filename = 'model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(last_model, file)
    print(f"\n✅ Model berhasil disimpan di laptop sebagai: {filename}")