import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def preprocess_and_engineer_data(input_path, output_path):
    print("Initializing Data Pipeline...")
    
    # Load raw data
    df = pd.read_csv(input_path, sep="\t")
    print(f"Loaded {df.shape[0]} records.")

    # Handle missing values by filling with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Outlier removal using the 3-sigma rule
    for col in df.select_dtypes(include="number").columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
    
    # Normalization (Min-Max Scaling)
    cols_to_scale = ["Matematik_Puani", "Okuma_Puani", "Yazma_Puani", 
                     "Devam_Orani", "Calisma_Suresi", "Sosyal_Ortam_Puani", 
                     "Ekran_Suresi", "Uykusuz_Gece_Sayisi", "Yas"]
    
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Feature Engineering
    df["Performans_Skoru"] = (df["Matematik_Puani"] * 0.4 + 
                              df["Okuma_Puani"] * 0.3 + 
                              df["Yazma_Puani"] * 0.3)
    
    df["Fazla_Ekran"] = (df["Ekran_Suresi"] > df["Ekran_Suresi"].median()).astype(int)
    
    df["Yorgunluk_Indeksi"] = (df["Uykusuz_Gece_Sayisi"] * 1.5 + 
                               df["Ekran_Suresi"] * 1.2 - 
                               df["Sosyal_Ortam_Puani"] * 0.5)

    # Save processed data
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Processed data saved to {output_path}")
    return df

def train_and_evaluate_models(df):
    print("\n--- Model Training & Evaluation ---")
    
    if "Fazla_Ekran" not in df.columns:
        print("Target variable 'Fazla_Ekran' not found.")
        return

    X = df.drop(columns=["Fazla_Ekran", "Ogrenci_No"], errors="ignore")
    y = df["Fazla_Ekran"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    print("\n[Decision Tree Classifier]")
    print(f"Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)
    print("\n[K-Nearest Neighbors Classifier]")
    print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")

if __name__ == "__main__":
    # Dinamik dosya yolları (kodun nerede çalıştığından bağımsız)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, "data", "raw_data.tsv")
    processed_data_path = os.path.join(base_dir, "data", "processed_data.tsv")

    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data not found at {raw_data_path}")
    else:
        processed_df = preprocess_and_engineer_data(raw_data_path, processed_data_path)
        train_and_evaluate_models(processed_df)