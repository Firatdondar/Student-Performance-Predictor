# Student Performance Predictor & Analyzer

A Machine Learning pipeline designed to clean, process, and analyze student performance metrics to predict behavioral patterns such as excessive screen time.



## 📌 Project Overview
This data science project processes a dataset of student metrics (academic scores, attendance, study time, screen time, etc.). It performs comprehensive data cleaning, feature engineering, and deploys classification models to identify students with high screen time based on their academic and lifestyle habits.

## 🛠️ Data Preprocessing & Engineering
The pipeline strictly adheres to the following steps:
1. **Imputation:** Missing numerical values are handled via mean imputation to maintain statistical significance.
2. **Outlier Detection:** The robust 3-Sigma Rule (Standard Deviation) is applied across all numeric features to filter out extreme outliers.
3. **Normalization:** Features are scaled using `MinMaxScaler` to ensure models like KNN calculate distance metrics accurately.
4. **Feature Extraction:** * `Performans_Skoru` (Weighted academic score)
   * `Yorgunluk_Indeksi` (Fatigue index combining screen time, sleepless nights, and social isolation)
   * `Fazla_Ekran` (Binary target variable for classification)

## 🧠 Machine Learning Models
The project implements and evaluates two distinct classification algorithms to predict the `Fazla_Ekran` (High Screen Time) target:
* **Decision Tree Classifier**
* **K-Nearest Neighbors (KNN)**

## 🚀 Installation & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/StudentPerformancePredictor.git](https://github.com/your-username/StudentPerformancePredictor.git)
   cd StudentPerformancePredictor