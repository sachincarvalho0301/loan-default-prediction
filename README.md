# Loan Default Prediction (MLOps Project)

## 🚀 Overview
This project predicts the likelihood of loan default using machine learning.

It includes a complete MLOps pipeline:
- Data preprocessing
- Model training
- MLflow tracking
- SHAP explainability
- Drift detection (PSI)
- Docker deployment

---

## ⚙️ Features
- Data preprocessing (missing values, encoding, scaling)
- Handling class imbalance (SMOTE / class_weight)
- Model training (RandomForest / XGBoost)
- MLflow tracking (parameters, metrics, models)
- SHAP explainability
- Drift detection using PSI
- Docker support

---

## 📊 MLflow Tracking

Run MLflow UI:
```bash
mlflow ui --host 127.0.0.1 --port 5001