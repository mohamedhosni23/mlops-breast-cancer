# 🏥 MLOps Breast Cancer Classification

A complete MLOps pipeline for breast cancer classification using the UCI Breast Cancer dataset.

## 📋 Project Overview

| Component | Tool | Status |
|-----------|------|--------|
| Code Management | Git/GitHub | ✅ |
| Containerization | Docker | 🔄 |
| Data Versioning | DVC | 🔄 |
| Experiment Tracking | MLflow | ✅ |
| ML Pipeline | ZenML | 🔄 |
| Hyperparameter Tuning | Optuna | 🔄 |
| API Deployment | FastAPI | 🔄 |

## 🚀 Quick Start
```bash
# Clone repo
git clone https://github.com/mohamedhosni23/mlops-breast-cancer.git
cd mlops-breast-cancer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

# View MLflow dashboard
mlflow ui --port 5001
```

## 📊 Dataset

**UCI Breast Cancer Wisconsin (Diagnostic)**
- Samples: 569
- Features: 30
- Classes: Malignant / Benign
- Task: Binary Classification

## 📈 Results

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Random Forest | 95.61% | 96.55% | 99.39% |

## 👤 Author

MLOps Course Project 2025 - Dr. Salah Gontara
