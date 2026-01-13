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

## 🔍 Monitoring (Bonus)

The API includes a built-in monitoring dashboard:

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/dashboard` | Visual monitoring dashboard (auto-refresh) |
| `/stats` | JSON statistics |
| `/metrics` | Prometheus metrics |

### Metrics Tracked

- **Total Requests**: All API calls
- **Total Predictions**: Malignant vs Benign counts
- **Errors**: Error count by type
- **Latency**: Average response time in milliseconds

### Access Dashboard
```bash
# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Open in browser
http://localhost:8000/dashboard
```

## 🔄 Automatic Retraining (Bonus)

The project includes automatic model retraining capabilities.

### Commands
```bash
# Manual retrain
python src/retrain.py retrain

# Check if retrain is needed
python src/retrain.py check

# View retrain history
python src/retrain.py history

# Start scheduled retraining (every 60 minutes)
python src/retrain.py schedule --interval 60
```

### Configuration

Edit `configs/retrain_config.json`:
```json
{
    "min_accuracy_threshold": 0.90,
    "min_f1_threshold": 0.90,
    "retrain_interval_hours": 24,
    "auto_deploy": true,
    "max_retrain_per_day": 3
}
```

### Retrain Triggers

- **Scheduled**: Runs at configured interval
- **Performance**: When metrics drop below threshold
- **Manual**: On-demand retraining
