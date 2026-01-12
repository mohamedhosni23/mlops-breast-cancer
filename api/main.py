"""
FastAPI Application with Monitoring
====================================
Provides REST API endpoints with Prometheus metrics.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import time
from typing import List
from datetime import datetime
from collections import deque
import threading

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="MLOps project with monitoring - Breast cancer classification",
    version="2.0.0"
)

# ===========================================
# PROMETHEUS METRICS
# ===========================================

# Request counter
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

# Request latency histogram
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Prediction counter
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions',
    ['result']  # malignant or benign
)

# Error counter
ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of errors',
    ['error_type']
)

# Current model version gauge
MODEL_VERSION = Gauge(
    'model_version_info',
    'Current model version',
    ['version']
)

# ===========================================
# IN-MEMORY METRICS STORAGE (for dashboard)
# ===========================================

class MetricsStore:
    def __init__(self, max_size=1000):
        self.requests = deque(maxlen=max_size)
        self.predictions = deque(maxlen=max_size)
        self.errors = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        # Summary stats
        self.total_requests = 0
        self.total_predictions = 0
        self.total_errors = 0
        self.malignant_count = 0
        self.benign_count = 0
    
    def log_request(self, endpoint, latency, status):
        with self.lock:
            self.requests.append({
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'latency': latency,
                'status': status
            })
            self.total_requests += 1
    
    def log_prediction(self, result, probability):
        with self.lock:
            self.predictions.append({
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'probability': probability
            })
            self.total_predictions += 1
            if result == 'malignant':
                self.malignant_count += 1
            else:
                self.benign_count += 1
    
    def log_error(self, error_type, message):
        with self.lock:
            self.errors.append({
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': message
            })
            self.total_errors += 1
    
    def get_stats(self):
        with self.lock:
            recent_latencies = [r['latency'] for r in list(self.requests)[-100:]]
            avg_latency = np.mean(recent_latencies) if recent_latencies else 0
            
            return {
                'total_requests': self.total_requests,
                'total_predictions': self.total_predictions,
                'total_errors': self.total_errors,
                'malignant_predictions': self.malignant_count,
                'benign_predictions': self.benign_count,
                'avg_latency_ms': round(avg_latency * 1000, 2),
                'recent_requests': list(self.requests)[-10:],
                'recent_errors': list(self.errors)[-10:]
            }

metrics_store = MetricsStore()

# ===========================================
# MODEL LOADING
# ===========================================

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

def load_model():
    """Load the trained model and scaler."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact.get("scaler")

def get_model_version():
    """Get current model version from version_info.json."""
    try:
        import json
        with open("models/version_info.json", "r") as f:
            info = json.load(f)
            return info.get("current", "unknown")
    except:
        return "unknown"

model, scaler = load_model()
current_version = get_model_version()
MODEL_VERSION.labels(version=current_version).set(1)

# Feature names
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# ===========================================
# REQUEST/RESPONSE MODELS
# ===========================================

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_malignant: float
    probability_benign: float
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    total_requests: int
    total_errors: int

# ===========================================
# MIDDLEWARE FOR METRICS
# ===========================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        
        # Log metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        
        metrics_store.log_request(endpoint, latency, response.status_code)
        
        return response
    
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        metrics_store.log_error(type(e).__name__, str(e))
        raise

# ===========================================
# API ENDPOINTS
# ===========================================

@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint with stats."""
    stats = metrics_store.get_stats()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=current_version,
        total_requests=stats['total_requests'],
        total_errors=stats['total_errors']
    )

@app.get("/info")
def model_info():
    """Get model information."""
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "model_version": current_version,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "classes": ["malignant", "benign"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make a prediction with latency tracking."""
    start_time = time.time()
    
    if model is None:
        ERROR_COUNT.labels(error_type="ModelNotLoaded").inc()
        metrics_store.log_error("ModelNotLoaded", "Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.features) != 30:
        ERROR_COUNT.labels(error_type="InvalidFeatures").inc()
        metrics_store.log_error("InvalidFeatures", f"Expected 30, got {len(request.features)}")
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features, got {len(request.features)}"
        )
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Scale if scaler exists
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Determine label
        label = "benign" if prediction == 1 else "malignant"
        
        # Log prediction metrics
        PREDICTION_COUNT.labels(result=label).inc()
        metrics_store.log_prediction(label, float(probabilities[prediction]))
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=label,
            probability_malignant=float(probabilities[0]),
            probability_benign=float(probabilities[1]),
            latency_ms=round(latency * 1000, 2)
        )
    
    except Exception as e:
        ERROR_COUNT.labels(error_type="PredictionError").inc()
        metrics_store.log_error("PredictionError", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/stats")
def get_stats():
    """Get detailed statistics."""
    return metrics_store.get_stats()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Simple HTML monitoring dashboard."""
    stats = metrics_store.get_stats()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLOps Monitoring Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #1a1a2e;
                color: #eee;
            }}
            h1 {{
                color: #00d4ff;
                border-bottom: 2px solid #00d4ff;
                padding-bottom: 10px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #16213e;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                color: #00d4ff;
            }}
            .metric-label {{
                color: #888;
                margin-top: 5px;
            }}
            .success {{ color: #00ff88; }}
            .warning {{ color: #ffcc00; }}
            .error {{ color: #ff4444; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #333;
            }}
            th {{
                background: #16213e;
                color: #00d4ff;
            }}
            .status-ok {{ color: #00ff88; }}
            .status-error {{ color: #ff4444; }}
        </style>
    </head>
    <body>
        <h1>🔍 MLOps Monitoring Dashboard</h1>
        <p>Model Version: <strong>{current_version}</strong> | Auto-refresh: 5s</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value success">{stats['total_requests']}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['total_predictions']}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value error">{stats['total_errors']}</div>
                <div class="metric-label">Total Errors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value warning">{stats['avg_latency_ms']} ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" style="color: #ff6b6b;">{stats['malignant_predictions']}</div>
                <div class="metric-label">Malignant Predictions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #51cf66;">{stats['benign_predictions']}</div>
                <div class="metric-label">Benign Predictions</div>
            </div>
        </div>
        
        <h2>📋 Recent Requests</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Endpoint</th>
                <th>Latency</th>
                <th>Status</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{r['timestamp'][:19]}</td>
                <td>{r['endpoint']}</td>
                <td>{round(r['latency']*1000, 2)} ms</td>
                <td class="{'status-ok' if r['status'] == 200 else 'status-error'}">{r['status']}</td>
            </tr>
            ''' for r in reversed(stats['recent_requests'][-5:]))}
        </table>
        
        <h2>⚠️ Recent Errors</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Type</th>
                <th>Message</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{e['timestamp'][:19]}</td>
                <td class="status-error">{e['type']}</td>
                <td>{e['message'][:50]}</td>
            </tr>
            ''' for e in reversed(stats['recent_errors'][-5:])) if stats['recent_errors'] else '<tr><td colspan="3">No errors! 🎉</td></tr>'}
        </table>
        
        <p style="margin-top: 30px; color: #666;">
            Endpoints: 
            <a href="/docs" style="color: #00d4ff;">/docs</a> | 
            <a href="/stats" style="color: #00d4ff;">/stats</a> | 
            <a href="/metrics" style="color: #00d4ff;">/metrics</a>
        </p>
    </body>
    </html>
    """
    return html

@app.get("/features")
def get_features():
    """Get list of required features."""
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}
