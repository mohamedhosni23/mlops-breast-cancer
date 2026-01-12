"""
FastAPI Application for Model Serving
=====================================
Provides REST API endpoints for breast cancer prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="MLOps project - Breast cancer classification model",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

def load_model():
    """Load the trained model and scaler."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["scaler"]

model, scaler = load_model()

# Feature names for the breast cancer dataset
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

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
                            0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                            8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                            0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_malignant: float
    probability_benign: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# Endpoints
@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/info")
def model_info():
    """Get model information."""
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "classes": ["malignant", "benign"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.features) != 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 30 features, got {len(request.features)}"
        )
    
    # Prepare features
    features = np.array(request.features).reshape(1, -1)
    
    # Scale features if scaler exists
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return PredictionResponse(
        prediction=int(prediction),
        prediction_label="benign" if prediction == 1 else "malignant",
        probability_malignant=float(probabilities[0]),
        probability_benign=float(probabilities[1])
    )

@app.get("/features")
def get_features():
    """Get list of required features."""
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}
