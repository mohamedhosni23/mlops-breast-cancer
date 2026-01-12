"""
ZenML Pipeline Steps
====================
Individual steps for the ML pipeline.
"""

import pandas as pd
import numpy as np
from zenml import step
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import mlflow
from typing import Tuple, Annotated
import os


@step
def load_data() -> Tuple[
    Annotated[pd.DataFrame, "X"],
    Annotated[pd.Series, "y"]
]:
    """Load the breast cancer dataset."""
    print("📂 Loading data...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


@step
def preprocess_data(
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[str, "scaler_path"]
]:
    """Split and scale the data."""
    print("⚙️ Preprocessing data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    scaler_path = "models/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler_path


@step
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10
) -> RandomForestClassifier:
    """Train the Random Forest model."""
    print("🏋️ Training model...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained with {n_estimators} trees, max_depth={max_depth}")
    return model


@step
def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "f1"],
    Annotated[float, "roc_auc"]
]:
    """Evaluate the model."""
    print("📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1 Score: {f1:.4f}")
    print(f"✓ ROC AUC: {roc_auc:.4f}")
    
    return accuracy, f1, roc_auc


@step
def save_model(
    model: RandomForestClassifier,
    scaler_path: str,
    accuracy: float
) -> str:
    """Save the trained model."""
    print("💾 Saving model...")
    
    model_path = "models/model.joblib"
    scaler = joblib.load(scaler_path)
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy
    }
    joblib.dump(artifact, model_path)
    
    print(f"✓ Model saved to {model_path}")
    return model_path
