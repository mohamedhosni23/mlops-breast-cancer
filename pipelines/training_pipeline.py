"""
ZenML Training Pipeline
=======================
Main pipeline that orchestrates all steps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zenml import pipeline
from steps import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model
)


@pipeline(name="breast_cancer_training_pipeline")
def training_pipeline(n_estimators: int = 100, max_depth: int = 10):
    """
    Complete training pipeline for breast cancer classification.
    """
    # Step 1: Load data
    X, y = load_data()
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler_path = preprocess_data(X, y)
    
    # Step 3: Train model
    model = train_model(X_train, y_train, n_estimators, max_depth)
    
    # Step 4: Evaluate model
    accuracy, f1, roc_auc = evaluate_model(model, X_test, y_test)
    
    # Step 5: Save model
    model_path = save_model(model, scaler_path, accuracy)
    
    return model_path


if __name__ == "__main__":
    print("="*60)
    print("🚀 STARTING ZENML PIPELINE - BASELINE")
    print("="*60)
    
    # Run baseline pipeline
    training_pipeline()
    
    print("="*60)
    print("✅ BASELINE PIPELINE COMPLETE!")
    print("="*60)
