"""
Training Module
===============
Main training script with MLflow integration.
"""

import os
import sys
import argparse
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_config, load_data, preprocess_data, print_data_summary
from src.evaluate import (
    calculate_metrics, print_metrics, plot_confusion_matrix, 
    plot_roc_curve, plot_feature_importance, get_classification_report
)


def get_model(config: dict, model_type: str = None):
    """Get model instance based on configuration."""
    model_type = model_type or config["model"]["type"]
    
    if model_type == "random_forest":
        params = config["model"]["random_forest"]
        model = RandomForestClassifier(**params)
    elif model_type == "svm":
        params = config["model"]["svm"]
        model = SVC(**params, probability=True)
    elif model_type == "logistic_regression":
        params = config["model"]["logistic_regression"]
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✓ Model created: {model_type}")
    return model, model_type


def train_model(model, X_train, y_train, config: dict) -> tuple:
    """Train the model with cross-validation."""
    cv_folds = config["training"]["cv_folds"]
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    print(f"✓ Cross-validation ({cv_folds} folds):")
    print(f"  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    model.fit(X_train, y_train)
    print("✓ Model trained on full training set")
    
    return model, cv_scores


def save_model(model, scaler, path: str = "models/model.joblib"):
    """Save trained model and scaler."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact = {"model": model, "scaler": scaler}
    joblib.dump(artifact, path)
    print(f"✓ Model saved to {path}")


def run_training(config_path: str = "configs/config.yaml", 
                 model_type: str = None, run_name: str = None) -> dict:
    """Run the complete training pipeline."""
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_type or config["model"]["type"]
        run_name = f"{model_name}_{timestamp}"
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING PIPELINE")
    print("="*60)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type or config["model"]["type"])
        
        # Step 1: Load Data
        print("\n📂 Step 1: Loading data...")
        X, y, feature_names, target_names = load_data(save_to_csv=True)
        print_data_summary(X, y, target_names)
        
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        
        # Step 2: Preprocess Data
        print("\n⚙️ Step 2: Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, config)
        
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("scale_features", config["data"]["scale_features"])
        
        # Step 3: Create Model
        print("\n🔧 Step 3: Creating model...")
        model, actual_model_type = get_model(config, model_type)
        
        model_params = config["model"][actual_model_type]
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Step 4: Train Model
        print("\n🏋️ Step 4: Training model...")
        model, cv_scores = train_model(model, X_train, y_train, config)
        
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        
        # Step 5: Evaluate Model
        print("\n📊 Step 5: Evaluating model...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, actual_model_type)
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        print("\nClassification Report:")
        print(get_classification_report(y_test, y_pred, list(target_names)))
        
        # Step 6: Generate Plots
        print("\n📈 Step 6: Generating plots...")
        os.makedirs("artifacts", exist_ok=True)
        
        plot_confusion_matrix(y_test, y_pred, 
                              save_path="artifacts/confusion_matrix.png",
                              target_names=list(target_names))
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        
        plot_roc_curve(y_test, y_prob, save_path="artifacts/roc_curve.png")
        mlflow.log_artifact("artifacts/roc_curve.png")
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names, top_n=10,
                                    save_path="artifacts/feature_importance.png")
            mlflow.log_artifact("artifacts/feature_importance.png")
        
        # Step 7: Save Model
        print("\n💾 Step 7: Saving model...")
        model_path = config["api"]["model_path"]
        save_model(model, scaler, model_path)
        
        mlflow.sklearn.log_model(model, "model")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved to: {model_path}")
        print("="*60 + "\n")
        
        return {
            "run_id": mlflow.active_run().info.run_id,
            "model_type": actual_model_type,
            "metrics": metrics,
            "model_path": model_path
        }


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer classification model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, choices=["random_forest", "svm", "logistic_regression"])
    parser.add_argument("--run-name", type=str)
    
    args = parser.parse_args()
    run_training(config_path=args.config, model_type=args.model, run_name=args.run_name)


if __name__ == "__main__":
    main()
