"""
Optuna Hyperparameter Optimization
==================================
Automatically finds the best hyperparameters for our model.
"""

import optuna
from optuna.integration import MLflowCallback
import mlflow
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# Load and prepare data once
def prepare_data():
    """Load and preprocess data."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Global data (loaded once)
X_train, X_test, y_train, y_test, scaler = prepare_data()


def objective(trial):
    """
    Optuna objective function.
    Defines the hyperparameter search space and evaluates each trial.
    """
    # Define hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    # Create model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    return cv_scores.mean()


def run_optimization(n_trials=10):
    """Run Optuna optimization study."""
    
    print("="*60)
    print("🎯 STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Number of trials: {n_trials}")
    print("="*60 + "\n")
    
    # Set up MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("optuna-optimization")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # Maximize F1 score
        study_name='breast-cancer-rf-optimization'
    )
    
    # MLflow callback for logging
    mlflow_callback = MLflowCallback(
        tracking_uri="mlruns",
        metric_name="f1_score"
    )
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best F1 Score (CV): {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  - {param}: {value}")
    print("="*60)
    
    # Train final model with best parameters
    print("\n🏆 Training final model with best parameters...")
    
    best_model = RandomForestClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📈 Test Set Results:")
    print(f"  - Accuracy: {test_accuracy:.4f}")
    print(f"  - F1 Score: {test_f1:.4f}")
    print(f"  - ROC AUC:  {test_roc_auc:.4f}")
    
    # Log best model to MLflow
    with mlflow.start_run(run_name="optuna_best_model"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_f1_score", study.best_value)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.sklearn.log_model(best_model, "model")
    
    # Save best model
    model_path = "models/model_optimized.joblib"
    artifact = {
        "model": best_model,
        "scaler": scaler,
        "best_params": study.best_params,
        "test_f1_score": test_f1
    }
    joblib.dump(artifact, model_path)
    print(f"\n✓ Best model saved to {model_path}")
    
    # Save optimization history
    history_df = study.trials_dataframe()
    history_df.to_csv("artifacts/optuna_history.csv", index=False)
    print("✓ Optimization history saved to artifacts/optuna_history.csv")
    
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*60)
    
    return study, best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Optuna optimization")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
    args = parser.parse_args()
    
    study, model = run_optimization(n_trials=args.n_trials)
