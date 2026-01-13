"""
Automatic Retraining Module
===========================
Handles scheduled and conditional model retraining.
"""

import os
import sys
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RetrainManager:
    """Manages model retraining logic."""
    
    def __init__(self, config_path="configs/retrain_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.history_path = "models/retrain_history.json"
        
    def load_config(self):
        """Load retrain configuration."""
        default_config = {
            "min_accuracy_threshold": 0.90,
            "min_f1_threshold": 0.90,
            "retrain_interval_hours": 24,
            "auto_deploy": False,
            "max_retrain_per_day": 3
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def load_history(self):
        """Load retrain history."""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return {"retrains": [], "last_retrain": None}
    
    def save_history(self, history):
        """Save retrain history."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def should_retrain(self, current_metrics=None):
        """
        Determine if retraining is needed.
        
        Conditions:
        1. Performance below threshold
        2. Time since last retrain exceeds interval
        3. Manual trigger
        """
        history = self.load_history()
        reasons = []
        
        # Check performance threshold
        if current_metrics:
            if current_metrics.get('accuracy', 1.0) < self.config['min_accuracy_threshold']:
                reasons.append(f"Accuracy {current_metrics['accuracy']:.4f} below threshold {self.config['min_accuracy_threshold']}")
            if current_metrics.get('f1_score', 1.0) < self.config['min_f1_threshold']:
                reasons.append(f"F1 Score {current_metrics['f1_score']:.4f} below threshold {self.config['min_f1_threshold']}")
        
        # Check time interval
        if history['last_retrain']:
            last_retrain = datetime.fromisoformat(history['last_retrain'])
            hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
            if hours_since >= self.config['retrain_interval_hours']:
                reasons.append(f"Time since last retrain: {hours_since:.1f} hours")
        else:
            reasons.append("No previous retrain recorded")
        
        # Check daily limit
        today_retrains = sum(
            1 for r in history['retrains']
            if r['timestamp'][:10] == datetime.now().strftime('%Y-%m-%d')
        )
        if today_retrains >= self.config['max_retrain_per_day']:
            return False, ["Daily retrain limit reached"]
        
        return len(reasons) > 0, reasons
    
    def retrain(self, reason="scheduled"):
        """
        Perform model retraining.
        
        Args:
            reason: Reason for retraining (scheduled, performance, manual)
        """
        print("\n" + "="*60)
        print("🔄 STARTING MODEL RETRAIN")
        print("="*60)
        print(f"Reason: {reason}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)
        
        # Set up MLflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("retrain-runs")
        
        with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log retrain metadata
            mlflow.set_tag("retrain_reason", reason)
            mlflow.set_tag("retrain_timestamp", datetime.now().isoformat())
            
            # Step 1: Load fresh data
            print("\n📂 Loading data...")
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)
            
            # In real scenario, you might load new data here
            # X_new, y_new = load_new_data()
            
            # Step 2: Preprocess
            print("⚙️ Preprocessing...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Step 3: Train new model
            print("🏋️ Training new model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Step 4: Evaluate
            print("📊 Evaluating...")
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            print(f"\n📈 New Model Metrics:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
            print(f"   ROC AUC:  {metrics['roc_auc']:.4f}")
            
            # Log metrics to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            
            # Step 5: Compare with current model
            current_model_path = "models/model.joblib"
            deploy_new = False
            
            if os.path.exists(current_model_path):
                current_artifact = joblib.load(current_model_path)
                current_model = current_artifact['model']
                current_scaler = current_artifact.get('scaler')
                
                if current_scaler:
                    X_test_current = current_scaler.transform(X_test)
                else:
                    X_test_current = X_test_scaled
                
                current_pred = current_model.predict(X_test_current)
                current_f1 = f1_score(y_test, current_pred)
                
                print(f"\n📊 Comparison:")
                print(f"   Current model F1: {current_f1:.4f}")
                print(f"   New model F1:     {metrics['f1_score']:.4f}")
                
                if metrics['f1_score'] >= current_f1:
                    deploy_new = True
                    print("   ✅ New model is better or equal")
                else:
                    print("   ⚠️ Current model is better, keeping it")
            else:
                deploy_new = True
                print("\n✅ No current model found, will deploy new model")
            
            # Step 6: Save new model
            new_model_path = f"models/model_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            artifact = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(artifact, new_model_path)
            print(f"\n💾 New model saved to {new_model_path}")
            
            # Step 7: Deploy if better
            if deploy_new and self.config.get('auto_deploy', False):
                print("\n🚀 Auto-deploying new model...")
                joblib.dump(artifact, current_model_path)
                print(f"✅ Deployed to {current_model_path}")
                mlflow.set_tag("deployed", "true")
            else:
                mlflow.set_tag("deployed", "false")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Update history
            history = self.load_history()
            history['retrains'].append({
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'metrics': metrics,
                'deployed': deploy_new and self.config.get('auto_deploy', False),
                'model_path': new_model_path
            })
            history['last_retrain'] = datetime.now().isoformat()
            self.save_history(history)
            
            print("\n" + "="*60)
            print("✅ RETRAIN COMPLETE!")
            print("="*60)
            
            return metrics, deploy_new


class RetrainScheduler:
    """Handles scheduled retraining."""
    
    def __init__(self):
        self.manager = RetrainManager()
        self.scheduler = None
    
    def scheduled_job(self):
        """Job that runs on schedule."""
        print(f"\n⏰ Scheduled retrain check at {datetime.now().isoformat()}")
        
        should_retrain, reasons = self.manager.should_retrain()
        
        if should_retrain:
            print(f"Retrain triggered: {reasons}")
            self.manager.retrain(reason="scheduled")
        else:
            print(f"Retrain skipped: {reasons}")
    
    def start(self, interval_minutes=60):
        """Start the scheduler."""
        from apscheduler.schedulers.background import BackgroundScheduler
        
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.scheduled_job,
            'interval',
            minutes=interval_minutes,
            id='retrain_job',
            name='Scheduled Model Retrain'
        )
        self.scheduler.start()
        
        print(f"🕐 Retrain scheduler started (interval: {interval_minutes} minutes)")
        return self.scheduler
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown()
            print("🛑 Retrain scheduler stopped")


def run_manual_retrain():
    """Run a manual retrain."""
    manager = RetrainManager()
    return manager.retrain(reason="manual")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Retrain Manager")
    parser.add_argument("action", choices=["retrain", "check", "history", "schedule"])
    parser.add_argument("--interval", type=int, default=60, help="Schedule interval in minutes")
    
    args = parser.parse_args()
    
    manager = RetrainManager()
    
    if args.action == "retrain":
        manager.retrain(reason="manual")
        
    elif args.action == "check":
        should_retrain, reasons = manager.should_retrain()
        print(f"\nShould retrain: {should_retrain}")
        print(f"Reasons: {reasons}")
        
    elif args.action == "history":
        history = manager.load_history()
        print("\n" + "="*50)
        print("📋 RETRAIN HISTORY")
        print("="*50)
        if not history['retrains']:
            print("No retrains recorded yet.")
        else:
            for r in history['retrains'][-5:]:
                print(f"\n{r['timestamp'][:19]}")
                print(f"  Reason: {r['reason']}")
                print(f"  F1 Score: {r['metrics']['f1_score']:.4f}")
                print(f"  Deployed: {r['deployed']}")
        print("="*50)
        
    elif args.action == "schedule":
        print("Starting scheduled retraining...")
        print("Press Ctrl+C to stop\n")
        
        scheduler = RetrainScheduler()
        scheduler.start(interval_minutes=args.interval)
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
