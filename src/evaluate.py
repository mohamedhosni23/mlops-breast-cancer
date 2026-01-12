"""
Evaluation Module
=================
Functions for evaluating ML models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Calculate all classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print(f"📈 EVALUATION RESULTS: {model_name}")
    print("="*50)
    for name, value in metrics.items():
        print(f"  {name.replace('_', ' ').title():.<20} {value:.4f}")
    print("="*50 + "\n")


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list = None) -> str:
    """Get detailed classification report."""
    if target_names is None:
        target_names = ["Malignant", "Benign"]
    return classification_report(y_true, y_pred, target_names=target_names)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          save_path: str = None, target_names: list = None) -> plt.Figure:
    """Plot and optionally save confusion matrix."""
    if target_names is None:
        target_names = ["Malignant", "Benign"]
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names, ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   save_path: str = None) -> plt.Figure:
    """Plot and optionally save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 10,
                           save_path: str = None) -> plt.Figure:
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print("⚠ Model doesn't have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    return fig
