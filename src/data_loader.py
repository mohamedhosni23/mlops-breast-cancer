"""
Data Loader Module
==================
Handles loading and preprocessing the breast cancer dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(save_to_csv: bool = True) -> tuple:
    """Load the breast cancer dataset from scikit-learn."""
    data = load_breast_cancer()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    if save_to_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/breast_cancer.csv", index=False)
        print("✓ Data saved to data/breast_cancer.csv")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    return X, y, data.feature_names, data.target_names


def preprocess_data(X: pd.DataFrame, y: pd.Series, config: dict) -> tuple:
    """Preprocess the data: split and scale."""
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    scale_features = config["data"]["scale_features"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing")
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        print("✓ Features scaled using StandardScaler")
    
    return X_train, X_test, y_train, y_test, scaler


def print_data_summary(X: pd.DataFrame, y: pd.Series, target_names: list):
    """Print a summary of the dataset."""
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nClass distribution:")
    for idx, name in enumerate(target_names):
        count = (y == idx).sum()
        percentage = count / len(y) * 100
        print(f"  - {name}: {count} ({percentage:.1f}%)")
    print("="*50 + "\n")


if __name__ == "__main__":
    config = load_config()
    X, y, feature_names, target_names = load_data(save_to_csv=True)
    print_data_summary(X, y, target_names)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, config)
    print("✓ Data loader test complete!")
