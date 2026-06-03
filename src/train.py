# src/train.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_csv("../data/processed/customer_features.csv")
    # Separate features and target
    X = df.drop(["is_high_risk"], axis=1)
    y = df["is_high_risk"]
    return X, y

def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }
    return metrics

def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            for metric_name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(metric_name, value)
            
            mlflow.sklearn.log_model(model, "model")
            
            results[name] = {
                "model": model,
                "metrics": metrics,
                "report": classification_report(y_test, y_pred)
            }
            
            print(f"{name} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    return results, X_train.columns.tolist()

if __name__ == "__main__":
    mlflow.set_experiment("Credit_Risk_Modeling")
    results, feature_names = train_models()
    print("\n✅ Training completed!")